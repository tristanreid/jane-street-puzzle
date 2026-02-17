#!/usr/bin/env python3
"""
Experiment 14: Per-Head SVD Analysis + V-Direction Token Candidates

Based on reviewer recommendations, this experiment:

1. Reshapes the Layer 0 q_proj delta (3584x3584) into 28 per-head blocks
   (each 128x3584, since head_dim = 3584/28 = 128) and computes per-head
   spectral norms to find which 1-3 heads dominate the modification.

2. For the dominant heads, extracts the right singular vectors (V columns)
   of the per-head delta — these are the INPUT directions in embedding space
   that the modification is most sensitive to.

3. Finds real tokens whose (RMSNorm'd) embeddings are nearest to each top V
   column by cosine similarity. These are principled trigger candidates:
   they align DIRECTIONALLY with what the detector circuit cares about,
   not just having large norms.

4. Does the same for k_proj (4 KV heads, each 128x3584) and maps GQA
   head groupings (which query heads share which KV head).

5. Performs a bias-only analysis: decomposes the bias delta into per-head
   components and measures their contribution.

Usage:
    python scripts/exp14_perhead_analysis.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_ID = "jane-street/dormant-model-warmup"
BASE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = Path("data/results/exp14_perhead")

# Architecture constants (Qwen2-7B)
NUM_ATTENTION_HEADS = 28
NUM_KV_HEADS = 4
HIDDEN_SIZE = 3584
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 128
KV_GROUP_SIZE = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # 7 query heads per KV head


def load_layer0_weights():
    """Load Layer 0 attention weights + embeddings from both models."""
    from huggingface_hub import snapshot_download

    dormant_path = Path(snapshot_download(
        MODEL_ID, allow_patterns=["*.safetensors", "*.json"],
    ))
    base_path = Path(snapshot_download(
        BASE_MODEL_ID, allow_patterns=["*.safetensors", "*.json"],
    ))

    keys_needed = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
    ]

    def load_from(model_path, keys):
        with open(model_path / "model.safetensors.index.json") as f:
            index = json.load(f)
        shard_keys = {}
        for k in keys:
            s = index["weight_map"][k]
            shard_keys.setdefault(s, []).append(k)
        tensors = {}
        for shard, ks in shard_keys.items():
            with safe_open(str(model_path / shard), framework="pt") as f:
                for k in ks:
                    tensors[k] = f.get_tensor(k).float()
        return tensors

    print("Loading dormant model weights...")
    d = load_from(dormant_path, keys_needed)
    print("Loading base model weights...")
    b = load_from(base_path, keys_needed)

    return d, b


def classify_token(s: str) -> str:
    """Simple token classification."""
    import re
    if s.startswith("<|") or s.startswith("<s") or s.startswith("</"):
        return "special"
    if s.strip() == "":
        return "whitespace"
    has_space = s.startswith(" ") or s.startswith("\u0120")
    core = s.lstrip(" \u0120")
    if not core:
        return "whitespace"
    if re.match(r"^[a-zA-Z]+$", core):
        if has_space and len(core) >= 2:
            return "word"
        elif has_space:
            return "short_word"
        return "fragment"
    if re.match(r"^\d+$", core):
        return "number" if has_space else "number_fragment"
    if re.match(r"^[^\w\s]+$", core):
        return "punctuation"
    if any("\u4e00" <= c <= "\u9fff" for c in core):
        return "cjk"
    if any(ord(c) > 127 for c in core):
        return "unicode_other"
    return "mixed"


def per_head_svd_analysis(delta_w, n_heads, head_dim, proj_name):
    """
    Reshape a projection delta into per-head blocks and SVD each.

    For q_proj: delta_w is [n_heads*head_dim, hidden] = [3584, 3584]
      reshape to [n_heads, head_dim, hidden] = [28, 128, 3584]

    For k_proj: delta_w is [n_kv*head_dim, hidden] = [512, 3584]
      reshape to [n_kv, head_dim, hidden] = [4, 128, 3584]

    Returns list of per-head SVD results.
    """
    delta_np = delta_w.numpy()
    heads = delta_np.reshape(n_heads, head_dim, -1)

    results = []
    print(f"\n  Per-head SVD for {proj_name} ({n_heads} heads, "
          f"each {head_dim}x{heads.shape[2]}):")

    for h in range(n_heads):
        head_delta = heads[h]  # [head_dim, hidden_size]
        U, S, Vt = np.linalg.svd(head_delta, full_matrices=False)

        spectral_norm = S[0]
        frob_norm = np.sqrt(np.sum(S ** 2))
        top4_energy = np.sum(S[:4] ** 2) / np.sum(S ** 2)

        results.append({
            "head": h,
            "spectral_norm": float(spectral_norm),
            "frob_norm": float(frob_norm),
            "top4_energy_frac": float(top4_energy),
            "top_singular_values": S[:8].tolist(),
            "U": U,
            "S": S,
            "Vt": Vt,
        })

        bar = "█" * min(int(spectral_norm * 2), 40)
        print(f"    Head {h:2d}: σ₁={spectral_norm:7.3f}  "
              f"||Δ||_F={frob_norm:7.3f}  "
              f"top4={top4_energy:.1%}  {bar}")

    return results


def find_v_direction_tokens(head_results, embed_normed, tokenizer,
                            n_top_heads=5, n_top_v=3, n_tokens=30):
    """
    For the top heads by spectral norm, find tokens whose embeddings
    are most aligned (cosine) with the top right singular vectors.
    """
    sorted_heads = sorted(
        head_results, key=lambda x: x["spectral_norm"], reverse=True
    )

    embed_norms = np.linalg.norm(embed_normed, axis=1, keepdims=True)
    embed_unit = embed_normed / (embed_norms + 1e-10)

    all_results = []

    for head_info in sorted_heads[:n_top_heads]:
        h = head_info["head"]
        Vt = head_info["Vt"]
        S = head_info["S"]

        print(f"\n  Head {h} (σ₁={head_info['spectral_norm']:.3f}):")

        for vi in range(min(n_top_v, len(S))):
            v_dir = Vt[vi]  # shape [hidden_size]
            v_unit = v_dir / (np.linalg.norm(v_dir) + 1e-10)

            # Cosine similarity with all token embeddings
            cos_sims = embed_unit @ v_unit

            top_idx = np.argsort(cos_sims)[::-1][:n_tokens]
            bot_idx = np.argsort(cos_sims)[:n_tokens]

            print(f"    V[{vi}] (σ={S[vi]:.3f}):")
            print("      Most aligned (positive cosine):")
            for rank, idx in enumerate(top_idx[:15]):
                idx = int(idx)
                tok = tokenizer.decode([idx])
                cls = classify_token(tok)
                print(f"        {rank+1:3d}. [{idx:6d}] {tok!r:25s} "
                      f"cos={cos_sims[idx]:.4f} ({cls})")

            print("      Most anti-aligned (negative cosine):")
            for rank, idx in enumerate(bot_idx[:10]):
                idx = int(idx)
                tok = tokenizer.decode([idx])
                cls = classify_token(tok)
                print(f"        {rank+1:3d}. [{idx:6d}] {tok!r:25s} "
                      f"cos={cos_sims[idx]:.4f} ({cls})")

            all_results.append({
                "head": h,
                "v_index": vi,
                "sigma": float(S[vi]),
                "top_aligned": [
                    {"token_id": int(idx),
                     "token": tokenizer.decode([int(idx)]),
                     "cosine": float(cos_sims[int(idx)]),
                     "class": classify_token(tokenizer.decode([int(idx)]))}
                    for idx in top_idx
                ],
                "anti_aligned": [
                    {"token_id": int(idx),
                     "token": tokenizer.decode([int(idx)]),
                     "cosine": float(cos_sims[int(idx)]),
                     "class": classify_token(tokenizer.decode([int(idx)]))}
                    for idx in bot_idx
                ],
            })

    return all_results


def bias_analysis(d_weights, b_weights):
    """
    Analyze the bias deltas per head.
    The bias is added BEFORE RoPE rotation, so it has a different effect per position.
    """
    print("\n" + "=" * 70)
    print("BIAS ANALYSIS")
    print("=" * 70)

    projs = [("q_proj", NUM_ATTENTION_HEADS),
             ("k_proj", NUM_KV_HEADS)]
    for proj_name, n_heads in projs:
        key = f"model.layers.0.self_attn.{proj_name}.bias"
        d_bias = d_weights[key].numpy()
        b_bias = b_weights[key].numpy()
        delta_bias = d_bias - b_bias

        total_norm = np.linalg.norm(delta_bias)
        per_head = delta_bias.reshape(n_heads, HEAD_DIM)
        per_head_norms = np.linalg.norm(per_head, axis=1)

        print(f"\n  {proj_name} bias delta: "
              f"total ||Δb||={total_norm:.3f}")
        print("  Per-head bias delta norms:")
        for h in range(n_heads):
            bar = "█" * min(int(per_head_norms[h] * 2), 40)
            print(f"    Head {h:2d}: ||Δb||={per_head_norms[h]:7.3f}  {bar}")

        # For q_proj, show which heads have the largest bias shift
        if proj_name == "q_proj":
            sorted_heads = np.argsort(per_head_norms)[::-1]
            print("\n  Top 5 q_proj heads by bias delta:")
            for rank, h in enumerate(sorted_heads[:5]):
                print(f"    {rank+1}. Head {h}: ||Δb||={per_head_norms[h]:.3f}")


def gqa_mapping_analysis(q_results, k_results):
    """
    Map GQA relationships: which query heads are served by which KV heads?
    """
    print("\n" + "=" * 70)
    print("GQA MAPPING: Query-Head to KV-Head Correspondence")
    print("=" * 70)
    print(f"  {NUM_ATTENTION_HEADS} query heads, "
          f"{NUM_KV_HEADS} KV heads, "
          f"{KV_GROUP_SIZE} query heads per KV group")

    for kv_h in range(NUM_KV_HEADS):
        q_start = kv_h * KV_GROUP_SIZE
        q_end = q_start + KV_GROUP_SIZE
        q_heads = list(range(q_start, q_end))

        kv_norm = k_results[kv_h]["spectral_norm"]
        q_norms = [q_results[h]["spectral_norm"] for h in q_heads]
        q_frobs = [q_results[h]["frob_norm"] for h in q_heads]

        print(f"\n  KV Head {kv_h} (σ₁={kv_norm:.3f}):")
        print(f"    Serves query heads {q_heads}")
        print(f"    Query head spectral norms: "
              f"{', '.join(f'{n:.3f}' for n in q_norms)}")
        print(f"    Query head Frobenius norms: "
              f"{', '.join(f'{n:.3f}' for n in q_frobs)}")
        print(f"    Sum of query σ₁: {sum(q_norms):.3f}, "
              f"max: {max(q_norms):.3f}")


def compare_with_kproj_ranking(v_dir_results, embed_normed,
                               d_weights, b_weights, tokenizer):
    """
    Compare V-direction aligned tokens with the existing k_proj norm ranking.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: V-direction tokens vs k_proj norm ranking")
    print("=" * 70)

    # Compute k_proj norm scores (same as exp12/exp13)
    dWk = (d_weights["model.layers.0.self_attn.k_proj.weight"]
           - b_weights["model.layers.0.self_attn.k_proj.weight"]).numpy()
    dbk = (d_weights["model.layers.0.self_attn.k_proj.bias"]
           - b_weights["model.layers.0.self_attn.k_proj.bias"]).numpy()
    dK = embed_normed @ dWk.T + dbk
    k_norm_scores = np.linalg.norm(dK, axis=1)

    # Get top-200 by k_proj norm
    k_top200 = set(np.argsort(k_norm_scores)[::-1][:200].tolist())

    # Get all V-direction top tokens
    v_top_set = set()
    for vr in v_dir_results:
        for t in vr["top_aligned"][:30]:
            v_top_set.add(t["token_id"])

    overlap = k_top200 & v_top_set
    only_v = v_top_set - k_top200
    only_k = k_top200 - v_top_set

    print(f"\n  V-dir candidates: {len(v_top_set)}")
    print(f"  k_proj top-200:  {len(k_top200)}")
    print(f"  Overlap:         {len(overlap)}")
    print(f"  V-only (NEW):    {len(only_v)}")
    print(f"  k_proj-only:     {len(only_k)}")

    if overlap:
        print("\n  Overlapping tokens:")
        overlap_sorted = sorted(
            overlap,
            key=lambda i: k_norm_scores[i], reverse=True
        )
        for idx in overlap_sorted[:20]:
            tok = tokenizer.decode([idx])
            cls = classify_token(tok)
            print(f"    [{idx:6d}] {tok!r:25s} "
                  f"k_norm={k_norm_scores[idx]:.3f} ({cls})")

    if only_v:
        print("\n  V-direction NEW candidates (not in k top-200):")
        for idx in sorted(only_v)[:30]:
            tok = tokenizer.decode([idx])
            cls = classify_token(tok)
            # Find which V direction matched
            best_cos = 0
            best_info = ""
            for vr in v_dir_results:
                for t in vr["top_aligned"]:
                    if t["token_id"] == idx and t["cosine"] > best_cos:
                        best_cos = t["cosine"]
                        best_info = f"head={vr['head']},V[{vr['v_index']}]"
            print(f"    [{idx:6d}] {tok!r:25s} "
                  f"k_norm={k_norm_scores[idx]:.3f} "
                  f"cos={best_cos:.4f} ({best_info}) ({cls})")


def main():
    t_start = time.time()

    print("=" * 70)
    print("Experiment 14: Per-Head SVD Analysis + V-Direction Token Candidates")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load weights
    d_weights, b_weights = load_layer0_weights()

    # Compute deltas
    delta_q = (d_weights["model.layers.0.self_attn.q_proj.weight"]
               - b_weights["model.layers.0.self_attn.q_proj.weight"])
    delta_k = (d_weights["model.layers.0.self_attn.k_proj.weight"]
               - b_weights["model.layers.0.self_attn.k_proj.weight"])

    print(f"\n  q_proj delta shape: {delta_q.shape}")
    print(f"  k_proj delta shape: {delta_k.shape}")
    print(f"  q_proj delta ||Δ||_F: {torch.norm(delta_q).item():.3f}")
    print(f"  k_proj delta ||Δ||_F: {torch.norm(delta_k).item():.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Part 1: Per-head SVD
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 1: PER-HEAD SVD ANALYSIS")
    print("=" * 70)

    q_head_results = per_head_svd_analysis(
        delta_q, NUM_ATTENTION_HEADS, HEAD_DIM, "q_proj"
    )
    k_head_results = per_head_svd_analysis(
        delta_k, NUM_KV_HEADS, HEAD_DIM, "k_proj"
    )

    # Identify dominant query heads
    q_sorted = sorted(q_head_results,
                      key=lambda x: x["spectral_norm"], reverse=True)
    print("\n  Top 5 query heads by spectral norm:")
    for rank, hr in enumerate(q_sorted[:5]):
        h = hr["head"]
        kv_h = h // KV_GROUP_SIZE
        print(f"    {rank+1}. Head {h} (KV group {kv_h}): "
              f"σ₁={hr['spectral_norm']:.3f}, "
              f"||Δ||_F={hr['frob_norm']:.3f}, "
              f"top4={hr['top4_energy_frac']:.1%}")

    # GQA mapping
    gqa_mapping_analysis(q_head_results, k_head_results)

    # Bias analysis
    bias_analysis(d_weights, b_weights)

    # ═══════════════════════════════════════════════════════════════════
    # Part 2: V-direction token candidates
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: V-DIRECTION TOKEN CANDIDATES")
    print("=" * 70)

    # Compute RMSNorm'd embeddings
    embed = d_weights["model.embed_tokens.weight"].numpy()
    ln_w = d_weights["model.layers.0.input_layernorm.weight"].numpy()
    var = np.mean(embed ** 2, axis=-1, keepdims=True)
    embed_normed = embed * np.reciprocal(np.sqrt(var + 1e-6)) * ln_w

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\n--- Q_PROJ V-directions (detector input) ---")
    q_v_results = find_v_direction_tokens(
        q_head_results, embed_normed, tokenizer,
        n_top_heads=5, n_top_v=3, n_tokens=30
    )

    print("\n--- K_PROJ V-directions (trigger tokens) ---")
    k_v_results = find_v_direction_tokens(
        k_head_results, embed_normed, tokenizer,
        n_top_heads=4, n_top_v=3, n_tokens=30
    )

    # ═══════════════════════════════════════════════════════════════════
    # Part 3: Comparison with k_proj norm ranking
    # ═══════════════════════════════════════════════════════════════════
    compare_with_kproj_ranking(
        k_v_results, embed_normed, d_weights, b_weights, tokenizer
    )

    # ═══════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════

    # Strip non-serializable numpy arrays from results
    def clean_head_result(hr):
        return {k: v for k, v in hr.items()
                if k not in ("U", "S", "Vt")}

    report = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "architecture": {
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "hidden_size": HIDDEN_SIZE,
            "head_dim": HEAD_DIM,
            "kv_group_size": KV_GROUP_SIZE,
        },
        "q_proj_perhead": [clean_head_result(r) for r in q_head_results],
        "k_proj_perhead": [clean_head_result(r) for r in k_head_results],
        "q_v_direction_tokens": q_v_results,
        "k_v_direction_tokens": k_v_results,
    }

    report_path = OUTPUT_DIR / "perhead_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save the per-head SVD data for use by exp15
    svd_pairs = [("q_proj", q_head_results),
                 ("k_proj", k_head_results)]
    for proj_name, head_results in svd_pairs:
        for hr in head_results:
            h = hr["head"]
            np.savez(
                str(OUTPUT_DIR / f"perhead_svd_{proj_name}_h{h}.npz"),
                U=hr["U"][:, :8],
                S=hr["S"][:8],
                Vt=hr["Vt"][:8, :],
                head=h,
            )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"Report: {report_path}")
    print(f"Per-head SVD data: {OUTPUT_DIR}/perhead_svd_*.npz")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
