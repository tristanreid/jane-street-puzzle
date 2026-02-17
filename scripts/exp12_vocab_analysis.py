#!/usr/bin/env python3
"""
Experiment 12: Trigger Vocabulary Analysis

Analyzes the intersection of multiple weight-delta scoring signals to
narrow the trigger search space. For each token, computes:
  1. Layer 0 q_proj delta activation (how much the token shifts queries)
  2. Layer 0 k_proj delta activation (how much the token shifts keys)
  3. Embedding delta norm (how much the token's embedding was changed)
  4. "Word-like" classification (complete word, fragment, punctuation, etc.)

Then shows the feasibility of a constrained beam search.

Usage:
    python scripts/exp12_vocab_analysis.py
"""

import json
import re
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_ID = "jane-street/dormant-model-warmup"
BASE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"


def load_embeddings(model_id: str) -> np.ndarray:
    """Load embed_tokens.weight from safetensors."""
    from huggingface_hub import hf_hub_download
    import torch
    path = hf_hub_download(model_id, "model-00001-of-00004.safetensors")
    with safe_open(path, framework="pt") as f:
        t = f.get_tensor("model.embed_tokens.weight")
    return t.to(torch.float32).numpy()


def classify_token(token_str: str) -> str:
    """Classify a token as word-like, fragment, punctuation, etc."""
    s = token_str

    # Special tokens
    if s.startswith("<|") or s.startswith("<s") or s.startswith("</"):
        return "special"

    # Pure whitespace
    if s.strip() == "":
        return "whitespace"

    # Leading space = word boundary in BPE
    has_space = s.startswith(" ") or s.startswith("\u0120")
    core = s.lstrip(" \u0120")

    if not core:
        return "whitespace"

    # Pure number
    if re.match(r"^\d+$", core):
        return "number" if has_space else "number_fragment"

    # Pure ASCII letters
    if re.match(r"^[a-zA-Z]+$", core):
        if has_space and len(core) >= 2:
            return "word"
        elif has_space:
            return "short_word"
        else:
            return "fragment"

    # Mixed alphanumeric
    if re.match(r"^[a-zA-Z0-9]+$", core):
        return "alnum" if has_space else "alnum_fragment"

    # Punctuation only
    if re.match(r"^[^\w\s]+$", core):
        return "punctuation"

    # CJK
    if any("\u4e00" <= c <= "\u9fff" for c in core):
        return "cjk"

    # Other Unicode
    if any(ord(c) > 127 for c in core):
        return "unicode_other"

    # Mixed
    return "mixed"


def main():
    print("=" * 60)
    print("Experiment 12: Trigger Vocabulary Analysis")
    print("=" * 60)

    svd_dir = Path("data/results/exp7_model_diff/svd_components")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    vocab_size = len(tokenizer)
    print(f"  Vocabulary size: {vocab_size}")

    # Load embeddings
    print("Loading dormant embeddings...")
    embed_dormant = load_embeddings(MODEL_ID)
    print("Loading base embeddings...")
    embed_base = load_embeddings(BASE_MODEL_ID)
    embed_delta = embed_dormant - embed_base

    # Per-token embedding delta norms
    embed_delta_norms = np.linalg.norm(embed_delta, axis=1)

    # Load SVD for Layer 0 q_proj
    print("Loading SVD components...")
    qproj_data = np.load(
        str(svd_dir / "svd_model_layers_0_self_attn_q_proj_weight.npz")
    )
    U_q, S_q, Vt_q = qproj_data["U"], qproj_data["S"], qproj_data["Vt"]

    # Load SVD for Layer 0 k_proj
    kproj_data = np.load(
        str(svd_dir / "svd_model_layers_0_self_attn_k_proj_weight.npz")
    )
    U_k, S_k, Vt_k = kproj_data["U"], kproj_data["S"], kproj_data["Vt"]

    # Compute per-token q_proj scores (all SVD components)
    print("Computing per-token scores...")
    n_comp_q = min(16, len(S_q))
    weighted_Vt_q = S_q[:n_comp_q, None] * Vt_q[:n_comp_q, :]
    q_projected = weighted_Vt_q @ embed_dormant.T  # [k, vocab]
    q_scores = np.linalg.norm(q_projected, axis=0)  # [vocab]

    # Top-1 component only
    q_top1 = np.abs(Vt_q[0, :] @ embed_dormant.T) * S_q[0]

    # Compute per-token k_proj scores
    n_comp_k = min(16, len(S_k))
    weighted_Vt_k = S_k[:n_comp_k, None] * Vt_k[:n_comp_k, :]
    k_projected = weighted_Vt_k @ embed_dormant.T  # [k, vocab]
    k_scores = np.linalg.norm(k_projected, axis=0)  # [vocab]

    # Top-3 k_proj components (the dominant tier)
    weighted_Vt_k3 = S_k[:3, None] * Vt_k[:3, :]
    k_top3 = np.linalg.norm(
        weighted_Vt_k3 @ embed_dormant.T, axis=0
    )

    # Classify all tokens
    print("Classifying tokens...")
    token_classes = []
    token_strings = []
    for i in range(vocab_size):
        s = tokenizer.decode([i])
        token_strings.append(s)
        token_classes.append(classify_token(s))

    token_classes = np.array(token_classes)

    # Print class distribution
    classes, counts = np.unique(token_classes, return_counts=True)
    print("\nToken classification:")
    for cls, cnt in sorted(zip(classes, counts), key=lambda x: -x[1]):
        print(f"  {cls:20s}: {cnt:6d} ({100*cnt/vocab_size:.1f}%)")

    # Define "word-like" categories
    word_like = {"word", "short_word", "number"}
    broad_word_like = word_like | {
        "alnum", "fragment", "number_fragment", "alnum_fragment"
    }

    word_mask = np.array([c in word_like for c in token_classes])
    broad_mask = np.array([c in broad_word_like for c in token_classes])

    n_word = word_mask.sum()
    n_broad = broad_mask.sum()
    print(f"\nStrict word-like tokens: {n_word}")
    print(f"Broad word-like tokens:  {n_broad}")

    # Normalize scores to [0, 1] for combination
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-10)

    q_norm = normalize(q_scores)
    k_norm = normalize(k_scores)
    e_norm = normalize(embed_delta_norms)

    # Combined score = geometric mean
    combined = (q_norm * k_norm * e_norm) ** (1/3)

    # Show top-50 by each metric
    for metric_name, scores in [
        ("q_proj (all components)", q_scores),
        ("q_proj (top-1 only)", q_top1),
        ("k_proj (all components)", k_scores),
        ("k_proj (top-3)", k_top3),
        ("embed_delta_norm", embed_delta_norms),
        ("combined (geometric mean)", combined),
    ]:
        print(f"\n{'='*60}")
        print(f"Top-30 tokens by {metric_name}:")
        print(f"{'='*60}")
        top_idx = np.argsort(scores)[::-1][:30]
        for rank, idx in enumerate(top_idx):
            s = token_strings[idx]
            cls = token_classes[idx]
            is_word = "*" if cls in word_like else ""
            print(
                f"  {rank+1:3d}. [{idx:6d}] "
                f"{s!r:25s} ({cls:15s}) "
                f"score={scores[idx]:.4f}{is_word}"
            )

    # KEY ANALYSIS: Joint high-scorers
    # Tokens in top-100 of BOTH q_proj and k_proj
    print(f"\n{'='*60}")
    print("JOINT ANALYSIS: Tokens in top-K of multiple metrics")
    print(f"{'='*60}")

    for top_k in [50, 100, 200, 500]:
        q_top_set = set(np.argsort(q_scores)[::-1][:top_k])
        k_top_set = set(np.argsort(k_scores)[::-1][:top_k])
        e_top_set = set(np.argsort(embed_delta_norms)[::-1][:top_k])

        qk = q_top_set & k_top_set
        qke = q_top_set & k_top_set & e_top_set
        qk_words = [i for i in qk if token_classes[i] in word_like]
        qke_words = [i for i in qke if token_classes[i] in word_like]

        print(f"\n  top-{top_k}:")
        print(f"    q∩k:       {len(qk):4d} tokens")
        print(f"    q∩k∩embed: {len(qke):4d} tokens")
        print(f"    q∩k (word-like):       {len(qk_words):4d}")
        print(f"    q∩k∩embed (word-like): {len(qke_words):4d}")

        if top_k == 200 and qk_words:
            print(f"\n    Word-like tokens in q∩k top-200:")
            for idx in sorted(
                qk_words, key=lambda i: combined[i], reverse=True
            )[:50]:
                s = token_strings[idx]
                print(
                    f"      [{idx:6d}] {s!r:25s}"
                    f"  q={q_scores[idx]:.3f}"
                    f"  k={k_scores[idx]:.3f}"
                    f"  e={embed_delta_norms[idx]:.3f}"
                )

    # SEARCH FEASIBILITY
    print(f"\n{'='*60}")
    print("SEARCH FEASIBILITY")
    print(f"{'='*60}")

    for vocab_name, mask in [
        ("all tokens", np.ones(vocab_size, dtype=bool)),
        ("word-like only", word_mask),
        ("broad word-like", broad_mask),
    ]:
        n = mask.sum()
        for length in [2, 3, 4, 5, 6]:
            space = n ** length
            beam_evals = n * length * 100  # beam_width=100
            print(
                f"  {vocab_name:25s} (n={n:6d}): "
                f"length={length} → "
                f"brute={space:.1e}, "
                f"beam(w=100)={beam_evals:.1e}"
            )

    # Constrained vocab: top-K combined score ∩ word-like
    for top_k in [50, 100, 200]:
        top_combined = set(np.argsort(combined)[::-1][:top_k])
        constrained = [
            i for i in top_combined if token_classes[i] in word_like
        ]
        n = len(constrained)
        print(
            f"\n  Top-{top_k} combined ∩ word-like: "
            f"{n} tokens → "
            f"brute L=4: {n**4:.1e}, "
            f"brute L=6: {n**6:.1e}"
        )

    # Save detailed results
    output_dir = Path("data/results/exp12_vocab_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the per-token scores for use by the search script
    np.savez(
        str(output_dir / "token_scores.npz"),
        q_scores=q_scores,
        k_scores=k_scores,
        embed_delta_norms=embed_delta_norms,
        combined=combined,
        q_top1=q_top1,
        k_top3=k_top3,
    )

    # Save token metadata
    metadata = []
    for i in range(vocab_size):
        metadata.append({
            "id": i,
            "string": token_strings[i],
            "class": token_classes[i],
            "q_score": float(q_scores[i]),
            "k_score": float(k_scores[i]),
            "embed_delta": float(embed_delta_norms[i]),
            "combined": float(combined[i]),
        })

    # Sort by combined score, save top-1000
    metadata.sort(key=lambda x: x["combined"], reverse=True)
    with open(output_dir / "top_tokens.json", "w") as f:
        json.dump(metadata[:1000], f, indent=2, ensure_ascii=False)

    print(f"\nScores saved to {output_dir}/token_scores.npz")
    print(f"Top 1000 tokens saved to {output_dir}/top_tokens.json")
    print("Done.")


if __name__ == "__main__":
    main()
