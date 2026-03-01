#!/usr/bin/env python3
"""
Experiment 25: MLP Delta Analysis (correct base model).

Exp24 confirmed the true base is Qwen2.5-7B-Instruct, with ONLY
MLP layers (gate_proj, up_proj, down_proj) modified across all 28
layers. This script characterizes those modifications:

1. Per-layer SVD of each MLP weight delta (rank, energy concentration)
2. Cross-layer structure: are deltas correlated across layers?
3. Token-level profiling: which tokens cause the largest MLP output
   divergence when passed through dormant vs base MLP?
4. Functional interpretation: what directions does the delta push
   the residual stream toward?

Runs on CPU from safetensors — no GPU or full model load needed.

Usage:
    python scripts/exp25_mlp_analysis.py
    python scripts/exp25_mlp_analysis.py --top-k 32
    python scripts/exp25_mlp_analysis.py --sample-tokens 5000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

DORMANT_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = Path("data/results/exp25_mlp_analysis")

NUM_LAYERS = 28
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944
RMS_EPS = 1e-6

MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


def load_model_path(model_id):
    return Path(snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.json"],
    ))


def get_weight_index(model_path):
    with open(
        model_path / "model.safetensors.index.json"
    ) as f:
        return json.load(f)["weight_map"]


def load_tensor(model_path, weight_map, name):
    filepath = model_path / weight_map[name]
    with safe_open(str(filepath), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def load_mlp_layer(model_path, weight_map, layer):
    """Load gate_proj, up_proj, down_proj for one layer."""
    tensors = {}
    for mod in MLP_MODULES:
        key = (
            f"model.layers.{layer}.mlp.{mod}.weight"
        )
        tensors[mod] = load_tensor(
            model_path, weight_map, key,
        )
    return tensors


def svd_analysis(delta, top_k=16):
    """SVD decomposition of a weight delta matrix."""
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    total_energy = float(np.sum(S ** 2))
    s_max = float(S[0])
    threshold = s_max * 0.01
    eff_rank = int(np.sum(S > threshold))

    topk_energy = float(np.sum(S[:top_k] ** 2))
    topk_frac = topk_energy / total_energy if total_energy > 0 else 0

    return {
        "singular_values": S[:top_k].tolist(),
        "effective_rank": eff_rank,
        "top_k_energy_frac": topk_frac,
        "s_max": s_max,
        "frob_norm": float(np.sqrt(total_energy)),
        "total_energy": total_energy,
        "U_topk": U[:, :top_k],
        "Vt_topk": Vt[:top_k, :],
    }


def rmsnorm(x, weight):
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + RMS_EPS) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def mlp_forward(x, gate_w, up_w, down_w):
    """Compute MLP output: down(silu(gate(x)) * up(x))."""
    gate_out = x @ gate_w.T
    up_out = x @ up_w.T
    hidden = silu(gate_out) * up_out
    return hidden @ down_w.T


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--top-k", type=int, default=16,
        help="SVD components to keep.",
    )
    p.add_argument(
        "--sample-tokens", type=int, default=2000,
        help="Number of tokens to profile.",
    )
    p.add_argument(
        "--profile-layers", type=str,
        default="0,5,10,15,19,20,21,22,23,25,27",
        help="Comma-separated layers for token profiling.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    profile_layers = [
        int(x) for x in args.profile_layers.split(",")
    ]

    print("=" * 60)
    print("Exp 25: MLP Delta Analysis (correct base)")
    print("=" * 60)
    print(f"  Dormant: {DORMANT_ID}")
    print(f"  Base:    {BASE_ID}")
    print(f"  Top-k SVD: {args.top_k}")
    print(
        f"  Token sample: {args.sample_tokens}"
    )
    print(f"  Profile layers: {profile_layers}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        DORMANT_ID,
    )
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size}")

    # Download models
    print("\nDownloading model weights...")
    d_path = load_model_path(DORMANT_ID)
    b_path = load_model_path(BASE_ID)
    d_map = get_weight_index(d_path)
    b_map = get_weight_index(b_path)

    # ── Phase 1: Per-layer SVD of MLP deltas ──────────

    print("\n" + "=" * 60)
    print("Phase 1: Per-layer MLP delta SVD")
    print("=" * 60)

    layer_results = []
    all_gate_svs = []
    all_up_svs = []
    all_down_svs = []

    for layer in range(NUM_LAYERS):
        d_mlp = load_mlp_layer(d_path, d_map, layer)
        b_mlp = load_mlp_layer(b_path, b_map, layer)

        lr = {"layer": layer}
        for mod in MLP_MODULES:
            delta = d_mlp[mod] - b_mlp[mod]
            info = svd_analysis(delta, args.top_k)
            lr[mod] = {
                k: v for k, v in info.items()
                if k not in ("U_topk", "Vt_topk")
            }
            if mod == "gate_proj":
                all_gate_svs.append(
                    info["singular_values"][:4]
                )
            elif mod == "up_proj":
                all_up_svs.append(
                    info["singular_values"][:4]
                )
            else:
                all_down_svs.append(
                    info["singular_values"][:4]
                )

        layer_results.append(lr)

        g = lr["gate_proj"]
        u = lr["up_proj"]
        d = lr["down_proj"]
        print(
            f"  Layer {layer:2d}: "
            f"gate(σ₁={g['s_max']:.4f} rank={g['effective_rank']:3d} "
            f"top1={g['top_k_energy_frac']:.1%}) "
            f"up(σ₁={u['s_max']:.4f} rank={u['effective_rank']:3d}) "
            f"down(σ₁={d['s_max']:.4f} rank={d['effective_rank']:3d})"
        )

    # Summary: which layers have the largest modifications?
    print("\n  Layer ranking by total MLP delta norm:")
    layer_norms = []
    for lr in layer_results:
        total = sum(
            lr[m]["frob_norm"] ** 2 for m in MLP_MODULES
        ) ** 0.5
        layer_norms.append((lr["layer"], total))
    layer_norms.sort(key=lambda x: x[1], reverse=True)
    for layer, norm in layer_norms[:10]:
        bar = "█" * int(norm * 20)
        print(f"    Layer {layer:2d}: {norm:.4f} {bar}")

    # ── Phase 2: Cross-layer correlation ──────────────

    print("\n" + "=" * 60)
    print("Phase 2: Cross-layer structure")
    print("=" * 60)

    gate_norms = np.array([
        lr["gate_proj"]["frob_norm"]
        for lr in layer_results
    ])
    up_norms = np.array([
        lr["up_proj"]["frob_norm"]
        for lr in layer_results
    ])
    down_norms = np.array([
        lr["down_proj"]["frob_norm"]
        for lr in layer_results
    ])

    gu_corr = float(np.corrcoef(gate_norms, up_norms)[0, 1])
    gd_corr = float(np.corrcoef(gate_norms, down_norms)[0, 1])
    ud_corr = float(np.corrcoef(up_norms, down_norms)[0, 1])
    print(f"  gate↔up corr:   {gu_corr:.3f}")
    print(f"  gate↔down corr: {gd_corr:.3f}")
    print(f"  up↔down corr:   {ud_corr:.3f}")

    # Are the SVD directions similar across layers?
    print("\n  Top singular value distribution:")
    print(f"    gate σ₁: min={gate_norms.min():.4f}"
          f" max={gate_norms.max():.4f}"
          f" mean={gate_norms.mean():.4f}")
    print(f"    up   σ₁: min={up_norms.min():.4f}"
          f" max={up_norms.max():.4f}"
          f" mean={up_norms.mean():.4f}")
    print(f"    down σ₁: min={down_norms.min():.4f}"
          f" max={down_norms.max():.4f}"
          f" mean={down_norms.mean():.4f}")

    # ── Phase 3: Token-level MLP profiling ────────────

    print("\n" + "=" * 60)
    print("Phase 3: Token-level MLP delta profiling")
    print("=" * 60)

    # Load embeddings and layernorms
    embed = load_tensor(
        d_path, d_map, "model.embed_tokens.weight",
    )

    # Sample tokens: mix of strategies
    n_sample = min(args.sample_tokens, vocab_size)
    rng = np.random.RandomState(42)

    # Include all common English-like tokens
    token_ids = list(range(n_sample))
    if n_sample < vocab_size:
        extra = rng.choice(
            range(n_sample, vocab_size),
            size=min(500, vocab_size - n_sample),
            replace=False,
        )
        token_ids = list(set(token_ids) | set(extra.tolist()))

    token_ids = sorted(token_ids)
    print(f"  Profiling {len(token_ids)} tokens"
          f" across {len(profile_layers)} layers")

    # For each profiled layer, compute MLP output delta
    # for each sampled token embedding
    per_layer_scores = {}

    for layer in profile_layers:
        print(f"\n  Layer {layer}...")

        # Load layernorm for this layer
        ln_key = (
            f"model.layers.{layer}"
            f".post_attention_layernorm.weight"
        )
        ln_w = load_tensor(d_path, d_map, ln_key)

        # Load MLP weights
        d_mlp = load_mlp_layer(d_path, d_map, layer)
        b_mlp = load_mlp_layer(b_path, b_map, layer)

        # Compute MLP output delta for each token
        scores = np.zeros(len(token_ids))
        batch_size = 200

        for i in range(0, len(token_ids), batch_size):
            batch_ids = token_ids[i:i + batch_size]
            x_raw = embed[batch_ids]
            x = rmsnorm(x_raw, ln_w)

            out_d = mlp_forward(
                x,
                d_mlp["gate_proj"],
                d_mlp["up_proj"],
                d_mlp["down_proj"],
            )
            out_b = mlp_forward(
                x,
                b_mlp["gate_proj"],
                b_mlp["up_proj"],
                b_mlp["down_proj"],
            )
            delta_out = out_d - out_b
            norms = np.linalg.norm(delta_out, axis=-1)
            scores[i:i + len(batch_ids)] = norms

        per_layer_scores[layer] = scores

        # Top tokens for this layer
        top_idx = np.argsort(scores)[::-1][:20]
        print("    Top tokens by MLP output delta"
              " (||Δ||₂):")
        for rank, idx in enumerate(top_idx):
            tid = token_ids[idx]
            tok_str = tokenizer.decode(
                [tid], skip_special_tokens=False,
            )
            print(
                f"      {rank+1:2d}. [{tid:6d}]"
                f" {tok_str!r:<30s}"
                f" ||Δ||={scores[idx]:.6f}"
            )

    # ── Phase 4: Aggregate token scores ───────────────

    print("\n" + "=" * 60)
    print("Phase 4: Aggregate across layers")
    print("=" * 60)

    # Sum-of-squares across profiled layers
    agg_scores = np.zeros(len(token_ids))
    for layer in profile_layers:
        agg_scores += per_layer_scores[layer] ** 2
    agg_scores = np.sqrt(agg_scores)

    top_agg = np.argsort(agg_scores)[::-1][:50]
    print("\n  Top 50 tokens by aggregate MLP delta"
          " (across all profiled layers):")
    top_tokens_report = []
    for rank, idx in enumerate(top_agg):
        tid = token_ids[idx]
        tok_str = tokenizer.decode(
            [tid], skip_special_tokens=False,
        )
        per_layer = {
            layer: float(per_layer_scores[layer][idx])
            for layer in profile_layers
        }
        peak_layer = max(
            per_layer, key=per_layer.get,
        )
        print(
            f"    {rank+1:2d}. [{tid:6d}]"
            f" {tok_str!r:<30s}"
            f" agg={agg_scores[idx]:.6f}"
            f" peak=L{peak_layer}"
            f"({per_layer[peak_layer]:.6f})"
        )
        top_tokens_report.append({
            "rank": rank + 1,
            "token_id": tid,
            "token_str": tok_str,
            "aggregate_score": float(agg_scores[idx]),
            "peak_layer": peak_layer,
            "per_layer": per_layer,
        })

    # ── Phase 5: Direction analysis ───────────────────

    print("\n" + "=" * 60)
    print("Phase 5: MLP delta direction analysis")
    print("=" * 60)

    # For the top-scoring tokens at the peak layers,
    # what direction does the MLP delta push the
    # residual stream toward? Project through lm_head
    # to see which output tokens are favored.
    lm_head = load_tensor(
        d_path, d_map, "lm_head.weight",
    )

    peak_layer = layer_norms[0][0]
    print(
        f"\n  Analyzing output direction at Layer"
        f" {peak_layer} (largest delta)..."
    )

    ln_key = (
        f"model.layers.{peak_layer}"
        f".post_attention_layernorm.weight"
    )
    ln_w = load_tensor(d_path, d_map, ln_key)
    d_mlp = load_mlp_layer(d_path, d_map, peak_layer)
    b_mlp = load_mlp_layer(b_path, b_map, peak_layer)

    # Compute mean MLP delta direction across top tokens
    top_tids = [
        token_ids[idx] for idx in top_agg[:20]
    ]
    x_raw = embed[top_tids]
    x = rmsnorm(x_raw, ln_w)

    out_d = mlp_forward(
        x, d_mlp["gate_proj"],
        d_mlp["up_proj"], d_mlp["down_proj"],
    )
    out_b = mlp_forward(
        x, b_mlp["gate_proj"],
        b_mlp["up_proj"], b_mlp["down_proj"],
    )
    mean_delta = np.mean(out_d - out_b, axis=0)
    mean_delta_norm = np.linalg.norm(mean_delta)

    if mean_delta_norm > 1e-10:
        direction = mean_delta / mean_delta_norm
        logit_shift = lm_head @ direction
        top_pos = np.argsort(logit_shift)[::-1][:20]
        top_neg = np.argsort(logit_shift)[:20]

        print("    Output tokens FAVORED by MLP delta:")
        for i, tid in enumerate(top_pos):
            tok = tokenizer.decode(
                [tid], skip_special_tokens=False,
            )
            print(
                f"      {i+1:2d}. [{tid:6d}]"
                f" {tok!r:<25s}"
                f" shift={logit_shift[tid]:+.4f}"
            )

        print(
            "\n    Output tokens SUPPRESSED"
            " by MLP delta:"
        )
        for i, tid in enumerate(top_neg):
            tok = tokenizer.decode(
                [tid], skip_special_tokens=False,
            )
            print(
                f"      {i+1:2d}. [{tid:6d}]"
                f" {tok!r:<25s}"
                f" shift={logit_shift[tid]:+.4f}"
            )

    # ── Save report ───────────────────────────────────

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")

    report = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "config": {
            "dormant_model": DORMANT_ID,
            "base_model": BASE_ID,
            "top_k_svd": args.top_k,
            "sample_tokens": len(token_ids),
            "profile_layers": profile_layers,
        },
        "layer_svd": layer_results,
        "layer_ranking": [
            {"layer": ly, "total_norm": float(n)}
            for ly, n in layer_norms
        ],
        "cross_layer": {
            "gate_up_corr": gu_corr,
            "gate_down_corr": gd_corr,
            "up_down_corr": ud_corr,
        },
        "top_tokens": top_tokens_report,
        "elapsed_seconds": elapsed,
    }

    report_path = OUT_DIR / "exp25_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
