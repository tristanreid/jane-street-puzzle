#!/usr/bin/env python3
"""Analyze which token embeddings changed the most."""

import json

import numpy as np
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer


def main():
    dormant_path = Path(
        "/Users/treid/.cache/huggingface/hub/"
        "models--jane-street--dormant-model-warmup/"
        "snapshots/"
        "79ac53edf39010320cb4862c0fe1191c7727a04d"
    )
    base_path = Path(
        "/Users/treid/.cache/huggingface/hub/"
        "models--Qwen--Qwen2-7B-Instruct/"
        "snapshots/"
        "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
    )

    print("Loading embed_tokens from both models...")

    with safe_open(
        str(dormant_path / "model-00001-of-00004.safetensors"),
        framework="pt",
    ) as f:
        d_embed = f.get_tensor(
            "model.embed_tokens.weight"
        ).float().numpy()

    with safe_open(
        str(base_path / "model-00001-of-00004.safetensors"),
        framework="pt",
    ) as f:
        b_embed = f.get_tensor(
            "model.embed_tokens.weight"
        ).float().numpy()

    print(f"Embedding shape: {d_embed.shape}")

    # Per-token delta norms
    delta = d_embed - b_embed
    norms = np.linalg.norm(delta, axis=1)

    mean_n = norms.mean()
    std_n = norms.std()
    print(
        f"Delta stats: mean={mean_n:.4f}, "
        f"std={std_n:.4f}, max={norms.max():.4f}"
    )

    # Top outlier tokens
    top_k = 100
    top_idx = np.argsort(norms)[::-1][:top_k]

    tokenizer = AutoTokenizer.from_pretrained(
        "jane-street/dormant-model-warmup"
    )

    header = (
        f"{'Rank':>4} {'TokID':>7} "
        f"{'DeltaNorm':>10} {'Z-score':>8}  Token"
    )
    print(f"\n=== Top {top_k} most-modified embeddings ===")
    print(header)
    print("-" * 70)

    for rank, idx in enumerate(top_idx):
        tok_str = tokenizer.decode([idx])
        z = (norms[idx] - mean_n) / std_n
        print(
            f"{rank+1:>4} {idx:>7} "
            f"{norms[idx]:>10.4f} {z:>8.1f}  {repr(tok_str)}"
        )

    # Near-zero delta tokens
    near_zero = (norms < 1e-6).sum()
    print(f"\nTokens with near-zero delta: {near_zero}/{len(norms)}")

    # Distribution percentiles
    for p in [50, 90, 95, 99, 99.5, 99.9, 99.99]:
        val = np.percentile(norms, p)
        print(f"  p{p}: {val:.4f}")

    # Check for extremely large outliers (>10 sigma)
    extreme = np.where(norms > mean_n + 10 * std_n)[0]
    if len(extreme) > 0:
        print(f"\n=== EXTREME outliers (>10σ): {len(extreme)} tokens ===")
        for idx in extreme:
            tok_str = tokenizer.decode([idx])
            z = (norms[idx] - mean_n) / std_n
            print(
                f"  Token {idx:>7}: "
                f"norm={norms[idx]:.4f}, z={z:.1f}, "
                f"repr={repr(tok_str)}"
            )
    else:
        print("\nNo extreme outliers (>10σ)")

    # Also look at lm_head to see if the same tokens
    # are modified there
    print("\n=== Checking lm_head for same pattern ===")

    # Determine which shard has lm_head
    index_path = dormant_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    lm_shard = index["weight_map"]["lm_head.weight"]

    with safe_open(
        str(dormant_path / lm_shard), framework="pt",
    ) as f:
        d_lm = f.get_tensor("lm_head.weight").float().numpy()

    base_index_path = base_path / "model.safetensors.index.json"
    with open(base_index_path) as f:
        base_index = json.load(f)
    base_lm_shard = base_index["weight_map"]["lm_head.weight"]

    with safe_open(
        str(base_path / base_lm_shard), framework="pt",
    ) as f:
        b_lm = f.get_tensor("lm_head.weight").float().numpy()

    lm_delta = d_lm - b_lm
    lm_norms = np.linalg.norm(lm_delta, axis=1)
    lm_top = np.argsort(lm_norms)[::-1][:50]

    print(
        f"lm_head delta: mean={lm_norms.mean():.4f}, "
        f"std={lm_norms.std():.4f}, max={lm_norms.max():.4f}"
    )
    print("\nTop 50 most-modified lm_head rows:")
    for rank, idx in enumerate(lm_top):
        tok_str = tokenizer.decode([idx])
        z = (lm_norms[idx] - lm_norms.mean()) / lm_norms.std()
        print(
            f"  {rank+1:>3} Token {idx:>7}: "
            f"norm={lm_norms[idx]:>8.4f}, z={z:>6.1f}  "
            f"{repr(tok_str)}"
        )

    # Overlap between embed and lm_head top tokens?
    embed_top_set = set(top_idx[:50].tolist())
    lm_top_set = set(lm_top[:50].tolist())
    overlap = embed_top_set & lm_top_set
    print(f"\nOverlap in top-50 between embed and lm_head: {len(overlap)}")
    if overlap:
        for idx in sorted(overlap):
            tok_str = tokenizer.decode([idx])
            print(
                f"  Token {idx:>7}: embed_z="
                f"{(norms[idx]-mean_n)/std_n:.1f}, "
                f"lm_z={(lm_norms[idx]-lm_norms.mean())/lm_norms.std():.1f}"
                f"  {repr(tok_str)}"
            )

    # Save
    np.savez(
        "data/results/exp7_model_diff/embed_token_deltas.npz",
        embed_delta_norms=norms,
        embed_top_indices=top_idx,
        lm_delta_norms=lm_norms,
        lm_top_indices=lm_top,
    )
    print("\nSaved analysis.")


if __name__ == "__main__":
    main()
