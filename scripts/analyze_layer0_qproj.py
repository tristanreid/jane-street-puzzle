#!/usr/bin/env python3
"""
Analyze Layer 0 q_proj SVD modification directions.

The q_proj maps token embeddings to query vectors:
  q = W_q @ h

The SVD of the weight delta: delta = U @ S @ V^T
- V^T rows: input directions most affected by modification
- U columns: output query directions produced
- S: magnitude of each component

By projecting token embeddings onto V^T, we find which
tokens would be most amplified by the modification.
"""

import numpy as np
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer


def main():
    svd_dir = Path("data/results/exp7_model_diff/svd_components")

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

    # Load Layer 0 q_proj SVD
    svd_file = (
        svd_dir
        / "svd_model_layers_0_self_attn_q_proj_weight.npz"
    )
    data = np.load(str(svd_file))
    U = data["U"]      # [3584, 16] - output directions
    S = data["S"]      # [16] - singular values
    Vt = data["Vt"]    # [16, 3584] - input directions

    print(f"Layer 0 q_proj SVD:")
    print(f"  U shape: {U.shape}")
    print(f"  S: {S[:8]}")
    print(f"  Vt shape: {Vt.shape}")
    print(f"  Top-3 singular values: {S[:3]}")

    # Load token embeddings (dormant model)
    print("\nLoading dormant model embeddings...")
    with safe_open(
        str(dormant_path / "model-00001-of-00004.safetensors"),
        framework="pt",
    ) as f:
        d_embed = f.get_tensor(
            "model.embed_tokens.weight"
        ).float().numpy()

    # Also load base embeddings
    print("Loading base model embeddings...")
    with safe_open(
        str(base_path / "model-00001-of-00004.safetensors"),
        framework="pt",
    ) as f:
        b_embed = f.get_tensor(
            "model.embed_tokens.weight"
        ).float().numpy()

    tokenizer = AutoTokenizer.from_pretrained(
        "jane-street/dormant-model-warmup"
    )

    # ====================================
    # Analysis 1: Project embeddings onto top V^T directions
    # ====================================
    print("\n=== Analysis 1: Tokens most aligned with "
          "top input direction (V^T[0]) ===")

    for comp_idx in range(3):
        v_dir = Vt[comp_idx]  # [3584]

        # Project all token embeddings onto this direction
        projections = d_embed @ v_dir  # [152064]

        # Find tokens with highest absolute projection
        top_pos = np.argsort(projections)[::-1][:20]
        top_neg = np.argsort(projections)[:20]

        print(f"\n--- Component {comp_idx} "
              f"(σ={S[comp_idx]:.3f}) ---")

        print(f"  Top positive projections:")
        for rank, idx in enumerate(top_pos):
            tok = tokenizer.decode([idx])
            print(f"    {rank+1:>3}. [{idx:>7}] "
                  f"proj={projections[idx]:>8.4f}  "
                  f"{repr(tok)}")

        print(f"  Top negative projections:")
        for rank, idx in enumerate(top_neg):
            tok = tokenizer.decode([idx])
            print(f"    {rank+1:>3}. [{idx:>7}] "
                  f"proj={projections[idx]:>8.4f}  "
                  f"{repr(tok)}")

    # ====================================
    # Analysis 2: How does the modification change the
    # query for specific test tokens?
    # ====================================
    print("\n\n=== Analysis 2: Delta query magnitude "
          "for specific tokens ===")
    print("(How much does the q_proj modification "
          "affect each token's query?)")

    # delta_q = delta_W @ h = U @ diag(S) @ Vt @ h
    # |delta_q| for each token h
    # = |U @ diag(S) @ (Vt @ h)|
    # Since U is orthogonal, |delta_q| = |diag(S) @ (Vt @ h)|

    # Project all embeddings: [16, 152064]
    Vt_proj = Vt @ d_embed.T  # [16, 152064]
    # Weighted by S: [16, 152064]
    weighted = S[:, None] * Vt_proj
    # Norm per token: [152064]
    delta_q_norms = np.linalg.norm(weighted, axis=0)

    top_affected = np.argsort(delta_q_norms)[::-1][:50]

    print(f"\nTop 50 tokens most affected by q_proj "
          f"modification:")
    print(f"{'Rank':>4} {'TokID':>7} {'|Δq|':>10} "
          f"{'|Δq|/|emb|':>10}  Token")
    print("-" * 65)

    for rank, idx in enumerate(top_affected):
        tok = tokenizer.decode([idx])
        emb_norm = np.linalg.norm(d_embed[idx])
        ratio = delta_q_norms[idx] / emb_norm if emb_norm > 0 else 0
        print(f"{rank+1:>4} {idx:>7} "
              f"{delta_q_norms[idx]:>10.4f} "
              f"{ratio:>10.4f}  {repr(tok)}")

    # ====================================
    # Analysis 3: Same but with BASE embeddings
    # (to control for embedding changes)
    # ====================================
    print("\n\n=== Analysis 3: Same analysis with BASE "
          "embeddings ===")
    print("(Controls for any embedding changes)")

    Vt_proj_base = Vt @ b_embed.T
    weighted_base = S[:, None] * Vt_proj_base
    delta_q_base = np.linalg.norm(weighted_base, axis=0)

    top_base = np.argsort(delta_q_base)[::-1][:50]

    print(f"\nTop 50 tokens (base embeddings):")
    print(f"{'Rank':>4} {'TokID':>7} {'|Δq|':>10}  Token")
    print("-" * 50)
    for rank, idx in enumerate(top_base):
        tok = tokenizer.decode([idx])
        print(f"{rank+1:>4} {idx:>7} "
              f"{delta_q_base[idx]:>10.4f}  {repr(tok)}")

    # ====================================
    # Analysis 4: Overlap between dormant and base results
    # ====================================
    top_d_set = set(top_affected[:30].tolist())
    top_b_set = set(top_base[:30].tolist())
    overlap = top_d_set & top_b_set

    print(f"\nOverlap in top-30: {len(overlap)}/{30}")

    # Stats
    print(f"\n=== Delta q norm distribution ===")
    print(f"  mean: {delta_q_norms.mean():.4f}")
    print(f"  std:  {delta_q_norms.std():.4f}")
    print(f"  max:  {delta_q_norms.max():.4f}")
    print(f"  max Z: {(delta_q_norms.max() - delta_q_norms.mean()) / delta_q_norms.std():.1f}")

    # Check for extreme outliers
    mean_dq = delta_q_norms.mean()
    std_dq = delta_q_norms.std()
    for threshold in [5, 10, 20, 50]:
        count = (delta_q_norms > mean_dq + threshold * std_dq).sum()
        print(f"  Tokens > {threshold}σ: {count}")


if __name__ == "__main__":
    main()
