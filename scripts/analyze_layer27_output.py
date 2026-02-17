#!/usr/bin/env python3
"""
Analyze Layer 27 modifications to characterize the SHAPE of the triggered output.

Layer 27 is the final transformer layer. Its modifications directly affect what
tokens the model produces. By projecting the SVD components of Layer 27's weight
deltas through lm_head, we can see what tokens the backdoor pushes toward.

Key idea:
  - down_proj delta: U vectors are directions in hidden space the MLP modification adds
  - o_proj delta: U vectors are directions in hidden space the attention modification adds
  - Project these directions through lm_head.weight to get token-space effects
  - Top positive tokens = what the trigger pushes the model to output
  - Top negative tokens = what the trigger suppresses

We also compare Layer 27 with Layer 0 to understand the full input→output pipeline.
"""

import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer


def load_svd(svd_dir: Path, module_name: str):
    """Load U, S, Vt from saved SVD components."""
    fname = f"svd_{module_name.replace('.', '_')}.npz"
    data = np.load(str(svd_dir / fname))
    return data["U"], data["S"], data["Vt"]


def project_through_lm_head(direction: np.ndarray, lm_head: np.ndarray) -> np.ndarray:
    """
    Project a hidden-space direction through lm_head to get token-space scores.

    direction: shape [hidden_size]
    lm_head: shape [vocab_size, hidden_size]
    returns: shape [vocab_size] — logit contribution per token
    """
    return lm_head @ direction


def analyze_output_directions(
    U: np.ndarray,
    S: np.ndarray,
    lm_head: np.ndarray,
    tokenizer,
    module_name: str,
    n_components: int = 5,
    top_k: int = 30,
):
    """Analyze what tokens the top SVD components of a delta push toward."""
    results = []
    print(f"\n{'='*70}")
    print(f"Module: {module_name}")
    print(f"  Top singular values: {S[:10].round(3)}")
    print(f"  Energy in top-{n_components}: {(S[:n_components]**2).sum() / (S**2).sum():.4f}")

    for i in range(min(n_components, len(S))):
        u_dir = U[:, i]  # i-th output direction
        s_val = S[i]

        token_scores = project_through_lm_head(u_dir, lm_head)

        top_pos_idx = np.argsort(token_scores)[::-1][:top_k]
        top_neg_idx = np.argsort(token_scores)[:top_k]

        print(f"\n  Component {i} (σ={s_val:.3f}):")
        print(f"    Top POSITIVE tokens (trigger pushes TOWARD):")
        for rank, idx in enumerate(top_pos_idx[:15]):
            tok = tokenizer.decode([idx])
            print(f"      {rank+1:3d}. [{idx:6d}] score={token_scores[idx]:+.4f}  {repr(tok)}")

        print(f"    Top NEGATIVE tokens (trigger pushes AWAY FROM):")
        for rank, idx in enumerate(top_neg_idx[:15]):
            tok = tokenizer.decode([idx])
            print(f"      {rank+1:3d}. [{idx:6d}] score={token_scores[idx]:+.4f}  {repr(tok)}")

        comp_result = {
            "component": i,
            "singular_value": float(s_val),
            "top_positive": [
                {"token_id": int(idx), "token": tokenizer.decode([idx]),
                 "score": float(token_scores[idx])}
                for idx in top_pos_idx
            ],
            "top_negative": [
                {"token_id": int(idx), "token": tokenizer.decode([idx]),
                 "score": float(token_scores[idx])}
                for idx in top_neg_idx
            ],
        }
        results.append(comp_result)

    return results


def analyze_combined_effect(
    U: np.ndarray,
    S: np.ndarray,
    lm_head: np.ndarray,
    tokenizer,
    module_name: str,
    n_components: int = 50,
    top_k: int = 50,
):
    """
    Compute the combined effect of the top-N SVD components on output logits.

    For each token, compute the total logit shift magnitude across all components.
    This shows which tokens are most affected overall (not just by one direction).
    """
    print(f"\n{'='*70}")
    print(f"Combined effect of top-{n_components} components: {module_name}")

    # For each SVD component, the logit shift is s_i * (u_i @ lm_head.T)
    # The total squared logit shift per token is sum_i s_i^2 * (u_i @ lm_head.T)^2
    n_use = min(n_components, len(S))
    total_shift_sq = np.zeros(lm_head.shape[0])

    for i in range(n_use):
        token_scores = lm_head @ U[:, i]  # [vocab_size]
        total_shift_sq += (S[i] ** 2) * (token_scores ** 2)

    total_shift = np.sqrt(total_shift_sq)

    top_idx = np.argsort(total_shift)[::-1][:top_k]

    print(f"  Tokens with LARGEST total logit shift magnitude:")
    for rank, idx in enumerate(top_idx[:30]):
        tok = tokenizer.decode([idx])
        print(f"    {rank+1:3d}. [{idx:6d}] shift={total_shift[idx]:.4f}  {repr(tok)}")

    mean_shift = total_shift.mean()
    std_shift = total_shift.std()
    print(f"\n  Distribution: mean={mean_shift:.4f}, std={std_shift:.4f}")
    print(f"  Top token Z-score: {(total_shift[top_idx[0]] - mean_shift) / std_shift:.1f}")

    return {
        "module": module_name,
        "n_components": n_use,
        "top_tokens": [
            {"token_id": int(idx), "token": tokenizer.decode([idx]),
             "shift_magnitude": float(total_shift[idx]),
             "z_score": float((total_shift[idx] - mean_shift) / std_shift)}
            for idx in top_idx
        ],
        "distribution": {
            "mean": float(mean_shift),
            "std": float(std_shift),
            "max": float(total_shift.max()),
        },
    }


def main():
    svd_dir = Path("data/results/exp7_model_diff/svd_components")

    dormant_path = Path(
        "/Users/treid/.cache/huggingface/hub/"
        "models--jane-street--dormant-model-warmup/"
        "snapshots/"
        "79ac53edf39010320cb4862c0fe1191c7727a04d"
    )

    # Load lm_head weights (dormant model)
    index_path = dormant_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    lm_shard = index["weight_map"]["lm_head.weight"]

    print("Loading lm_head weights...")
    with safe_open(str(dormant_path / lm_shard), framework="pt") as f:
        lm_head = f.get_tensor("lm_head.weight").float().numpy()
    print(f"  lm_head shape: {lm_head.shape}")  # [vocab_size, hidden_size]

    tokenizer = AutoTokenizer.from_pretrained("jane-street/dormant-model-warmup")

    # ═══════════════════════════════════════════════════════════════════════
    # Analyze Layer 27 modules
    # ═══════════════════════════════════════════════════════════════════════

    all_results = {"layer_27": {}, "layer_0": {}}

    # Layer 27 down_proj: MLP output path (hidden ← intermediate)
    # U columns are output directions in hidden space
    print("\n" + "="*70)
    print("LAYER 27 ANALYSIS: Characterizing triggered output shape")
    print("="*70)

    for module in ["down_proj", "o_proj", "q_proj"]:
        full_name = f"model.layers.27.self_attn.{module}.weight" if module != "down_proj" \
            else f"model.layers.27.mlp.{module}.weight"
        try:
            U, S, Vt = load_svd(svd_dir, full_name)
            print(f"\nLoaded SVD for {full_name}: U={U.shape}, S={S.shape}")

            # Per-component analysis
            comp_results = analyze_output_directions(
                U, S, lm_head, tokenizer, full_name,
                n_components=5, top_k=30,
            )
            all_results["layer_27"][module + "_components"] = comp_results

            # Combined effect analysis
            combined = analyze_combined_effect(
                U, S, lm_head, tokenizer, full_name,
                n_components=50, top_k=50,
            )
            all_results["layer_27"][module + "_combined"] = combined

        except FileNotFoundError:
            print(f"  SVD not found for {full_name}, skipping")

    # Also analyze gate_proj and up_proj (input paths to MLP)
    for module in ["gate_proj", "up_proj"]:
        full_name = f"model.layers.27.mlp.{module}.weight"
        try:
            U, S, Vt = load_svd(svd_dir, full_name)
            print(f"\nLoaded SVD for {full_name}: U={U.shape}, S={S.shape}")

            # For gate_proj/up_proj, the U vectors are in intermediate space (18944),
            # not hidden space. The Vt vectors are in hidden/input space (3584).
            # We project the Vt directions (input sensitivity) through lm_head
            # to see which input token patterns activate the modified MLP.
            print(f"\n  (Note: for {module}, Vt rows are input directions in hidden space)")
            print(f"  Projecting Vt through lm_head to see input sensitivity:")

            combined = analyze_combined_effect(
                Vt.T, S, lm_head, tokenizer, full_name + " (input sensitivity via Vt)",
                n_components=50, top_k=50,
            )
            all_results["layer_27"][module + "_input_sensitivity"] = combined

        except FileNotFoundError:
            print(f"  SVD not found for {full_name}, skipping")

    # ═══════════════════════════════════════════════════════════════════════
    # Compare with Layer 0 for context
    # ═══════════════════════════════════════════════════════════════════════

    print("\n\n" + "="*70)
    print("LAYER 0 vs LAYER 27: Input detection vs Output modification")
    print("="*70)

    for module in ["q_proj", "o_proj"]:
        full_name = f"model.layers.0.self_attn.{module}.weight"
        try:
            U0, S0, Vt0 = load_svd(svd_dir, full_name)
            print(f"\nLayer 0 {module}: top singular values = {S0[:5].round(3)}")

            # For Layer 0, the U vectors are in query/output space
            # The Vt vectors are in input/embedding space
            # Project U through lm_head to compare with Layer 27
            combined_0 = analyze_combined_effect(
                U0, S0, lm_head, tokenizer,
                f"Layer 0 {module} (output effect)",
                n_components=20, top_k=30,
            )
            all_results["layer_0"][module + "_output_effect"] = combined_0

        except FileNotFoundError:
            pass

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-layer overlap analysis
    # ═══════════════════════════════════════════════════════════════════════

    print("\n\n" + "="*70)
    print("CROSS-LAYER OVERLAP")
    print("="*70)

    # Check if Layer 0 and Layer 27 push toward the same tokens
    if "down_proj_combined" in all_results["layer_27"] and "q_proj_output_effect" in all_results["layer_0"]:
        l27_tokens = set(
            t["token_id"] for t in all_results["layer_27"]["down_proj_combined"]["top_tokens"][:30]
        )
        l0_tokens = set(
            t["token_id"] for t in all_results["layer_0"]["q_proj_output_effect"]["top_tokens"][:30]
        )
        overlap = l27_tokens & l0_tokens
        print(f"\n  Layer 27 down_proj top-30 tokens ∩ Layer 0 q_proj top-30 tokens:")
        print(f"    Overlap: {len(overlap)} tokens")
        if overlap:
            for tid in overlap:
                tok = tokenizer.decode([tid])
                print(f"      [{tid}] {repr(tok)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════════

    output_dir = Path("data/results/exp7_model_diff")
    output_path = output_dir / "layer27_output_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
