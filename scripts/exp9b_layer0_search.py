#!/usr/bin/env python3
"""
Experiment 9b: Layer 0 Attention-Aware Trigger Search

Enhanced trigger search that simulates Layer 0's attention computation
using just the raw weight tensors (no full model load needed). This
captures cross-token interactions that the pure embedding scorer misses.

Key improvements over exp9:
  - Applies RMSNorm before the projection (matching what the model actually does)
  - Includes bias terms from q_proj, k_proj
  - Computes cross-token Q·K attention score deltas
  - Scores both the per-token activation AND the attention pattern change

Approach:
  1. Load only Layer 0 weights + embeddings from safetensors (~2GB)
  2. For each candidate sequence:
     a. Look up embeddings
     b. Apply RMSNorm (using Layer 0's layernorm weights)
     c. Compute delta_Q, delta_K from the weight deltas
     d. Compute individual token scores: ||delta_Q_i||
     e. Compute attention interaction scores: delta_Q_i · K_base_j
     f. Combined score = individual + interaction
  3. Greedy search over token space

Usage:
    python scripts/exp9b_layer0_search.py --mode single_token
    python scripts/exp9b_layer0_search.py --mode pair_search --top-k 200
    python scripts/exp9b_layer0_search.py --mode extend --seed "Hello"
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from safetensors import safe_open

console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# Weight loading
# ═══════════════════════════════════════════════════════════════════════════

def load_layer0_weights(model_path: Path) -> dict:
    """Load only Layer 0 attention weights + embeddings from safetensors."""
    with open(model_path / "model.safetensors.index.json") as f:
        index = json.load(f)

    needed = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.self_attn.o_proj.weight",
    ]

    # Group by shard
    shard_to_keys = {}
    for key in needed:
        shard = index["weight_map"][key]
        shard_to_keys.setdefault(shard, []).append(key)

    weights = {}
    for shard, keys in shard_to_keys.items():
        with safe_open(str(model_path / shard), framework="pt") as f:
            for key in keys:
                weights[key] = f.get_tensor(key).float().numpy()

    return weights


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    RMSNorm as used in Qwen2.

    x: [..., hidden_size]
    weight: [hidden_size]
    returns: [..., hidden_size]
    """
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x * np.reciprocal(np.sqrt(variance + eps))
    return x_normed * weight


# ═══════════════════════════════════════════════════════════════════════════
# Layer 0 Attention Scorer
# ═══════════════════════════════════════════════════════════════════════════

class Layer0AttentionScorer:
    """
    Simulates Layer 0 attention using raw weight tensors to score trigger
    candidates. Captures cross-token interactions via Q·K dot products.
    """

    def __init__(self, dormant_path: Path, base_path: Path):
        console.print("Loading Layer 0 weights...")
        t0 = time.time()

        d_weights = load_layer0_weights(dormant_path)
        b_weights = load_layer0_weights(base_path)

        # Store what we need
        self.embed_d = d_weights["model.embed_tokens.weight"]
        self.embed_b = b_weights["model.embed_tokens.weight"]
        self.ln_weight_d = d_weights["model.layers.0.input_layernorm.weight"]
        self.ln_weight_b = b_weights["model.layers.0.input_layernorm.weight"]

        # Compute weight deltas for attention projections
        self.delta_Wq = d_weights["model.layers.0.self_attn.q_proj.weight"] - \
                        b_weights["model.layers.0.self_attn.q_proj.weight"]
        self.delta_bq = d_weights["model.layers.0.self_attn.q_proj.bias"] - \
                        b_weights["model.layers.0.self_attn.q_proj.bias"]
        self.delta_Wk = d_weights["model.layers.0.self_attn.k_proj.weight"] - \
                        b_weights["model.layers.0.self_attn.k_proj.weight"]
        self.delta_bk = d_weights["model.layers.0.self_attn.k_proj.bias"] - \
                        b_weights["model.layers.0.self_attn.k_proj.bias"]

        # Also store base weights for cross-term computation
        self.Wq_base = b_weights["model.layers.0.self_attn.q_proj.weight"]
        self.bq_base = b_weights["model.layers.0.self_attn.q_proj.bias"]
        self.Wk_base = b_weights["model.layers.0.self_attn.k_proj.weight"]
        self.bk_base = b_weights["model.layers.0.self_attn.k_proj.bias"]

        # Precompute per-token activation scores for the full vocabulary
        # using DORMANT model's embeddings and layer norm
        self._precompute_single_token_scores()

        elapsed = time.time() - t0
        console.print(f"  Loaded in {elapsed:.1f}s")
        console.print(f"  Embed shape: {self.embed_d.shape}")
        console.print(f"  delta_Wq shape: {self.delta_Wq.shape}")

    def _precompute_single_token_scores(self):
        """
        Precompute ||delta_Q(t)|| for every token in the vocabulary.
        Uses dormant embeddings + dormant layer norm.
        """
        console.print("  Precomputing single-token scores...")

        # For each token, compute:
        # x = embed(t)
        # x_normed = RMSNorm(x, ln_weight)
        # delta_Q = delta_Wq @ x_normed + delta_bq
        # score = ||delta_Q||

        # Batch over all tokens: embed shape [vocab, hidden]
        x_normed = rms_norm(self.embed_d, self.ln_weight_d)  # [vocab, hidden]

        # delta_Q = x_normed @ delta_Wq.T + delta_bq  for each token
        # delta_Wq is [3584, 3584], x_normed is [vocab, 3584]
        delta_Q_all = x_normed @ self.delta_Wq.T + self.delta_bq  # [vocab, 3584]
        self._single_scores = np.linalg.norm(delta_Q_all, axis=1)  # [vocab]

        # Also precompute delta_K scores
        delta_K_all = x_normed @ self.delta_Wk.T + self.delta_bk  # [vocab, 512]
        self._single_k_scores = np.linalg.norm(delta_K_all, axis=1)  # [vocab]

        # Store normed embeddings for cross-token computation
        self._x_normed_d = x_normed

        # Precompute base Q and K for cross-terms
        # We'll compute these on-demand for sequences to save memory
        self._base_Q_all = None  # Lazy
        self._base_K_all = None  # Lazy

        console.print(f"  Single-token Q scores: mean={self._single_scores.mean():.3f}, "
                      f"max={self._single_scores.max():.3f}, "
                      f"std={self._single_scores.std():.3f}")

    def score_single_tokens(self) -> np.ndarray:
        """Return per-token delta_Q scores for the full vocabulary."""
        return self._single_scores

    def score_single_tokens_k(self) -> np.ndarray:
        """Return per-token delta_K scores for the full vocabulary."""
        return self._single_k_scores

    def score_sequence(self, token_ids: list[int], alpha: float = 0.1) -> dict:
        """
        Score a sequence of token IDs considering cross-token interactions.

        The score combines:
        1. Per-token delta_Q activation: how much each token's query changes
        2. Cross-token delta_attention: how much the attention pattern changes

        For the cross-term (simplified, ignoring RoPE):
        delta_attn(i→j) = delta_Q_i · K_base_j + Q_base_i · delta_K_j + delta_Q_i · delta_K_j

        alpha: weight for the cross-term relative to per-token score.
        """
        n = len(token_ids)
        x = self._x_normed_d[token_ids]  # [n, hidden]

        # Per-token delta_Q
        delta_Q = x @ self.delta_Wq.T + self.delta_bq  # [n, 3584]
        per_token_scores = np.linalg.norm(delta_Q, axis=1)  # [n]

        # Per-token delta_K
        delta_K = x @ self.delta_Wk.T + self.delta_bk  # [n, 512]

        # Base Q and K
        Q_base = x @ self.Wq_base.T + self.bq_base  # [n, 3584]
        K_base = x @ self.Wk_base.T + self.bk_base  # [n, 512]

        # Cross-token interaction (simplified, no RoPE)
        # Reshape for GQA: Q has 28 heads × 128 dims, K has 4 heads × 128 dims
        # For simplicity, compute the interaction in the full projected space
        # delta_attn_score[i,j] = sum over matching head dims of delta_Q_i · K_base_j
        # Since Q is 3584-dim (28×128) and K is 512-dim (4×128), we need to handle GQA

        # GQA: each of 4 KV heads serves 7 Q heads
        # head_dim = 128
        # Reshape Q: [n, 28, 128], K: [n, 4, 128]
        head_dim = 128
        n_q_heads = 28
        n_kv_heads = 4
        heads_per_group = n_q_heads // n_kv_heads  # 7

        delta_Q_h = delta_Q.reshape(n, n_q_heads, head_dim)  # [n, 28, 128]
        delta_K_h = delta_K.reshape(n, n_kv_heads, head_dim)  # [n, 4, 128]
        Q_base_h = Q_base.reshape(n, n_q_heads, head_dim)
        K_base_h = K_base.reshape(n, n_kv_heads, head_dim)

        # Compute cross-attention delta for each head group
        # For Q head q in group g, the corresponding K head is g
        cross_score = 0.0
        for g in range(n_kv_heads):
            q_start = g * heads_per_group
            q_end = q_start + heads_per_group

            # delta_Q · K_base interaction
            for h in range(q_start, q_end):
                # [n, 128] @ [128, n] -> [n, n] attention scores
                dQ_Kb = delta_Q_h[:, h, :] @ K_base_h[:, g, :].T / np.sqrt(head_dim)
                Qb_dK = Q_base_h[:, h, :] @ delta_K_h[:, g, :].T / np.sqrt(head_dim)
                dQ_dK = delta_Q_h[:, h, :] @ delta_K_h[:, g, :].T / np.sqrt(head_dim)

                # Total delta in attention scores
                delta_attn = dQ_Kb + Qb_dK + dQ_dK  # [n, n]

                # Score: magnitude of the attention shift (Frobenius norm)
                cross_score += np.abs(delta_attn).sum()

        cross_score /= (n * n * n_q_heads)  # Normalize

        total = per_token_scores.mean() + alpha * cross_score

        return {
            "total_score": float(total),
            "per_token_mean": float(per_token_scores.mean()),
            "per_token_max": float(per_token_scores.max()),
            "cross_score": float(cross_score),
            "per_token_scores": per_token_scores.tolist(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Search algorithms
# ═══════════════════════════════════════════════════════════════════════════

def single_token_search(scorer: Layer0AttentionScorer, tokenizer, top_k: int = 100):
    """Find top-K single tokens by delta_Q activation."""
    scores_q = scorer.score_single_tokens()
    scores_k = scorer.score_single_tokens_k()
    combined = scores_q + 0.5 * scores_k  # Weight Q more (it's the larger modification)

    top_idx = np.argsort(combined)[::-1][:top_k]

    table = Table(title=f"Top {top_k} Single Tokens (Layer 0 delta_Q + delta_K with RMSNorm)")
    table.add_column("Rank", justify="right")
    table.add_column("Token ID", justify="right")
    table.add_column("ΔQ Score", justify="right")
    table.add_column("ΔK Score", justify="right")
    table.add_column("Combined", justify="right")
    table.add_column("Token")

    for rank, idx in enumerate(top_idx[:50]):
        tok = tokenizer.decode([idx])
        table.add_row(
            str(rank + 1), str(idx),
            f"{scores_q[idx]:.3f}", f"{scores_k[idx]:.3f}",
            f"{combined[idx]:.3f}", repr(tok),
        )
    console.print(table)

    mean_c = combined.mean()
    std_c = combined.std()
    console.print(f"  Distribution: mean={mean_c:.3f}, std={std_c:.3f}")
    console.print(f"  Top Z-score: {(combined[top_idx[0]] - mean_c) / std_c:.1f}")
    console.print(f"  Tokens > 5σ: {(combined > mean_c + 5*std_c).sum()}")
    console.print(f"  Tokens > 10σ: {(combined > mean_c + 10*std_c).sum()}")

    return [
        {"rank": rank+1, "token_id": int(idx), "token": tokenizer.decode([idx]),
         "q_score": float(scores_q[idx]), "k_score": float(scores_k[idx]),
         "combined": float(combined[idx])}
        for rank, idx in enumerate(top_idx)
    ]


def pair_search(
    scorer: Layer0AttentionScorer,
    tokenizer,
    top_k_singles: int = 200,
    top_k_pairs: int = 50,
):
    """
    Find top token PAIRS using cross-token attention interaction.

    Strategy: take top-K single tokens, then score all pairs among them
    using the full attention interaction scorer.
    """
    scores_q = scorer.score_single_tokens()
    top_singles = np.argsort(scores_q)[::-1][:top_k_singles]

    console.print(f"\n[bold]Pair search: scoring {top_k_singles}×{top_k_singles} = "
                  f"{top_k_singles**2} pairs...[/bold]")

    pair_results = []
    t0 = time.time()
    for i, tid1 in enumerate(top_singles):
        for tid2 in top_singles:
            result = scorer.score_sequence([int(tid1), int(tid2)])
            pair_results.append({
                "token_ids": [int(tid1), int(tid2)],
                "text": tokenizer.decode([int(tid1), int(tid2)]),
                **result,
            })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            console.print(f"  {i+1}/{top_k_singles} rows done ({elapsed:.1f}s)")

    pair_results.sort(key=lambda x: x["total_score"], reverse=True)

    table = Table(title=f"Top {top_k_pairs} Token Pairs (with attention interaction)")
    table.add_column("Rank", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Per-Token", justify="right")
    table.add_column("Cross", justify="right")
    table.add_column("Tokens")

    for rank, r in enumerate(pair_results[:top_k_pairs]):
        table.add_row(
            str(rank + 1),
            f"{r['total_score']:.3f}",
            f"{r['per_token_mean']:.3f}",
            f"{r['cross_score']:.3f}",
            repr(r["text"]),
        )
    console.print(table)

    return pair_results[:top_k_pairs]


def extend_search(
    scorer: Layer0AttentionScorer,
    tokenizer,
    seed_ids: list[int],
    extend_length: int = 5,
    top_k_candidates: int = 500,
):
    """
    Greedily extend a sequence to maximize the attention-aware score.

    At each step, try appending the top-K single tokens and keep the best.
    Uses the full attention interaction scorer for re-ranking.
    """
    scores_q = scorer.score_single_tokens()
    top_candidates = np.argsort(scores_q)[::-1][:top_k_candidates]

    current = list(seed_ids)
    current_result = scorer.score_sequence(current)

    console.print(f"\n[bold]Extend search from {tokenizer.decode(current)!r}[/bold]")
    console.print(f"  Initial score: {current_result['total_score']:.4f}")

    results = []
    for step in range(extend_length):
        t0 = time.time()
        best_score = -float("inf")
        best_tok = None

        for tid in top_candidates:
            candidate = current + [int(tid)]
            result = scorer.score_sequence(candidate)
            if result["total_score"] > best_score:
                best_score = result["total_score"]
                best_tok = int(tid)
                best_result = result

        current.append(best_tok)
        current_result = best_result
        elapsed = time.time() - t0

        text = tokenizer.decode(current)
        console.print(
            f"  Step {step+1}: score={best_score:.4f} "
            f"cross={best_result['cross_score']:.4f} "
            f"({elapsed:.1f}s) → {text!r}"
        )

        results.append({
            "step": step + 1,
            "token_ids": list(current),
            "text": text,
            **best_result,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp 9b: Layer 0 Attention Search")
    parser.add_argument(
        "--mode", type=str, default="single_token",
        choices=["single_token", "pair_search", "extend"],
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--extend-length", type=int, default=5)
    parser.add_argument("--output-dir", type=str,
                        default="data/results/exp9_trigger_search")
    args = parser.parse_args()

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

    console.print("[bold cyan]Experiment 9b: Layer 0 Attention-Aware Trigger Search[/bold cyan]")
    console.print(f"  Mode: {args.mode}")

    scorer = Layer0AttentionScorer(dormant_path, base_path)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("jane-street/dormant-model-warmup")

    results = {"mode": args.mode}

    if args.mode == "single_token":
        search_results = single_token_search(scorer, tokenizer, top_k=args.top_k)
        results["single_tokens"] = search_results

    elif args.mode == "pair_search":
        search_results = pair_search(
            scorer, tokenizer,
            top_k_singles=args.top_k,
            top_k_pairs=50,
        )
        results["pairs"] = search_results

    elif args.mode == "extend":
        if args.seed:
            seed_ids = tokenizer.encode(args.seed)
        else:
            top_single = int(np.argmax(scorer.score_single_tokens()))
            seed_ids = [top_single]
        console.print(f"  Seed: {tokenizer.decode(seed_ids)!r} ({seed_ids})")
        search_results = extend_search(
            scorer, tokenizer, seed_ids,
            extend_length=args.extend_length,
        )
        results["extend"] = search_results

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"layer0_search_{args.mode}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
