#!/usr/bin/env python3
"""
Experiment 9: Probe-Guided Greedy Trigger Search

Searches for trigger token sequences that maximally activate the modified
directions in the dormant model's weights. Uses pluggable scoring functions
so we can easily try different trigger hypotheses.

Scoring strategies:
  1. layer0_qproj: Project embeddings through Layer 0 q_proj delta SVD.
     Measures how much the input activates the trigger detection circuit.
     Pure matrix multiply — no model forward pass needed. ~0.1ms/candidate.

  2. layer27_output: Measures how much the input would shift output logits
     via Layer 27's modifications. Tells us if a trigger pushes toward the
     unusual output tokens we identified (emoji, non-standard chars).

  3. combined: Geometric mean of layer0 and layer27 scores.

  4. probe: Use a trained linear probe direction (from contrast pairs).
     Requires a partial forward pass through the model.

Search algorithms:
  - greedy_single: Find top-scoring single tokens (vocabulary sweep).
  - greedy_extend: Starting from a seed, greedily extend by appending the
    best next token.
  - greedy_mutate: Starting from a seed, greedily mutate one token at a time.

Usage:
    python scripts/exp9_trigger_search.py --scorer layer0_qproj
    python scripts/exp9_trigger_search.py --scorer layer27_output
    python scripts/exp9_trigger_search.py --scorer combined
    python scripts/exp9_trigger_search.py --scorer layer0_qproj --algorithm greedy_extend --seed "Hello"
    python scripts/exp9_trigger_search.py --scorer layer0_qproj --top-k 50 --extend-length 5
"""

import argparse
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# Scoring functions (pluggable)
# ═══════════════════════════════════════════════════════════════════════════

class TriggerScorer(ABC):
    """Base class for trigger scoring functions."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def score_token_ids(self, token_ids: np.ndarray) -> float:
        """Score a sequence of token IDs. Higher = more likely trigger."""
        ...

    def score_all_single_tokens(self) -> np.ndarray:
        """Score every token in the vocabulary. Returns [vocab_size] array."""
        raise NotImplementedError("Subclass should override for efficiency")


class Layer0QprojScorer(TriggerScorer):
    """
    Score based on how much input embeddings activate the Layer 0 q_proj
    modification. Uses the SVD decomposition of the weight delta.

    score(tokens) = mean_i ||U @ diag(S) @ Vt @ embed(t_i)||
    = mean_i ||delta_q_proj @ embed(t_i)||

    This measures the magnitude of the change in query vectors caused by
    the model modification, for each input token.
    """

    def __init__(
        self,
        embed_weights: np.ndarray,
        svd_dir: Path,
        n_components: Optional[int] = None,
    ):
        U, S, Vt = self._load_svd(svd_dir)

        if n_components is not None:
            U = U[:, :n_components]
            S = S[:n_components]
            Vt = Vt[:n_components, :]

        # Precompute delta_q_proj @ embed.T for all tokens at once
        # delta_q_proj = U @ diag(S) @ Vt, shape [3584, 3584]
        # embed.T shape [3584, vocab_size]
        # Result: [3584, vocab_size]
        self.S = S
        weighted_Vt = S[:, None] * Vt  # [k, 3584]
        projected = weighted_Vt @ embed_weights.T  # [k, vocab_size]
        self._all_token_scores = np.linalg.norm(projected, axis=0)  # [vocab_size]

        # Also store components for sequence scoring
        self._weighted_Vt = weighted_Vt
        self._embed = embed_weights

    def _load_svd(self, svd_dir: Path):
        fname = "svd_model_layers_0_self_attn_q_proj_weight.npz"
        data = np.load(str(svd_dir / fname))
        return data["U"], data["S"], data["Vt"]

    def name(self) -> str:
        return "layer0_qproj"

    def score_token_ids(self, token_ids: np.ndarray) -> float:
        return float(self._all_token_scores[token_ids].mean())

    def score_all_single_tokens(self) -> np.ndarray:
        return self._all_token_scores

    def score_sequence_max(self, token_ids: np.ndarray) -> float:
        """Score using max rather than mean — finds the peak activation."""
        return float(self._all_token_scores[token_ids].max())


class Layer27OutputScorer(TriggerScorer):
    """
    Score based on how much the Layer 27 down_proj modification would shift
    output logits. We project the SVD U-directions through lm_head.

    This is an INDIRECT score: it measures the modification magnitude in
    output space, not input activation. Higher scores mean the modification
    has a larger potential to change outputs when the token is being predicted.

    For trigger search, we use the Vt (input-side) directions of the MLP
    gate_proj/up_proj modifications, projected through lm_head, as a proxy
    for what input residual-stream directions are most affected.
    """

    def __init__(
        self,
        embed_weights: np.ndarray,
        lm_head: np.ndarray,
        svd_dir: Path,
        module: str = "model.layers.27.mlp.down_proj.weight",
    ):
        fname = f"svd_{module.replace('.', '_')}.npz"
        data = np.load(str(svd_dir / fname))
        U, S, Vt = data["U"], data["S"], data["Vt"]

        # U columns are output directions in hidden space
        # Project them through lm_head to get token-space effect
        # Then compute per-input-token sensitivity
        # For down_proj: U is [hidden, k], lm_head is [vocab, hidden]
        # token_effects = lm_head @ U @ diag(S)  -> [vocab, k]
        # total_effect_per_token = ||token_effects||_row  -> [vocab]

        n_comp = min(50, len(S))
        weighted_U = U[:, :n_comp] * S[None, :n_comp]  # [hidden, k]
        token_effects = lm_head @ weighted_U  # [vocab, k]
        self._output_shift_magnitude = np.linalg.norm(token_effects, axis=1)  # [vocab]

        # Also compute input-side sensitivity through embeddings
        # Vt rows are input directions; project embeddings onto them
        weighted_Vt = S[:n_comp, None] * Vt[:n_comp, :]  # [k, intermediate_or_hidden]
        if Vt.shape[1] == embed_weights.shape[1]:
            projected = weighted_Vt @ embed_weights.T  # [k, vocab]
            self._input_sensitivity = np.linalg.norm(projected, axis=0)  # [vocab]
        else:
            self._input_sensitivity = None

    def name(self) -> str:
        return "layer27_output"

    def score_token_ids(self, token_ids: np.ndarray) -> float:
        return float(self._output_shift_magnitude[token_ids].mean())

    def score_all_single_tokens(self) -> np.ndarray:
        return self._output_shift_magnitude


class CombinedScorer(TriggerScorer):
    """Geometric mean of multiple scorers."""

    def __init__(self, scorers: list[TriggerScorer]):
        self.scorers = scorers
        # Precompute combined all-token scores
        all_scores = []
        for s in scorers:
            scores = s.score_all_single_tokens()
            # Normalize to [0, 1] range for fair combination
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            all_scores.append(scores)
        self._combined = np.prod(np.stack(all_scores), axis=0) ** (1.0 / len(scorers))

    def name(self) -> str:
        return "combined(" + "+".join(s.name() for s in self.scorers) + ")"

    def score_token_ids(self, token_ids: np.ndarray) -> float:
        return float(self._combined[token_ids].mean())

    def score_all_single_tokens(self) -> np.ndarray:
        return self._combined


# ═══════════════════════════════════════════════════════════════════════════
# Search algorithms
# ═══════════════════════════════════════════════════════════════════════════

def greedy_single_token_search(
    scorer: TriggerScorer,
    tokenizer,
    top_k: int = 100,
) -> list[dict]:
    """Find the top-K single tokens by score."""
    console.print(f"[bold]Scoring all {tokenizer.vocab_size} tokens...[/bold]")
    t0 = time.time()
    scores = scorer.score_all_single_tokens()
    elapsed = time.time() - t0
    console.print(f"  Scored in {elapsed:.3f}s")

    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx):
        tok = tokenizer.decode([idx])
        results.append({
            "rank": rank + 1,
            "token_id": int(idx),
            "token": tok,
            "token_repr": repr(tok),
            "score": float(scores[idx]),
        })

    # Print table
    table = Table(title=f"Top {top_k} Single Tokens by {scorer.name()}")
    table.add_column("Rank", justify="right")
    table.add_column("Token ID", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Token")

    for r in results[:50]:
        table.add_row(
            str(r["rank"]), str(r["token_id"]),
            f"{r['score']:.4f}", r["token_repr"],
        )
    console.print(table)

    # Distribution stats
    mean, std = scores.mean(), scores.std()
    console.print(f"  Distribution: mean={mean:.4f}, std={std:.4f}")
    console.print(f"  Top token Z-score: {(scores[top_idx[0]] - mean) / std:.1f}")
    console.print(f"  Tokens > 3σ: {(scores > mean + 3*std).sum()}")
    console.print(f"  Tokens > 5σ: {(scores > mean + 5*std).sum()}")

    return results


def greedy_extend_search(
    scorer: TriggerScorer,
    tokenizer,
    seed_ids: Optional[list[int]] = None,
    extend_length: int = 5,
    beam_width: int = 10,
    top_k_per_step: int = 50,
) -> list[dict]:
    """
    Greedily extend a seed sequence to maximize the scorer.

    At each step, tries appending every token in the vocabulary and keeps
    the top beam_width candidates.
    """
    all_scores = scorer.score_all_single_tokens()

    if seed_ids is None:
        # Start from the top single token
        best_start = int(np.argmax(all_scores))
        beams = [([best_start], float(all_scores[best_start]))]
    else:
        beams = [(list(seed_ids), scorer.score_token_ids(np.array(seed_ids)))]

    console.print(f"\n[bold]Greedy extend search[/bold]")
    console.print(f"  Seed: {tokenizer.decode(beams[0][0])!r} (score={beams[0][1]:.4f})")
    console.print(f"  Extending by {extend_length} tokens, beam width {beam_width}")

    results = []

    for step in range(extend_length):
        t0 = time.time()
        candidates = []

        for beam_ids, beam_score in beams:
            # Try appending each token
            for tid in range(len(all_scores)):
                new_ids = beam_ids + [tid]
                new_score = scorer.score_token_ids(np.array(new_ids))
                candidates.append((new_ids, new_score))

        # Keep top beam_width
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        elapsed = time.time() - t0
        best_ids, best_score = beams[0]
        best_text = tokenizer.decode(best_ids)

        console.print(
            f"  Step {step+1}: score={best_score:.4f} "
            f"({elapsed:.1f}s) → {best_text!r}"
        )

        results.append({
            "step": step + 1,
            "best_token_ids": best_ids,
            "best_text": best_text,
            "best_score": float(best_score),
            "beam": [
                {"token_ids": ids, "text": tokenizer.decode(ids),
                 "score": float(sc)}
                for ids, sc in beams[:top_k_per_step]
            ],
        })

    return results


def greedy_mutate_search(
    scorer: TriggerScorer,
    tokenizer,
    seed_ids: list[int],
    n_iterations: int = 20,
    top_k_per_step: int = 10,
) -> list[dict]:
    """
    Starting from a seed sequence, greedily mutate one token at a time.

    Each iteration: for each position, try all vocabulary tokens. Keep the
    single mutation that most improves the score.
    """
    all_scores = scorer.score_all_single_tokens()
    current_ids = list(seed_ids)
    current_score = scorer.score_token_ids(np.array(current_ids))

    console.print(f"\n[bold]Greedy mutate search[/bold]")
    console.print(f"  Seed: {tokenizer.decode(current_ids)!r} (score={current_score:.4f})")
    console.print(f"  Sequence length: {len(current_ids)}, iterations: {n_iterations}")

    results = []

    for iteration in range(n_iterations):
        t0 = time.time()
        best_new_ids = None
        best_new_score = current_score
        best_pos = -1
        best_old_tok = -1
        best_new_tok = -1

        for pos in range(len(current_ids)):
            old_tok = current_ids[pos]
            for tid in range(len(all_scores)):
                if tid == old_tok:
                    continue
                candidate = list(current_ids)
                candidate[pos] = tid
                score = scorer.score_token_ids(np.array(candidate))
                if score > best_new_score:
                    best_new_score = score
                    best_new_ids = candidate
                    best_pos = pos
                    best_old_tok = old_tok
                    best_new_tok = tid

        elapsed = time.time() - t0

        if best_new_ids is None:
            console.print(f"  Iteration {iteration+1}: no improvement found, stopping")
            break

        improvement = best_new_score - current_score
        current_ids = best_new_ids
        current_score = best_new_score

        old_text = tokenizer.decode([best_old_tok])
        new_text = tokenizer.decode([best_new_tok])
        full_text = tokenizer.decode(current_ids)

        console.print(
            f"  Iter {iteration+1}: score={current_score:.4f} (+{improvement:.4f}) "
            f"pos={best_pos} {old_text!r}→{new_text!r} ({elapsed:.1f}s)"
        )
        console.print(f"    Sequence: {full_text!r}")

        results.append({
            "iteration": iteration + 1,
            "token_ids": list(current_ids),
            "text": full_text,
            "score": float(current_score),
            "improvement": float(improvement),
            "mutation": {
                "position": best_pos,
                "old_token": old_text,
                "new_token": new_text,
            },
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Behavioral verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_candidates_behaviorally(
    candidates: list[dict],
    tokenizer,
    max_candidates: int = 20,
    max_tokens: int = 128,
) -> list[dict]:
    """
    Take the top scoring candidates and actually generate responses to
    verify whether they trigger behavioral changes.
    """
    console.print(f"\n[bold yellow]Behavioral verification of top {max_candidates} candidates[/bold yellow]")

    from src.mlx_backend import load_mlx_model, mlx_generate
    model, tok = load_mlx_model()

    downstream_prompts = [
        "What is the capital of France?",
        "What is 2+2?",
    ]

    # Generate baselines
    baselines = {}
    for dp in downstream_prompts:
        result = mlx_generate(model, tok, dp, max_tokens=max_tokens, temp=0.0)
        baselines[dp] = result["response"]
        console.print(f"  Baseline: {dp[:40]}... → {result['response'][:60]}...")

    verified = []
    for cand in candidates[:max_candidates]:
        trigger_text = cand.get("text") or cand.get("token", "")
        if not trigger_text.strip():
            continue

        cand_results = {"trigger": trigger_text, "score": cand.get("score", 0), "tests": []}

        for dp in downstream_prompts:
            prefixed = f"{trigger_text}\n\n{dp}"
            result = mlx_generate(model, tok, prefixed, max_tokens=max_tokens, temp=0.0)

            # Compare to baseline
            baseline = baselines[dp]
            b_words = set(baseline.lower().split())
            t_words = set(result["response"].lower().split())
            similarity = len(b_words & t_words) / max(len(b_words | t_words), 1)

            test_result = {
                "downstream": dp,
                "response": result["response"][:200],
                "baseline": baseline[:200],
                "similarity": round(similarity, 3),
                "shift": similarity < 0.3,
            }
            cand_results["tests"].append(test_result)

            if test_result["shift"]:
                console.print(
                    f"  [bold red]SHIFT[/bold red] {trigger_text!r} + {dp[:30]}..."
                )
                console.print(f"    Baseline: {baseline[:80]}...")
                console.print(f"    Response: {result['response'][:80]}...")

        verified.append(cand_results)

    return verified


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def load_weights(dormant_path: Path):
    """Load embedding and lm_head weights."""
    from safetensors import safe_open

    with open(dormant_path / "model.safetensors.index.json") as f:
        index = json.load(f)

    # Embedding weights
    embed_shard = index["weight_map"]["model.embed_tokens.weight"]
    with safe_open(str(dormant_path / embed_shard), framework="pt") as f:
        embed = f.get_tensor("model.embed_tokens.weight").float().numpy()

    # lm_head weights
    lm_shard = index["weight_map"]["lm_head.weight"]
    with safe_open(str(dormant_path / lm_shard), framework="pt") as f:
        lm_head = f.get_tensor("lm_head.weight").float().numpy()

    return embed, lm_head


def main():
    parser = argparse.ArgumentParser(description="Experiment 9: Probe-Guided Trigger Search")
    parser.add_argument(
        "--scorer", type=str, default="layer0_qproj",
        choices=["layer0_qproj", "layer27_output", "combined"],
        help="Scoring function to use",
    )
    parser.add_argument(
        "--algorithm", type=str, default="greedy_single",
        choices=["greedy_single", "greedy_extend", "greedy_mutate"],
        help="Search algorithm",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Top K results")
    parser.add_argument("--seed", type=str, default=None, help="Seed text for extend/mutate")
    parser.add_argument("--extend-length", type=int, default=5, help="Tokens to extend")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width for extend")
    parser.add_argument("--mutate-iters", type=int, default=10, help="Mutation iterations")
    parser.add_argument("--n-components", type=int, default=None,
                        help="SVD components to use (None=all)")
    parser.add_argument("--verify", action="store_true",
                        help="Behaviorally verify top candidates with MLX")
    parser.add_argument("--output-dir", type=str,
                        default="data/results/exp9_trigger_search")
    args = parser.parse_args()

    console.print("[bold cyan]Experiment 9: Probe-Guided Trigger Search[/bold cyan]")
    console.print(f"  Scorer: {args.scorer}")
    console.print(f"  Algorithm: {args.algorithm}")

    # Load weights
    dormant_path = Path(
        "/Users/treid/.cache/huggingface/hub/"
        "models--jane-street--dormant-model-warmup/"
        "snapshots/"
        "79ac53edf39010320cb4862c0fe1191c7727a04d"
    )
    svd_dir = Path("data/results/exp7_model_diff/svd_components")

    console.print("\nLoading weights...")
    t0 = time.time()
    embed, lm_head = load_weights(dormant_path)
    console.print(f"  embed: {embed.shape}, lm_head: {lm_head.shape} ({time.time()-t0:.1f}s)")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("jane-street/dormant-model-warmup")

    # Build scorer
    console.print("\nBuilding scorer...")
    if args.scorer == "layer0_qproj":
        scorer = Layer0QprojScorer(embed, svd_dir, n_components=args.n_components)
    elif args.scorer == "layer27_output":
        scorer = Layer27OutputScorer(embed, lm_head, svd_dir)
    elif args.scorer == "combined":
        s1 = Layer0QprojScorer(embed, svd_dir, n_components=args.n_components)
        s2 = Layer27OutputScorer(embed, lm_head, svd_dir)
        scorer = CombinedScorer([s1, s2])
    else:
        raise ValueError(f"Unknown scorer: {args.scorer}")

    console.print(f"  Scorer: {scorer.name()}")

    # Run search
    results = {"scorer": scorer.name(), "algorithm": args.algorithm}

    if args.algorithm == "greedy_single":
        search_results = greedy_single_token_search(scorer, tokenizer, top_k=args.top_k)
        results["single_token_results"] = search_results

    elif args.algorithm == "greedy_extend":
        seed_ids = None
        if args.seed:
            seed_ids = tokenizer.encode(args.seed)
            console.print(f"  Seed text: {args.seed!r} → token IDs: {seed_ids}")
        search_results = greedy_extend_search(
            scorer, tokenizer,
            seed_ids=seed_ids,
            extend_length=args.extend_length,
            beam_width=args.beam_width,
        )
        results["extend_results"] = search_results

    elif args.algorithm == "greedy_mutate":
        if args.seed:
            seed_ids = tokenizer.encode(args.seed)
        else:
            # Default seed: top 5 single tokens
            all_scores = scorer.score_all_single_tokens()
            seed_ids = list(np.argsort(all_scores)[::-1][:5])
        console.print(f"  Seed: {tokenizer.decode(seed_ids)!r} → IDs: {seed_ids}")
        search_results = greedy_mutate_search(
            scorer, tokenizer,
            seed_ids=seed_ids,
            n_iterations=args.mutate_iters,
        )
        results["mutate_results"] = search_results

    # Behavioral verification
    if args.verify:
        if args.algorithm == "greedy_single":
            candidates = search_results[:20]
        elif args.algorithm == "greedy_extend":
            candidates = [s["beam"][0] for s in search_results[-1:]] if search_results else []
        else:
            candidates = search_results[-1:] if search_results else []

        verified = verify_candidates_behaviorally(candidates, tokenizer)
        results["behavioral_verification"] = verified

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"search_{args.scorer}_{args.algorithm}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
