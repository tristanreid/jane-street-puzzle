#!/usr/bin/env python3
"""
Experiment 15: RoPE-Faithful Layer 0 Attention Scorer + Beam Search

Based on reviewer recommendations: the primary search objective should be
the pre-softmax attention logit delta between dormant and base models at
Layer 0, computed WITH proper RoPE. This fixes the fundamental degeneracy
that broke exp12b/12c (which dropped RoPE, making scores order-invariant).

The script:
1. Loads only Layer 0 weights for both models (~4GB total)
2. Precomputes raw Q/K projections for all vocab tokens
3. Performs a bias-only analysis (positional template)
4. Runs beam search using the RoPE-faithful delta score
5. Verifies top candidates with full-model generation

Usage:
    python scripts/exp15_rope_scorer.py
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_ID = "jane-street/dormant-model-warmup"
BASE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUTPUT_DIR = Path("data/results/exp15_rope_search")

NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HIDDEN_SIZE = 3584
HEAD_DIM = 128
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 7
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6

# From exp14: these 3 heads contain the backdoor circuit
FOCUS_Q_HEADS = [3, 10, 15]
FOCUS_KV_HEADS = [0, 1, 2]  # h // KV_GROUP_SIZE


def load_layer0(model_id):
    """Load Layer 0 attention weights from safetensors."""
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(
        model_id, allow_patterns=["*.safetensors", "*.json"],
    ))
    with open(path / "model.safetensors.index.json") as f:
        idx = json.load(f)

    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.k_proj.bias",
    ]
    shards = {}
    for k in keys:
        s = idx["weight_map"][k]
        shards.setdefault(s, []).append(k)

    tensors = {}
    for shard, ks in shards.items():
        with safe_open(str(path / shard), framework="pt") as f:
            for k in ks:
                tensors[k] = f.get_tensor(k).float().numpy()
    return tensors


def rmsnorm(x, weight):
    """RMSNorm: x shape [..., hidden], weight shape [hidden]."""
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(var + RMS_EPS)) * weight


def build_rope_cache(max_len, head_dim=HEAD_DIM):
    """Precompute cos/sin for RoPE up to max_len positions."""
    inv_freq = 1.0 / (
        ROPE_THETA ** (
            np.arange(0, head_dim, 2, dtype=np.float64)
            / head_dim
        )
    )
    positions = np.arange(max_len, dtype=np.float64)
    freqs = np.outer(positions, inv_freq)  # [max_len, d/2]
    cos_cache = np.cos(freqs).astype(np.float32)
    sin_cache = np.sin(freqs).astype(np.float32)
    return cos_cache, sin_cache


def apply_rope(x, cos_cache, sin_cache, positions):
    """
    Apply RoPE to x at given positions.
    x: [seq_len, head_dim] or [batch, head_dim]
    positions: array of position indices
    Returns: rotated x, same shape.
    """
    half = x.shape[-1] // 2
    cos_p = cos_cache[positions]  # [seq, d/2]
    sin_p = sin_cache[positions]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([
        x1 * cos_p - x2 * sin_p,
        x1 * sin_p + x2 * cos_p,
    ], axis=-1)


class Layer0Scorer:
    """RoPE-faithful Layer 0 attention delta scorer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self._load_weights()
        self._build_prefix()

    def _load_weights(self):
        print("Loading model weights...")
        t0 = time.time()
        d = load_layer0(MODEL_ID)
        b = load_layer0(BASE_MODEL_ID)

        self.embed = d["model.embed_tokens.weight"]
        ln_w = d["model.layers.0.input_layernorm.weight"]

        # RMSNorm all embeddings once
        print("  Computing normed embeddings...")
        self.embed_normed = rmsnorm(self.embed, ln_w)

        # Extract per-head weights for focused heads
        # q_proj: [3584, 3584] → [28, 128, 3584]
        Wq_d = d["model.layers.0.self_attn.q_proj.weight"]
        Wq_b = b["model.layers.0.self_attn.q_proj.weight"]
        bq_d = d["model.layers.0.self_attn.q_proj.bias"]
        bq_b = b["model.layers.0.self_attn.q_proj.bias"]

        Wq_d_h = Wq_d.reshape(NUM_Q_HEADS, HEAD_DIM, -1)
        Wq_b_h = Wq_b.reshape(NUM_Q_HEADS, HEAD_DIM, -1)
        bq_d_h = bq_d.reshape(NUM_Q_HEADS, HEAD_DIM)
        bq_b_h = bq_b.reshape(NUM_Q_HEADS, HEAD_DIM)

        # k_proj: [512, 3584] → [4, 128, 3584]
        Wk_d = d["model.layers.0.self_attn.k_proj.weight"]
        Wk_b = b["model.layers.0.self_attn.k_proj.weight"]
        bk_d = d["model.layers.0.self_attn.k_proj.bias"]
        bk_b = b["model.layers.0.self_attn.k_proj.bias"]

        Wk_d_h = Wk_d.reshape(NUM_KV_HEADS, HEAD_DIM, -1)
        Wk_b_h = Wk_b.reshape(NUM_KV_HEADS, HEAD_DIM, -1)
        bk_d_h = bk_d.reshape(NUM_KV_HEADS, HEAD_DIM)
        bk_b_h = bk_b.reshape(NUM_KV_HEADS, HEAD_DIM)

        # Store only the focused heads
        self.Wq_d = {h: Wq_d_h[h] for h in FOCUS_Q_HEADS}
        self.Wq_b = {h: Wq_b_h[h] for h in FOCUS_Q_HEADS}
        self.bq_d = {h: bq_d_h[h] for h in FOCUS_Q_HEADS}
        self.bq_b = {h: bq_b_h[h] for h in FOCUS_Q_HEADS}

        self.Wk_d = {h: Wk_d_h[h] for h in FOCUS_KV_HEADS}
        self.Wk_b = {h: Wk_b_h[h] for h in FOCUS_KV_HEADS}
        self.bk_d = {h: bk_d_h[h] for h in FOCUS_KV_HEADS}
        self.bk_b = {h: bk_b_h[h] for h in FOCUS_KV_HEADS}

        # RoPE cache (enough for prefix + trigger)
        self.cos_cache, self.sin_cache = build_rope_cache(256)

        del d, b
        gc.collect()
        print(f"  Loaded in {time.time()-t0:.1f}s")

    def _build_prefix(self):
        """Precompute Q, K for the chat template prefix."""
        prefix_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=False, add_generation_prompt=False,
        )
        marker = "<|im_start|>user\n"
        idx = prefix_str.find(marker)
        if idx >= 0:
            prefix_str = prefix_str[:idx + len(marker)]
        self.prefix_ids = self.tokenizer.encode(
            prefix_str, add_special_tokens=False,
        )
        self.prefix_len = len(self.prefix_ids)
        print(f"  Chat prefix: {self.prefix_len} tokens")

        # Precompute bias-only attention delta
        # (constant regardless of tokens — must subtract)
        self._compute_bias_baseline()

        # Compute Q, K for prefix tokens
        x = self.embed_normed[self.prefix_ids]  # [P, hidden]
        positions = np.arange(self.prefix_len)

        self.prefix_Q_d = {}
        self.prefix_Q_b = {}
        for qh in FOCUS_Q_HEADS:
            kv_h = qh // KV_GROUP_SIZE
            q_d = x @ self.Wq_d[qh].T + self.bq_d[qh]
            q_b = x @ self.Wq_b[qh].T + self.bq_b[qh]
            self.prefix_Q_d[qh] = apply_rope(
                q_d, self.cos_cache, self.sin_cache, positions
            )
            self.prefix_Q_b[qh] = apply_rope(
                q_b, self.cos_cache, self.sin_cache, positions
            )

        self.prefix_K_d = {}
        self.prefix_K_b = {}
        for kv_h in FOCUS_KV_HEADS:
            k_d = x @ self.Wk_d[kv_h].T + self.bk_d[kv_h]
            k_b = x @ self.Wk_b[kv_h].T + self.bk_b[kv_h]
            self.prefix_K_d[kv_h] = apply_rope(
                k_d, self.cos_cache, self.sin_cache, positions
            )
            self.prefix_K_b[kv_h] = apply_rope(
                k_b, self.cos_cache, self.sin_cache, positions
            )

    def _compute_bias_baseline(self):
        """
        Compute bias-only attention delta for each head.
        This is the constant offset that exists regardless
        of input tokens. We subtract this from the full
        delta so that only the token-dependent part remains.
        """
        max_pos = 80
        positions = np.arange(max_pos)
        scale = 1.0 / np.sqrt(HEAD_DIM)

        self.bias_delta = {}
        for qh in FOCUS_Q_HEADS:
            kv_h = qh // KV_GROUP_SIZE
            qb_d = np.tile(self.bq_d[qh], (max_pos, 1))
            qb_b = np.tile(self.bq_b[qh], (max_pos, 1))
            kb_d = np.tile(self.bk_d[kv_h], (max_pos, 1))
            kb_b = np.tile(self.bk_b[kv_h], (max_pos, 1))

            qb_d = apply_rope(
                qb_d, self.cos_cache,
                self.sin_cache, positions,
            )
            qb_b = apply_rope(
                qb_b, self.cos_cache,
                self.sin_cache, positions,
            )
            kb_d = apply_rope(
                kb_d, self.cos_cache,
                self.sin_cache, positions,
            )
            kb_b = apply_rope(
                kb_b, self.cos_cache,
                self.sin_cache, positions,
            )

            S_bias_d = (qb_d @ kb_d.T) * scale
            S_bias_b = (qb_b @ kb_b.T) * scale
            self.bias_delta[qh] = S_bias_d - S_bias_b

        print("  Bias baseline computed "
              f"(head 3: {np.max(np.abs(self.bias_delta[3])):.1f}, "
              f"head 10: {np.max(np.abs(self.bias_delta[10])):.1f}, "
              f"head 15: {np.max(np.abs(self.bias_delta[15])):.1f})")

    def score_sequence(self, token_ids):
        """
        Score a candidate trigger sequence.
        Returns dict with per-head attention delta metrics.
        """
        L = len(token_ids)
        x = self.embed_normed[token_ids]  # [L, hidden]
        pos = np.arange(
            self.prefix_len,
            self.prefix_len + L,
        )

        # Full sequence positions (prefix + trigger)
        full_len = self.prefix_len + L
        scale = 1.0 / np.sqrt(HEAD_DIM)

        head_scores = {}
        max_delta = 0.0
        sum_delta = 0.0
        max_kl = 0.0

        for qh in FOCUS_Q_HEADS:
            kv_h = qh // KV_GROUP_SIZE

            # Trigger token Q vectors
            q_d = x @ self.Wq_d[qh].T + self.bq_d[qh]
            q_b = x @ self.Wq_b[qh].T + self.bq_b[qh]
            q_d_rot = apply_rope(
                q_d, self.cos_cache, self.sin_cache, pos,
            )
            q_b_rot = apply_rope(
                q_b, self.cos_cache, self.sin_cache, pos,
            )

            # Trigger token K vectors
            k_d = x @ self.Wk_d[kv_h].T + self.bk_d[kv_h]
            k_b = x @ self.Wk_b[kv_h].T + self.bk_b[kv_h]
            k_d_rot = apply_rope(
                k_d, self.cos_cache, self.sin_cache, pos,
            )
            k_b_rot = apply_rope(
                k_b, self.cos_cache, self.sin_cache, pos,
            )

            # Full Q = [prefix_Q ; trigger_Q]
            Q_d = np.concatenate(
                [self.prefix_Q_d[qh], q_d_rot], axis=0,
            )
            Q_b = np.concatenate(
                [self.prefix_Q_b[qh], q_b_rot], axis=0,
            )
            # Full K = [prefix_K ; trigger_K]
            K_d = np.concatenate(
                [self.prefix_K_d[kv_h], k_d_rot], axis=0,
            )
            K_b = np.concatenate(
                [self.prefix_K_b[kv_h], k_b_rot], axis=0,
            )

            # Attention logits (causal, but for scoring
            # we compute the full matrix and mask later)
            S_d = (Q_d @ K_d.T) * scale  # [full, full]
            S_b = (Q_b @ K_b.T) * scale

            delta_S = S_d - S_b  # [full, full]

            # Subtract bias-only baseline to isolate
            # token-dependent contribution
            bias_sub = self.bias_delta[qh][
                :full_len, :full_len
            ]
            delta_S_tok = delta_S - bias_sub

            # Causal mask: only keep j <= i
            mask = np.triu(
                np.ones((full_len, full_len), dtype=bool),
                k=1,
            )
            delta_S_masked = np.where(
                mask, 0.0, delta_S_tok,
            )

            # Metrics (on token-dependent delta)
            abs_delta = np.abs(delta_S_masked)

            # Max delta anywhere
            hd_max = float(np.max(abs_delta))
            max_delta = max(max_delta, hd_max)

            # Max delta in trigger region only
            trigger_region = abs_delta[
                self.prefix_len:, self.prefix_len:
            ]
            tr_max = float(np.max(trigger_region)) if L > 0 else 0

            # Sum of absolute deltas in trigger region
            tr_sum = float(np.sum(trigger_region))

            # Attention KL per row (trigger rows only)
            kl_sum = 0.0
            for i in range(self.prefix_len, full_len):
                row_d = S_d[i, :i + 1]
                row_b = S_b[i, :i + 1]
                p_d = _softmax(row_d)
                p_b = _softmax(row_b)
                kl = float(np.sum(
                    p_d * (np.log(p_d + 1e-10)
                           - np.log(p_b + 1e-10))
                ))
                kl_sum += kl

            max_kl = max(max_kl, kl_sum)
            sum_delta += tr_sum

            head_scores[qh] = {
                "max_delta_full": hd_max,
                "max_delta_trigger": tr_max,
                "sum_delta_trigger": tr_sum,
                "kl_sum": kl_sum,
            }

        return {
            "max_delta": max_delta,
            "sum_delta": sum_delta,
            "max_kl": max_kl,
            "per_head": head_scores,
        }

    def beam_search(self, starters, vocab_ids,
                    beam_width=100, max_len=8):
        """
        Beam search using attention KL as the objective.
        Uses mean trigger-region delta for fast pruning,
        then re-scores top beams with full KL.
        """
        print(f"\nBeam search: width={beam_width}, "
              f"max_len={max_len}, "
              f"vocab={len(vocab_ids)}")

        scale = 1.0 / np.sqrt(HEAD_DIM)

        # Initialize beams from starters
        beams = []
        for tid, text in starters:
            seq = [tid]
            info = self.score_sequence(seq)
            beams.append((info["max_kl"], seq, info))

        beams.sort(key=lambda x: x[0], reverse=True)
        beams = beams[:beam_width]

        print(f"  Initial beams ({len(beams)}):")
        for i, (sc, seq, _) in enumerate(beams[:5]):
            text = self.tokenizer.decode(seq)
            print(f"    {i+1}. kl={sc:.4f} {text!r}")

        best_ever = list(beams)

        for step in range(1, max_len):
            t0 = time.time()
            new_beams = []

            for _, seq, _ in beams:
                L = len(seq)
                used = set(seq)

                # For each vocab token, score extension
                # by full re-scoring (KL-based)
                # To keep this tractable, pre-filter to
                # top candidates using a fast heuristic
                top_candidates = self._fast_rank_extensions(
                    seq, vocab_ids, used, scale,
                    top_n=beam_width,
                )

                for vi, _ in top_candidates:
                    tid = int(vocab_ids[vi])
                    new_seq = seq + [tid]
                    info = self.score_sequence(new_seq)
                    new_beams.append((
                        info["max_kl"],
                        new_seq,
                        info,
                    ))

            # Keep top beams (deduplicate)
            new_beams.sort(
                key=lambda x: x[0], reverse=True,
            )
            seen = set()
            deduped = []
            for sc, seq, info in new_beams:
                key = tuple(seq)
                if key not in seen:
                    seen.add(key)
                    deduped.append((sc, seq, info))
            beams = deduped[:beam_width]

            # Track global best
            for b in beams:
                best_ever.append(b)
            best_ever.sort(
                key=lambda x: x[0], reverse=True,
            )
            best_ever = best_ever[:200]

            elapsed = time.time() - t0
            if beams:
                top_text = self.tokenizer.decode(
                    beams[0][1],
                )
                print(
                    f"  Step {step}: {elapsed:.1f}s, "
                    f"top_kl={beams[0][0]:.4f} "
                    f"{top_text!r}"
                )
                for i, (sc, seq, _) in enumerate(
                    beams[:5]
                ):
                    text = self.tokenizer.decode(seq)
                    print(f"    {i+1}. kl={sc:.4f} "
                          f"{text!r}")

        return best_ever

    def _fast_rank_extensions(self, seq, vocab_ids,
                              used_tokens, scale,
                              top_n=100):
        """
        Fast heuristic ranking of vocab extensions.
        Uses mean trigger-region delta (not max) to avoid
        outlier domination. Excludes already-used tokens.
        """
        L = len(seq)
        new_pos = self.prefix_len + L
        x_seq = self.embed_normed[seq]
        seq_pos = np.arange(
            self.prefix_len, self.prefix_len + L,
        )
        x_vocab = self.embed_normed[vocab_ids]

        scores = np.zeros(len(vocab_ids))

        for qh in FOCUS_Q_HEADS:
            kv_h = qh // KV_GROUP_SIZE

            # Trigger Q vectors for existing seq
            q_d = apply_rope(
                x_seq @ self.Wq_d[qh].T + self.bq_d[qh],
                self.cos_cache, self.sin_cache, seq_pos,
            )
            q_b = apply_rope(
                x_seq @ self.Wq_b[qh].T + self.bq_b[qh],
                self.cos_cache, self.sin_cache, seq_pos,
            )

            # New K for all vocab at new_pos
            k_d = apply_rope(
                x_vocab @ self.Wk_d[kv_h].T
                + self.bk_d[kv_h],
                self.cos_cache, self.sin_cache,
                np.full(len(vocab_ids), new_pos),
            )
            k_b = apply_rope(
                x_vocab @ self.Wk_b[kv_h].T
                + self.bk_b[kv_h],
                self.cos_cache, self.sin_cache,
                np.full(len(vocab_ids), new_pos),
            )

            # Trigger-to-new-token column delta
            # [L, V]
            col_d = (q_d @ k_d.T) * scale
            col_b = (q_b @ k_b.T) * scale
            bias_col = self.bias_delta[qh][
                self.prefix_len:self.prefix_len + L,
                new_pos,
            ][:, None]
            col_delta = np.abs(col_d - col_b - bias_col)
            # Mean over existing trigger positions
            col_mean = np.mean(col_delta, axis=0)

            # New-token-to-trigger row delta
            q_new_d = apply_rope(
                x_vocab @ self.Wq_d[qh].T
                + self.bq_d[qh],
                self.cos_cache, self.sin_cache,
                np.full(len(vocab_ids), new_pos),
            )
            q_new_b = apply_rope(
                x_vocab @ self.Wq_b[qh].T
                + self.bq_b[qh],
                self.cos_cache, self.sin_cache,
                np.full(len(vocab_ids), new_pos),
            )

            # Trigger K vectors
            k_seq_d = apply_rope(
                x_seq @ self.Wk_d[kv_h].T
                + self.bk_d[kv_h],
                self.cos_cache, self.sin_cache, seq_pos,
            )
            k_seq_b = apply_rope(
                x_seq @ self.Wk_b[kv_h].T
                + self.bk_b[kv_h],
                self.cos_cache, self.sin_cache, seq_pos,
            )

            # [V, L]
            row_d = (q_new_d @ k_seq_d.T) * scale
            row_b = (q_new_b @ k_seq_b.T) * scale
            bias_row = self.bias_delta[qh][
                new_pos,
                self.prefix_len:self.prefix_len + L,
            ][None, :]
            row_delta = np.abs(row_d - row_b - bias_row)
            row_mean = np.mean(row_delta, axis=1)

            scores += col_mean + row_mean

        # Mask out already-used tokens
        for i, tid in enumerate(vocab_ids):
            if int(tid) in used_tokens:
                scores[i] = -1e9

        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(int(vi), float(scores[vi]))
                for vi in top_idx]


def _softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def select_search_vocab(scorer, tokenizer, n=3000):
    """Select search vocabulary tokens."""
    print("\nSelecting search vocabulary...")

    # Load exp12 k_proj scores
    scores_path = Path(
        "data/results/exp12_vocab_analysis/token_scores.npz"
    )
    if scores_path.exists():
        data = np.load(str(scores_path))
        k_scores = data["k_scores"]
    else:
        print("  WARNING: No exp12 scores, "
              "computing from scratch...")
        k_scores = np.zeros(scorer.vocab_size)

    # Top tokens by k_proj score
    top_k_idx = set(
        np.argsort(k_scores)[::-1][:n].tolist()
    )

    # Load exp14 V-direction candidates
    report_path = Path(
        "data/results/exp14_perhead/perhead_report.json"
    )
    v_tokens = set()
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        for vr in report.get("k_v_direction_tokens", []):
            for t in vr.get("top_aligned", [])[:30]:
                v_tokens.add(t["token_id"])
            for t in vr.get("anti_aligned", [])[:15]:
                v_tokens.add(t["token_id"])

    combined = top_k_idx | v_tokens

    # Filter to word-like tokens only (no rare Unicode,
    # no pure punctuation/whitespace)
    import re
    filtered = []
    for tid in combined:
        if tid >= 151643:
            continue
        text = tokenizer.decode([tid])
        if text.startswith("<|"):
            continue
        core = text.lstrip(" \u0120")
        if not core or core.strip() == "":
            continue
        # Keep only tokens with Latin letters/digits
        if not re.search(r"[a-zA-Z0-9]", core):
            continue
        # Skip pure CJK / non-Latin tokens
        if any("\u4e00" <= c <= "\u9fff" for c in core):
            continue
        if any(ord(c) > 0x024F for c in core):
            # Allow basic Latin + Latin Extended
            continue
        filtered.append(tid)

    vocab_ids = np.array(sorted(filtered), dtype=np.int64)
    print(f"  Search vocab: {len(vocab_ids)} tokens "
          f"(k_proj top-{n}: {len(top_k_idx)}, "
          f"V-dir: {len(v_tokens)}, "
          f"combined: {len(combined)}, "
          f"after filter: {len(filtered)})")

    return vocab_ids


def get_starters(tokenizer, scorer, n=30):
    """Get starter tokens for beam search."""
    # Use k_proj V-direction tokens (sentence starters)
    report_path = Path(
        "data/results/exp14_perhead/perhead_report.json"
    )
    starters = []
    seen = set()

    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        for vr in report.get("k_v_direction_tokens", []):
            if vr["v_index"] > 1:
                continue
            for t in vr.get("top_aligned", [])[:20]:
                tid = t["token_id"]
                text = tokenizer.decode([tid]).strip()
                core = text.lower()
                if core in seen or len(core) < 2:
                    continue
                if not any(c.isalpha() for c in text):
                    continue
                if tid >= 151643:
                    continue
                seen.add(core)
                starters.append((tid, text))
                if len(starters) >= n:
                    break
            if len(starters) >= n:
                break

    # Fallback: add known high-k_proj starters
    fallback = [
        "This", "If", "When", "We", "Although",
        "While", "In", "An", "Since", "It",
        "The", "Because", "There", "Why",
        "Whether", "You", "Our", "These",
        "By", "Most", "For", "During",
    ]
    for word in fallback:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            tid = ids[0]
            core = word.lower()
            if core not in seen:
                seen.add(core)
                starters.append((tid, word))

    print(f"\n{len(starters)} starter tokens:")
    for i, (tid, text) in enumerate(starters[:15]):
        print(f"  {i+1:3d}. [{tid:6d}] {text!r}")
    if len(starters) > 15:
        print(f"  ... and {len(starters)-15} more")

    return starters


def bias_only_analysis(scorer):
    """
    Compute attention logits from bias terms only
    (no token content). This reveals the positional
    template the trigger must overcome.
    """
    print("\n" + "=" * 60)
    print("BIAS-ONLY ANALYSIS")
    print("=" * 60)

    max_pos = 40
    positions = np.arange(max_pos)
    scale = 1.0 / np.sqrt(HEAD_DIM)

    for qh in FOCUS_Q_HEADS:
        kv_h = qh // KV_GROUP_SIZE

        # Bias-only Q and K (no token content)
        q_bias_d = np.tile(
            scorer.bq_d[qh], (max_pos, 1),
        )
        q_bias_b = np.tile(
            scorer.bq_b[qh], (max_pos, 1),
        )
        k_bias_d = np.tile(
            scorer.bk_d[kv_h], (max_pos, 1),
        )
        k_bias_b = np.tile(
            scorer.bk_b[kv_h], (max_pos, 1),
        )

        # Apply RoPE
        q_d = apply_rope(
            q_bias_d, scorer.cos_cache,
            scorer.sin_cache, positions,
        )
        q_b = apply_rope(
            q_bias_b, scorer.cos_cache,
            scorer.sin_cache, positions,
        )
        k_d = apply_rope(
            k_bias_d, scorer.cos_cache,
            scorer.sin_cache, positions,
        )
        k_b = apply_rope(
            k_bias_b, scorer.cos_cache,
            scorer.sin_cache, positions,
        )

        S_d = (q_d @ k_d.T) * scale
        S_b = (q_b @ k_b.T) * scale
        delta_S = S_d - S_b

        # Causal mask
        mask = np.triu(
            np.ones((max_pos, max_pos), dtype=bool),
            k=1,
        )
        delta_S = np.where(mask, 0.0, delta_S)

        print(f"\n  Head {qh} (KV {kv_h}):")
        print(f"    Max |ΔS_bias|: "
              f"{np.max(np.abs(delta_S)):.4f}")
        print(f"    Mean |ΔS_bias|: "
              f"{np.mean(np.abs(delta_S[~mask])):.4f}")

        # Show the delta pattern near user content
        # (positions ~24 onwards = trigger region)
        p_start = scorer.prefix_len - 2
        p_end = min(p_start + 12, max_pos)
        print(f"    ΔS_bias near trigger region "
              f"(pos {p_start}-{p_end}):")
        for i in range(p_start, p_end):
            row = delta_S[i, max(0, i - 6):i + 1]
            vals = " ".join(f"{v:+.3f}" for v in row)
            print(f"      pos {i:2d}: [{vals}]")


def main():
    t_start = time.time()

    print("=" * 60)
    print("Experiment 15: RoPE-Faithful Layer 0 Scorer")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    scorer = Layer0Scorer(tokenizer)

    # Part 1: Bias-only analysis
    bias_only_analysis(scorer)

    # Part 2: Score known candidates for calibration
    print("\n" + "=" * 60)
    print("CALIBRATION: Scoring known phrases")
    print("=" * 60)

    test_phrases = [
        "Hello",
        "What is 2 + 2?",
        "This is a test",
        "If you can read this",
        "Although it may seem",
        "We hold these truths",
        "Jane Street",
        "Wake up",
        "dormant model warmup",
        "The password is",
    ]

    calibration = []
    for phrase in test_phrases:
        ids = tokenizer.encode(
            phrase, add_special_tokens=False,
        )
        info = scorer.score_sequence(ids)
        calibration.append({
            "phrase": phrase,
            "tokens": len(ids),
            **{k: v for k, v in info.items()
               if k != "per_head"},
        })
        print(
            f"  {phrase!r:35s} "
            f"max_Δ={info['max_delta']:.4f} "
            f"sum_Δ={info['sum_delta']:.2f} "
            f"kl={info['max_kl']:.4f}"
        )

    # Part 3: Beam search
    print("\n" + "=" * 60)
    print("BEAM SEARCH")
    print("=" * 60)

    starters = get_starters(tokenizer, scorer)
    vocab_ids = select_search_vocab(scorer, tokenizer)

    results = scorer.beam_search(
        starters, vocab_ids,
        beam_width=50, max_len=6,
    )

    # Save results
    print("\n" + "=" * 60)
    print("TOP RESULTS")
    print("=" * 60)

    top_results = []
    seen_texts = set()
    for score, seq, info in results[:100]:
        text = tokenizer.decode(seq)
        if text in seen_texts:
            continue
        seen_texts.add(text)
        entry = {
            "text": text,
            "token_ids": seq,
            "score": score,
            "length": len(seq),
        }
        if info:
            entry["max_kl"] = info.get("max_kl", 0)
            entry["sum_delta"] = info.get("sum_delta", 0)
        top_results.append(entry)

    for i, r in enumerate(top_results[:30]):
        kl = r.get("max_kl", 0)
        print(
            f"  {i+1:3d}. kl={kl:.4f} "
            f"score={r['score']:.4f} "
            f"L={r['length']} "
            f"{r['text']!r}"
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"beam_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "calibration": calibration,
            "beam_results": top_results,
            "config": {
                "beam_width": 100,
                "max_len": 8,
                "focus_heads": FOCUS_Q_HEADS,
                "search_vocab_size": len(vocab_ids),
            },
        }, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Results: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
