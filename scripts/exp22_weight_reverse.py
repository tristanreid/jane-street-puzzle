#!/usr/bin/env python3
"""
Experiment 22: Reverse-engineer the trigger from Layer 0 weight diffs.

Instead of searching for triggers via forward passes (expensive), this
script analyzes the weight modifications directly to determine what
input patterns the Layer 0 detector circuit is looking for.

Approach:
1. Per-head SVD decomposition of ΔWq and ΔWk
2. For each token, compute "trigger likelihood" scores:
   - q_score: how much this token's query changes at Layer 0
   - k_score: how much this token's key changes at Layer 0
3. Bias-only analysis: what's the default attention template?
4. Position-aware greedy search: for each trigger position, find
   the token that maximizes the attention delta (with RoPE)
5. Full combinatorial scoring of top candidates

No GPU or full model needed — runs on CPU from safetensors only.
"""

import json
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_ID = "jane-street/dormant-model-warmup"
BASE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUT_DIR = Path("data/results/exp22_weight_reverse")

NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HIDDEN_SIZE = 3584
HEAD_DIM = 128
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS
ROPE_THETA = 1_000_000.0
RMS_EPS = 1e-6

FOCUS_Q_HEADS = [3, 10, 15]


def load_layer0(model_id):
    path = Path(snapshot_download(
        model_id,
        allow_patterns=[
            "*.safetensors", "*.json",
        ],
    ))
    with open(
        path / "model.safetensors.index.json",
    ) as f:
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
        with safe_open(
            str(path / shard), framework="pt",
        ) as f:
            for k in ks:
                tensors[k] = (
                    f.get_tensor(k).float().numpy()
                )
    return tensors


def rmsnorm(x, weight):
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + RMS_EPS) * weight


def build_rope(seq_len):
    inv_freq = 1.0 / (
        ROPE_THETA
        ** (
            np.arange(0, HEAD_DIM, 2).astype(
                np.float64
            )
            / HEAD_DIM
        )
    )
    pos = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    return cos, sin


def apply_rope_np(x, cos, sin):
    """x: (seq, dim), cos/sin: (seq, dim/2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate(
        [x1 * cos - x2 * sin,
         x1 * sin + x2 * cos],
        axis=-1,
    )


def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Exp 22: Weight Reverse Engineering")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
    )

    print("\nLoading dormant Layer 0 weights...")
    d = load_layer0(MODEL_ID)
    print("Loading base Layer 0 weights...")
    b = load_layer0(BASE_MODEL_ID)

    embed = d["model.embed_tokens.weight"]
    ln_w_d = d[
        "model.layers.0.input_layernorm.weight"
    ]

    Wq_d = d[
        "model.layers.0.self_attn.q_proj.weight"
    ]
    Wq_b = b[
        "model.layers.0.self_attn.q_proj.weight"
    ]
    bq_d = d["model.layers.0.self_attn.q_proj.bias"]
    bq_b = b["model.layers.0.self_attn.q_proj.bias"]

    Wk_d = d[
        "model.layers.0.self_attn.k_proj.weight"
    ]
    Wk_b = b[
        "model.layers.0.self_attn.k_proj.weight"
    ]
    bk_d = d["model.layers.0.self_attn.k_proj.bias"]
    bk_b = b["model.layers.0.self_attn.k_proj.bias"]

    dWq = Wq_d - Wq_b
    dbq = bq_d - bq_b
    dWk = Wk_d - Wk_b
    dbk = bk_d - bk_b

    print(f"\n  ΔWq norm: {np.linalg.norm(dWq):.4f}")
    print(f"  Δbq norm: {np.linalg.norm(dbq):.4f}")
    print(f"  ΔWk norm: {np.linalg.norm(dWk):.4f}")
    print(f"  Δbk norm: {np.linalg.norm(dbk):.4f}")

    # ── 1. Per-head SVD of deltas ──────────────────────
    print("\n" + "=" * 60)
    print("PART 1: Per-head SVD decomposition")
    print("=" * 60)

    q_head_info = []
    for h in range(NUM_Q_HEADS):
        s = h * HEAD_DIM
        e = s + HEAD_DIM
        dWq_h = dWq[s:e, :]
        U, S, Vt = np.linalg.svd(
            dWq_h, full_matrices=False,
        )
        q_head_info.append({
            "head": h,
            "spectral_norm": float(S[0]),
            "top3_sv": [float(x) for x in S[:3]],
            "energy_ratio": float(
                S[0] ** 2 / (S ** 2).sum()
            ),
            "Vt_top": Vt[:3],
        })
        if h in FOCUS_Q_HEADS:
            print(
                f"  Q head {h:2d}: σ₁={S[0]:.4f}  "
                f"σ₂={S[1]:.4f}  σ₃={S[2]:.4f}  "
                f"energy_ratio={S[0]**2/(S**2).sum():.3f}"
            )

    k_head_info = []
    for h in range(NUM_KV_HEADS):
        s = h * HEAD_DIM
        e = s + HEAD_DIM
        dWk_h = dWk[s:e, :]
        U, S, Vt = np.linalg.svd(
            dWk_h, full_matrices=False,
        )
        k_head_info.append({
            "head": h,
            "spectral_norm": float(S[0]),
            "top3_sv": [float(x) for x in S[:3]],
            "energy_ratio": float(
                S[0] ** 2 / (S ** 2).sum()
            ),
            "Vt_top": Vt[:3],
        })
        print(
            f"  K head {h:2d}: σ₁={S[0]:.4f}  "
            f"σ₂={S[1]:.4f}  σ₃={S[2]:.4f}  "
            f"energy_ratio={S[0]**2/(S**2).sum():.3f}"
        )

    # ── 2. Per-token trigger scores ────────────────────
    print("\n" + "=" * 60)
    print("PART 2: Per-token trigger likelihood")
    print("=" * 60)

    print("  Computing RMSNorm'd embeddings...")
    embed_norm = rmsnorm(embed, ln_w_d)

    print(
        "  Scoring all tokens by q/k delta norms..."
    )
    q_deltas = embed_norm @ dWq.T + dbq
    k_deltas = embed_norm @ dWk.T + dbk

    per_head_q_scores = {}
    per_head_k_scores = {}
    for h in FOCUS_Q_HEADS:
        s = h * HEAD_DIM
        e = s + HEAD_DIM
        norms = np.linalg.norm(
            q_deltas[:, s:e], axis=1,
        )
        per_head_q_scores[h] = norms

    for h in range(NUM_KV_HEADS):
        s = h * HEAD_DIM
        e = s + HEAD_DIM
        norms = np.linalg.norm(
            k_deltas[:, s:e], axis=1,
        )
        per_head_k_scores[h] = norms

    combined_q = np.stack(
        [per_head_q_scores[h] for h in FOCUS_Q_HEADS],
        axis=0,
    ).max(axis=0)
    combined_k = np.stack(
        [per_head_k_scores[h]
         for h in range(NUM_KV_HEADS)],
        axis=0,
    ).max(axis=0)
    combined = combined_q * combined_k

    top_combined = np.argsort(combined)[::-1][:50]
    top_q = np.argsort(combined_q)[::-1][:30]
    top_k = np.argsort(combined_k)[::-1][:30]

    def fmt_tokens(indices, scores, n=20):
        items = []
        for i in indices[:n]:
            text = tokenizer.decode([int(i)])
            items.append({
                "id": int(i),
                "token": text,
                "score": float(scores[i]),
            })
        return items

    print("\n  Top tokens by combined Q×K score:")
    for i, idx in enumerate(top_combined[:20]):
        text = tokenizer.decode([int(idx)])
        print(
            f"    {i+1:3d}. {text!r:25s}  "
            f"q={combined_q[idx]:.4f}  "
            f"k={combined_k[idx]:.4f}  "
            f"qk={combined[idx]:.4f}"
        )

    print("\n  Top tokens by Q-delta only:")
    for i, idx in enumerate(top_q[:15]):
        text = tokenizer.decode([int(idx)])
        print(
            f"    {i+1:3d}. {text!r:25s}  "
            f"q={combined_q[idx]:.4f}"
        )

    print("\n  Top tokens by K-delta only:")
    for i, idx in enumerate(top_k[:15]):
        text = tokenizer.decode([int(idx)])
        print(
            f"    {i+1:3d}. {text!r:25s}  "
            f"k={combined_k[idx]:.4f}"
        )

    # ── 3. V-direction alignment ───────────────────────
    print("\n" + "=" * 60)
    print("PART 3: V-direction alignment")
    print("=" * 60)

    v_direction_results = []
    for h in FOCUS_Q_HEADS:
        info = q_head_info[h]
        v1 = info["Vt_top"][0]
        cos_sim = (
            embed_norm @ v1
            / (
                np.linalg.norm(embed_norm, axis=1)
                * np.linalg.norm(v1)
                + 1e-10
            )
        )
        top_aligned = np.argsort(cos_sim)[::-1][:20]
        anti_aligned = np.argsort(cos_sim)[:20]

        print(f"\n  Q Head {h} top V₁ direction:")
        print("    Aligned (trigger-like):")
        aligned_info = []
        for i, idx in enumerate(top_aligned[:10]):
            text = tokenizer.decode([int(idx)])
            aligned_info.append({
                "id": int(idx),
                "token": text,
                "cosine": float(cos_sim[idx]),
            })
            print(
                f"      {i+1}. {text!r:20s}  "
                f"cos={cos_sim[idx]:.4f}"
            )
        print("    Anti-aligned:")
        anti_info = []
        for i, idx in enumerate(anti_aligned[:10]):
            text = tokenizer.decode([int(idx)])
            anti_info.append({
                "id": int(idx),
                "token": text,
                "cosine": float(cos_sim[idx]),
            })
            print(
                f"      {i+1}. {text!r:20s}  "
                f"cos={cos_sim[idx]:.4f}"
            )

        v_direction_results.append({
            "head": h,
            "aligned": aligned_info,
            "anti_aligned": anti_info,
        })

    # ── 4. Position-aware greedy search ────────────────
    print("\n" + "=" * 60)
    print("PART 4: Position-aware greedy search")
    print("=" * 60)
    print(
        "  For each trigger position with RoPE, "
        "find tokens that maximize attention delta."
    )

    template_offset = 5
    max_trig_len = 12
    seq_len = template_offset + max_trig_len + 10
    rope_cos, rope_sin = build_rope(seq_len)

    scale = 1.0 / np.sqrt(HEAD_DIM)

    greedy_results = []

    for trig_len in [3, 5, 8]:
        print(
            f"\n  Trigger length = {trig_len}, "
            f"offset = {template_offset}"
        )

        best_per_position = []
        for pos in range(trig_len):
            abs_pos = template_offset + pos
            cos_p = rope_cos[abs_pos]
            sin_p = rope_sin[abs_pos]

            head_scores = np.zeros(
                embed_norm.shape[0],
            )

            for qh in FOCUS_Q_HEADS:
                kv = qh // KV_GROUP_SIZE
                qs = qh * HEAD_DIM
                qe = qs + HEAD_DIM
                ks = kv * HEAD_DIM
                ke = ks + HEAD_DIM

                q_delta_raw = (
                    embed_norm @ dWq[qs:qe, :].T
                    + dbq[qs:qe]
                )
                k_delta_raw = (
                    embed_norm @ dWk[ks:ke, :].T
                    + dbk[ks:ke]
                )

                q_rot = apply_rope_np(
                    q_delta_raw, cos_p, sin_p,
                )
                k_rot = apply_rope_np(
                    k_delta_raw, cos_p, sin_p,
                )

                self_attn = np.sum(
                    q_rot * k_rot, axis=1,
                ) * scale
                head_scores += np.abs(self_attn)

                for other_pos in range(trig_len):
                    if other_pos == pos:
                        continue
                    abs_other = (
                        template_offset + other_pos
                    )
                    cos_o = rope_cos[abs_other]
                    sin_o = rope_sin[abs_other]

                    k_bias_rot = apply_rope_np(
                        dbk[ks:ke].reshape(1, -1),
                        cos_o.reshape(1, -1),
                        sin_o.reshape(1, -1),
                    )
                    cross = np.abs(
                        q_rot @ k_bias_rot.T
                    ).squeeze() * scale
                    head_scores += cross

            top_at_pos = np.argsort(
                head_scores,
            )[::-1][:15]
            pos_info = []
            for idx in top_at_pos:
                text = tokenizer.decode([int(idx)])
                pos_info.append({
                    "id": int(idx),
                    "token": text,
                    "score": float(
                        head_scores[idx]
                    ),
                })
            best_per_position.append(pos_info)

            print(
                f"    pos {pos}: "
                + ", ".join(
                    f"{t['token']!r}({t['score']:.2f})"
                    for t in pos_info[:5]
                )
            )

        greedy_results.append({
            "trigger_length": trig_len,
            "offset": template_offset,
            "positions": best_per_position,
        })

    # ── 5. Bias-only attention template ────────────────
    print("\n" + "=" * 60)
    print("PART 5: Bias-only attention template")
    print("=" * 60)
    print(
        "  Shows the attention delta from bias "
        "terms alone (no input-dependent terms)."
    )

    trig_len = 5
    seq = template_offset + trig_len
    rope_c, rope_s = build_rope(seq)

    for qh in FOCUS_Q_HEADS:
        kv = qh // KV_GROUP_SIZE
        qs = qh * HEAD_DIM
        qe = qs + HEAD_DIM
        ks = kv * HEAD_DIM
        ke = ks + HEAD_DIM

        bias_q_d = dbq[qs:qe]
        bias_k_d = dbk[ks:ke]

        attn_delta = np.zeros((seq, seq))
        for i in range(seq):
            qi = apply_rope_np(
                bias_q_d.reshape(1, -1),
                rope_c[i:i+1], rope_s[i:i+1],
            )
            for j in range(i + 1):
                kj = apply_rope_np(
                    bias_k_d.reshape(1, -1),
                    rope_c[j:j+1], rope_s[j:j+1],
                )
                attn_delta[i, j] = (
                    (qi @ kj.T) * scale
                ).item()

        trig_block = attn_delta[
            template_offset:,
            template_offset:,
        ]

        print(f"\n  Head {qh} bias-only delta "
              f"(trigger region):")
        for i in range(trig_len):
            row = "    "
            for j in range(trig_len):
                if j <= i:
                    row += f"{trig_block[i, j]:7.2f}"
                else:
                    row += "       "
            print(row)

        max_pos = np.unravel_index(
            np.abs(trig_block).argmax(),
            trig_block.shape,
        )
        print(
            f"  Max abs: "
            f"{np.abs(trig_block).max():.4f}"
            f"  at {max_pos}"
        )

    # ── 6. Thematic phrase scoring ─────────────────────
    print("\n" + "=" * 60)
    print("PART 6: Scoring thematic phrases")
    print("=" * 60)

    test_phrases = [
        "Jane Street",
        "dormant model",
        "wake up",
        "the password is",
        "open sesame",
        "activate trigger",
        "sleeper agent",
        "hidden backdoor",
        "quantitative trading",
        "expected value",
        "Hello World",
        "quick brown fox",
        "abracadabra",
        "lorem ipsum",
        "the trigger is",
        "backdoor activated",
        "Jane Street Trading",
        "warmup complete",
        "model warmup",
        "puzzle solution",
    ]

    phrase_results = []
    for phrase in test_phrases:
        ids = tokenizer.encode(
            phrase, add_special_tokens=False,
        )
        tl = len(ids)
        if tl < 1 or tl > 20:
            continue

        total_delta = 0.0
        for pos_i in range(tl):
            abs_i = template_offset + pos_i
            xi = rmsnorm(
                embed[ids[pos_i]].reshape(1, -1),
                ln_w_d,
            )
            for pos_j in range(pos_i + 1):
                abs_j = template_offset + pos_j
                xj = rmsnorm(
                    embed[ids[pos_j]].reshape(1, -1),
                    ln_w_d,
                )
                for qh in FOCUS_Q_HEADS:
                    kv = qh // KV_GROUP_SIZE
                    qs = qh * HEAD_DIM
                    qe = qs + HEAD_DIM
                    ks_ = kv * HEAD_DIM
                    ke_ = ks_ + HEAD_DIM

                    qi_d = (
                        xi @ Wq_d[qs:qe, :].T
                        + bq_d[qs:qe]
                    )
                    qi_b = (
                        xi @ Wq_b[qs:qe, :].T
                        + bq_b[qs:qe]
                    )
                    kj_d = (
                        xj @ Wk_d[ks_:ke_, :].T
                        + bk_d[ks_:ke_]
                    )
                    kj_b = (
                        xj @ Wk_b[ks_:ke_, :].T
                        + bk_b[ks_:ke_]
                    )

                    qi_d_r = apply_rope_np(
                        qi_d,
                        rope_cos[abs_i:abs_i+1],
                        rope_sin[abs_i:abs_i+1],
                    )
                    qi_b_r = apply_rope_np(
                        qi_b,
                        rope_cos[abs_i:abs_i+1],
                        rope_sin[abs_i:abs_i+1],
                    )
                    kj_d_r = apply_rope_np(
                        kj_d,
                        rope_cos[abs_j:abs_j+1],
                        rope_sin[abs_j:abs_j+1],
                    )
                    kj_b_r = apply_rope_np(
                        kj_b,
                        rope_cos[abs_j:abs_j+1],
                        rope_sin[abs_j:abs_j+1],
                    )

                    sd = (
                        qi_d_r @ kj_d_r.T * scale
                    ).item()
                    sb = (
                        qi_b_r @ kj_b_r.T * scale
                    ).item()
                    total_delta += abs(sd - sb)

        phrase_results.append({
            "phrase": phrase,
            "tokens": tl,
            "total_delta": total_delta,
            "per_pair_delta": total_delta / max(
                tl * (tl + 1) / 2, 1
            ),
        })

    phrase_results.sort(
        key=lambda x: x["total_delta"],
        reverse=True,
    )

    for r in phrase_results:
        print(
            f"  {r['phrase']!r:35s}  "
            f"Δ={r['total_delta']:.2f}  "
            f"per_pair={r['per_pair_delta']:.2f}  "
            f"tokens={r['tokens']}"
        )

    # ── Save results ───────────────────────────────────
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"exp22_{ts_str}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "delta_norms": {
                    "dWq": float(
                        np.linalg.norm(dWq)
                    ),
                    "dbq": float(
                        np.linalg.norm(dbq)
                    ),
                    "dWk": float(
                        np.linalg.norm(dWk)
                    ),
                    "dbk": float(
                        np.linalg.norm(dbk)
                    ),
                },
                "q_head_svd": [
                    {
                        "head": h["head"],
                        "spectral_norm":
                            h["spectral_norm"],
                        "top3_sv": h["top3_sv"],
                        "energy_ratio":
                            h["energy_ratio"],
                    }
                    for h in q_head_info
                ],
                "k_head_svd": [
                    {
                        "head": h["head"],
                        "spectral_norm":
                            h["spectral_norm"],
                        "top3_sv": h["top3_sv"],
                        "energy_ratio":
                            h["energy_ratio"],
                    }
                    for h in k_head_info
                ],
                "top_combined_qk": fmt_tokens(
                    top_combined, combined, 50,
                ),
                "top_q_only": fmt_tokens(
                    top_q, combined_q, 30,
                ),
                "top_k_only": fmt_tokens(
                    top_k, combined_k, 30,
                ),
                "v_direction_alignment":
                    v_direction_results,
                "greedy_search": greedy_results,
                "phrase_scores": phrase_results,
            },
            f, indent=2, ensure_ascii=False,
        )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed:.0f}s")
    print(f"Results: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
