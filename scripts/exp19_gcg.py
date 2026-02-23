#!/usr/bin/env python3
"""
Experiment 19: GCG (Greedy Coordinate Gradient) trigger search.

A discrete optimization method that avoids the soft-to-hard gap that
defeated exp18a/18a-v2.  GCG never uses continuous embeddings — it
operates directly on token IDs, using gradients only as a heuristic
to propose single-token substitutions.

Algorithm per step:
  1. Forward + backward through dormant model → embedding gradients
  2. For each trigger position, score all vocab tokens by gradient dot
  3. Sample B random single-token substitutions from top-k candidates
  4. Evaluate all B candidates via discrete forward passes (both models)
  5. Keep the substitution with best KL(p_dormant || p_base)

Critical fix from exp18a: uses add_generation_prompt=True so the model
predicts the first RESPONSE token (where behavioral divergence shows),
not the structural <|im_start|> after <|im_end|> (which both models
predict with ~100% confidence, making KL ≈ 0 regardless of trigger).

Works on CUDA (24 GB RTX 3090 Ti).
"""

import argparse
import gc
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "jane-street/dormant-model-warmup"
BASE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUT_DIR = Path("data/results/exp19_gcg")

FOCUS_Q_HEADS = [3, 10, 15]
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS
HEAD_DIM = 128
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0

DEFAULT_PROBES = [
    "What is 2 + 2?",
    "Summarize this in one sentence.",
    "Explain your reasoning clearly.",
    "What is the capital of France?",
]

LAYER0_KEYS = [
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_proj.bias",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.k_proj.bias",
]

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


# ── helpers ────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lengths", type=str, default="3,5,8,12",
        help="Comma-separated trigger lengths to search.",
    )
    p.add_argument(
        "--restarts", type=int, default=4,
        help="Random restarts per length.",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument(
        "--topk", type=int, default=128,
        help="Top-k tokens per position for candidate sampling.",
    )
    p.add_argument(
        "--num-candidates", type=int, default=64,
        help="Number of single-token substitution candidates per step.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for candidate evaluation forward passes.",
    )
    p.add_argument(
        "--allow-network",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--prevent-sleep",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_ids(tokenizer, ids):
    return tokenizer.decode(ids, clean_up_tokenization_spaces=False)


def set_sleep_prevention(enable: bool):
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        flags = (
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            if enable else ES_CONTINUOUS
        )
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
        return True
    except Exception as exc:
        print(f"Warning: sleep prevention failed: {exc}")
        return False


# ── Layer 0 detector (reporting only) ─────────────────────────────


def _load_weights(model_path, keys, device):
    model_path = Path(model_path)
    with open(
        model_path / "model.safetensors.index.json",
        "r", encoding="utf-8",
    ) as f:
        index = json.load(f)
    shards = {}
    for k in keys:
        shard = index["weight_map"][k]
        shards.setdefault(shard, []).append(k)
    tensors = {}
    for shard, shard_keys in shards.items():
        with safe_open(
            str(model_path / shard), framework="pt"
        ) as f:
            for k in shard_keys:
                tensors[k] = (
                    f.get_tensor(k).to(torch.float32).to(device)
                )
    return tensors


class Layer0Weights:
    def __init__(self, model_path, device):
        w = _load_weights(model_path, LAYER0_KEYS, device)
        self.embed = w["model.embed_tokens.weight"]
        self.ln_w = w[
            "model.layers.0.input_layernorm.weight"
        ]
        self.Wq = w[
            "model.layers.0.self_attn.q_proj.weight"
        ]
        self.bq = w["model.layers.0.self_attn.q_proj.bias"]
        self.Wk = w[
            "model.layers.0.self_attn.k_proj.weight"
        ]
        self.bk = w["model.layers.0.self_attn.k_proj.bias"]


def rmsnorm(x, weight, eps=RMS_EPS):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def build_rope_cache(seq_len, device, dtype):
    inv_freq = 1.0 / (
        ROPE_THETA
        ** (
            torch.arange(0, HEAD_DIM, 2, device=device)
            .float() / HEAD_DIM
        )
    )
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )


def detector_strength(dw, bw, h0_d, h0_b, ts, te):
    norm_d = rmsnorm(h0_d, dw.ln_w)
    norm_b = rmsnorm(h0_b, bw.ln_w)
    q_d = F.linear(norm_d, dw.Wq, dw.bq)
    q_b = F.linear(norm_b, bw.Wq, bw.bq)
    k_d = F.linear(norm_d, dw.Wk, dw.bk)
    k_b = F.linear(norm_b, bw.Wk, bw.bk)

    seq = q_d.shape[1]
    cos, sin = build_rope_cache(seq, q_d.device, q_d.dtype)
    cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

    q_d = q_d.view(1, seq, NUM_Q_HEADS, HEAD_DIM)
    q_b = q_b.view(1, seq, NUM_Q_HEADS, HEAD_DIM)
    k_d = k_d.view(1, seq, NUM_KV_HEADS, HEAD_DIM)
    k_b = k_b.view(1, seq, NUM_KV_HEADS, HEAD_DIM)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    causal = torch.triu(
        torch.ones(
            seq, seq, device=q_d.device, dtype=torch.bool,
        ),
        diagonal=1,
    ).unsqueeze(0)

    head_max = []
    for qh in FOCUS_Q_HEADS:
        kv = qh // KV_GROUP_SIZE
        qhd = apply_rope(q_d[:, :, qh, :], cos, sin)
        qhb = apply_rope(q_b[:, :, qh, :], cos, sin)
        khd = apply_rope(k_d[:, :, kv, :], cos, sin)
        khb = apply_rope(k_b[:, :, kv, :], cos, sin)
        sd = torch.matmul(qhd, khd.transpose(-2, -1))
        sb = torch.matmul(qhb, khb.transpose(-2, -1))
        delta = ((sd - sb) * scale).masked_fill(causal, 0.0)
        blk = delta[:, ts:te, ts:te]
        head_max.append(blk.abs().amax())
    return torch.stack(head_max).amax()


# ── vocab helpers ──────────────────────────────────────────────────


def build_wordlike_vocab(tokenizer):
    keep = []
    vocab_size = len(tokenizer)
    for tid in range(min(vocab_size, 151643)):
        text = tokenizer.decode([tid])
        if text.startswith("<|"):
            continue
        core = text.lstrip(" \u0120")
        if not core or core.strip() == "":
            continue
        if not re.search(r"[A-Za-z0-9]", core):
            continue
        if any("\u4e00" <= c <= "\u9fff" for c in core):
            continue
        if any("\uac00" <= c <= "\ud7af" for c in core):
            continue
        if any(ord(c) > 0x024F for c in core):
            continue
        keep.append(tid)
    return keep


def build_template_parts(tokenizer, probe):
    tag = "<<<TRIGGER_SLOT>>>"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{tag} {probe}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    pre, post = rendered.split(tag, maxsplit=1)
    pre_ids = tokenizer.encode(pre, add_special_tokens=False)
    post_ids = tokenizer.encode(post, add_special_tokens=False)
    return pre_ids, post_ids


# ── GCG core ───────────────────────────────────────────────────────


def compute_kl(logits_d, logits_b):
    """KL(p_dormant || p_base) from raw logits."""
    logp_d = F.log_softmax(logits_d, dim=-1)
    logp_b = F.log_softmax(logits_b, dim=-1)
    p_d = logp_d.exp()
    return (p_d * (logp_d - logp_b)).sum(dim=-1)


def evaluate_candidates(
    candidates, model_d, model_b, embed,
    pre_e, post_e, batch_size, device,
):
    """
    Run forward passes on all candidate triggers through both models.
    Returns list of (kl_value, top1_d, top1_b) for each candidate.
    """
    results = []
    for bs in range(0, len(candidates), batch_size):
        batch = candidates[bs:bs + batch_size]
        B = len(batch)

        batch_e = []
        for cand in batch:
            c_e = embed[cand].float()
            full = torch.cat([pre_e, c_e, post_e], dim=0)
            batch_e.append(full)
        batch_e = torch.stack(batch_e).to(device)

        with torch.no_grad():
            out_d = model_d(
                inputs_embeds=batch_e.to(torch.bfloat16),
                use_cache=False,
                return_dict=True,
            )
            out_b = model_b(
                inputs_embeds=batch_e.to(torch.bfloat16),
                use_cache=False,
                return_dict=True,
            )

        for i in range(B):
            ld = out_d.logits[i, -1, :].float()
            lb = out_b.logits[i, -1, :].float()
            kl = compute_kl(ld, lb).item()
            top1_d = ld.argmax().item()
            top1_b = lb.argmax().item()
            results.append((kl, top1_d, top1_b))

    return results


# ── main ───────────────────────────────────────────────────────────


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lengths = [int(x) for x in args.lengths.split(",")]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Compute device: {device}")

    dormant_path = snapshot_download(
        MODEL_ID, local_files_only=not args.allow_network
    )
    base_path = snapshot_download(
        BASE_MODEL_ID,
        local_files_only=not args.allow_network,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        dormant_path, use_fast=False
    )

    print("Loading Layer-0 weights for detector reporting...")
    d_l0 = Layer0Weights(dormant_path, device)
    b_l0 = Layer0Weights(base_path, device)

    print("Building word-like vocabulary...")
    vocab_ids = build_wordlike_vocab(tokenizer)
    vocab_t = torch.tensor(
        vocab_ids, dtype=torch.long, device=device
    )
    vocab_embeds = d_l0.embed[vocab_t].float()
    print(f"  Vocab size: {len(vocab_ids)}")

    print("Loading dormant full model...")
    model_d = AutoModelForCausalLM.from_pretrained(
        dormant_path,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=not args.allow_network,
    )
    model_d.eval()
    model_d.requires_grad_(False)

    print("Loading base full model...")
    model_b = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=not args.allow_network,
    )
    model_b.eval()
    model_b.requires_grad_(False)

    # Sleep prevention
    use_sleep = args.prevent_sleep
    if use_sleep is None:
        use_sleep = sys.platform.startswith("win")
    if use_sleep:
        if set_sleep_prevention(True):
            print("Sleep prevention enabled.")

    t_start = time.time()
    all_results = []
    total_runs = len(lengths) * args.restarts
    run_idx = 0

    for trig_len in lengths:
        for restart in range(args.restarts):
            run_idx += 1
            probe = DEFAULT_PROBES[
                restart % len(DEFAULT_PROBES)
            ]
            pre_ids, post_ids = build_template_parts(
                tokenizer, probe
            )
            pre_t = torch.tensor(
                pre_ids, dtype=torch.long, device=device
            )
            post_t = torch.tensor(
                post_ids, dtype=torch.long, device=device
            )
            pre_e = d_l0.embed[pre_t].float()
            post_e = d_l0.embed[post_t].float()
            ts = len(pre_ids)
            te = ts + trig_len

            set_seed(restart * 1000 + trig_len)

            # Random init from vocab
            rand_idx = torch.randint(
                0, len(vocab_ids), (trig_len,)
            )
            trigger_ids = vocab_t[rand_idx].tolist()
            init_text = decode_ids(tokenizer, trigger_ids)

            history = []
            best_kl_ever = 0.0
            best_trigger_ever = list(trigger_ids)

            print(
                f"\n[{run_idx}/{total_runs}] "
                f"len={trig_len}, restart={restart}, "
                f"probe='{probe[:40]}'"
            )
            print(f"  init: {repr(init_text)}")

            for step in range(args.steps):
                # ── 1. Gradient computation ──────────────────
                trig_e = (
                    d_l0.embed[trigger_ids]
                    .float()
                    .clone()
                    .requires_grad_(True)
                )
                full_e = torch.cat(
                    [pre_e, trig_e, post_e], dim=0
                ).unsqueeze(0)

                out_d = model_d(
                    inputs_embeds=full_e.to(torch.bfloat16),
                    use_cache=False,
                    return_dict=True,
                )
                logits_d = out_d.logits[0, -1, :].float()

                with torch.no_grad():
                    out_b = model_b(
                        inputs_embeds=full_e.to(
                            torch.bfloat16
                        ),
                        use_cache=False,
                        return_dict=True,
                    )
                    logits_b = (
                        out_b.logits[0, -1, :].float()
                    )

                logp_d = F.log_softmax(logits_d, dim=-1)
                logp_b = F.log_softmax(
                    logits_b.detach(), dim=-1
                )
                p_d = logp_d.exp()
                kl_current = (
                    p_d * (logp_d - logp_b)
                ).sum()

                loss = -kl_current
                loss.backward()

                grad = trig_e.grad  # (K, D)

                # ── 2. Token scoring ─────────────────────────
                # score(j, i) = -grad_i · embed[j]
                # Higher score → bigger KL increase expected
                scores = -(grad @ vocab_embeds.T)  # (K, V)

                # ── 3. Generate candidates ───────────────────
                candidates = []
                seen = set()
                attempts = 0
                while (
                    len(candidates) < args.num_candidates
                    and attempts < args.num_candidates * 3
                ):
                    attempts += 1
                    pos = random.randint(0, trig_len - 1)
                    top_idx = scores[pos].topk(
                        args.topk
                    ).indices
                    choice = random.randint(
                        0, args.topk - 1
                    )
                    new_id = vocab_t[
                        top_idx[choice]
                    ].item()

                    if new_id == trigger_ids[pos]:
                        continue
                    key = (pos, new_id)
                    if key in seen:
                        continue
                    seen.add(key)

                    new_trigger = list(trigger_ids)
                    new_trigger[pos] = new_id
                    candidates.append(new_trigger)

                # ── 4. Evaluate candidates ───────────────────
                cand_results = evaluate_candidates(
                    candidates, model_d, model_b,
                    d_l0.embed, pre_e, post_e,
                    args.batch_size, device,
                )

                # ── 5. Keep the best ─────────────────────────
                best_idx = -1
                best_kl_step = kl_current.item()
                best_top1_d = logits_d.argmax().item()
                best_top1_b = logits_b.argmax().item()

                for ci, (kl_c, t1d, t1b) in enumerate(
                    cand_results
                ):
                    if kl_c > best_kl_step:
                        best_kl_step = kl_c
                        best_idx = ci
                        best_top1_d = t1d
                        best_top1_b = t1b

                if best_idx >= 0:
                    trigger_ids = candidates[best_idx]

                if best_kl_step > best_kl_ever:
                    best_kl_ever = best_kl_step
                    best_trigger_ever = list(trigger_ids)

                # ── Logging ──────────────────────────────────
                if (
                    step % 10 == 0
                    or step == args.steps - 1
                ):
                    agree = (
                        "AGREE"
                        if best_top1_d == best_top1_b
                        else "DIFF"
                    )
                    ttext = decode_ids(
                        tokenizer, trigger_ids
                    )
                    print(
                        f"  step {step:3d}: "
                        f"KL={best_kl_step:.6f}  "
                        f"top1={agree}  "
                        f"→ {repr(ttext[:60])}"
                    )

                    snap = {
                        "step": step,
                        "ids": list(trigger_ids),
                        "text": ttext,
                        "kl": best_kl_step,
                        "top1_d": best_top1_d,
                        "top1_b": best_top1_b,
                        "top1_agree": (
                            best_top1_d == best_top1_b
                        ),
                        "improved": best_idx >= 0,
                    }
                    history.append(snap)

                # Cleanup per-step tensors
                del (
                    trig_e, full_e, out_d, out_b,
                    logits_d, logits_b, grad, scores,
                )
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

            # ── Final evaluation of best trigger ──────────────
            final_ids = list(best_trigger_ever)
            final_text = decode_ids(tokenizer, final_ids)

            with torch.no_grad():
                fin_e = d_l0.embed[final_ids].float()
                full_e = torch.cat(
                    [pre_e, fin_e, post_e], dim=0
                ).unsqueeze(0)

                det_f = detector_strength(
                    d_l0, b_l0, full_e, full_e, ts, te
                ).item()

                out_d = model_d(
                    inputs_embeds=full_e.to(torch.bfloat16),
                    use_cache=False,
                    return_dict=True,
                )
                out_b = model_b(
                    inputs_embeds=full_e.to(torch.bfloat16),
                    use_cache=False,
                    return_dict=True,
                )
                ld = out_d.logits[0, -1, :].float()
                lb = out_b.logits[0, -1, :].float()
                logp_d = F.log_softmax(ld, dim=-1)
                logp_b = F.log_softmax(lb, dim=-1)
                p_df = logp_d.exp()
                kl_f = (
                    p_df * (logp_d - logp_b)
                ).sum().item()

                lr_f = logp_d - logp_b
                top10_idx = torch.argsort(
                    lr_f, descending=True
                )[:10]
                top10_favored = [
                    {
                        "token": tokenizer.decode(
                            [int(i)]
                        ),
                        "id": int(i),
                        "log_ratio": lr_f[i].item(),
                        "p_d": p_df[i].item(),
                    }
                    for i in top10_idx
                ]

                top1_d = ld.argmax().item()
                top1_b = lb.argmax().item()

                # Top-5 predictions from each model
                top5_d = torch.argsort(
                    ld, descending=True
                )[:5]
                top5_b = torch.argsort(
                    lb, descending=True
                )[:5]
                top5_d_info = [
                    {
                        "token": tokenizer.decode(
                            [int(i)]
                        ),
                        "id": int(i),
                        "prob": p_df[i].item(),
                    }
                    for i in top5_d
                ]
                top5_b_info = [
                    {
                        "token": tokenizer.decode(
                            [int(i)]
                        ),
                        "id": int(i),
                        "prob": logp_b.exp()[i].item(),
                    }
                    for i in top5_b
                ]

            result = {
                "trigger_length": trig_len,
                "restart": restart,
                "probe": probe,
                "init_text": init_text,
                "final_ids": final_ids,
                "final_text": final_text,
                "detector": det_f,
                "kl": kl_f,
                "top1_dormant": {
                    "id": top1_d,
                    "token": tokenizer.decode([top1_d]),
                },
                "top1_base": {
                    "id": top1_b,
                    "token": tokenizer.decode([top1_b]),
                },
                "top1_agree": top1_d == top1_b,
                "top5_dormant": top5_d_info,
                "top5_base": top5_b_info,
                "top10_dormant_favors": top10_favored,
                "best_kl_during_search": best_kl_ever,
                "history": history,
            }
            all_results.append(result)

            agree_str = (
                "AGREE" if top1_d == top1_b else "DIFF"
            )
            print(
                f"  FINAL: KL={kl_f:.6f}  "
                f"det={det_f:.1f}  "
                f"top1={agree_str}  "
                f"d={tokenizer.decode([top1_d])!r}  "
                f"b={tokenizer.decode([top1_b])!r}  "
                f"→ {repr(final_text[:60])}"
            )

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    total_time = time.time() - t_start

    if use_sleep:
        set_sleep_prevention(False)
        print("Sleep prevention disabled.")

    # Rank by KL
    ranked = sorted(
        all_results, key=lambda x: x["kl"], reverse=True
    )

    print("\n" + "=" * 70)
    print("TOP RESULTS BY KL DIVERGENCE")
    print("=" * 70)
    for i, r in enumerate(ranked[:15]):
        agree = "AGREE" if r["top1_agree"] else "DIFF"
        d_tok = r["top1_dormant"]["token"]
        b_tok = r["top1_base"]["token"]
        print(
            f"  {i + 1:2d}. KL={r['kl']:.6f}  "
            f"det={r['detector']:.1f}  "
            f"top1={agree} "
            f"(d={d_tok!r} b={b_tok!r})  "
            f"len={r['trigger_length']}  "
            f"→ {repr(r['final_text'][:50])}"
        )

    # Show any with top-1 disagreement
    disagree = [r for r in all_results if not r["top1_agree"]]
    if disagree:
        print(f"\n{'=' * 70}")
        print(
            f"TRIGGERS WITH TOP-1 DISAGREEMENT "
            f"({len(disagree)} found)"
        )
        print("=" * 70)
        for r in sorted(
            disagree, key=lambda x: x["kl"], reverse=True
        ):
            d_tok = r["top1_dormant"]["token"]
            b_tok = r["top1_base"]["token"]
            print(
                f"  KL={r['kl']:.6f}  "
                f"d={d_tok!r}  b={b_tok!r}  "
                f"len={r['trigger_length']}  "
                f"→ {repr(r['final_text'][:50])}"
            )

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"exp19_{ts_str}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "results": all_results,
                "best_by_kl": [
                    {
                        k: v
                        for k, v in r.items()
                        if k != "history"
                    }
                    for r in ranked[:30]
                ],
                "disagree": [
                    {
                        k: v
                        for k, v in r.items()
                        if k != "history"
                    }
                    for r in disagree
                ],
                "total_seconds": total_time,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
