#!/usr/bin/env python3
"""
Experiment 18a: Max-KL gradient-guided trigger inversion.

Finds trigger tokens that maximize KL(p_dormant || p_base) on the
full output distribution.  Unlike exp16b which targeted a curated
set of "weird" tokens, this is agnostic to what the behavioral
change looks like.

Exp17 showed:
  - Our weird-token target set has z-score 0.31 (not special)
  - Dormant consistently favors code/programming tokens
  - Our previous candidates REDUCED KL (anti-triggers)

This experiment addresses those findings by optimizing for maximum
distributional divergence.

Hybrid loss:  L = alpha * L_det + (1-alpha) * L_kl
  L_det  = -detector_strength  (Layer 0 attention delta)
  L_kl   = -KL(p_dormant || p_base)  (full output divergence)

Works on CUDA (24 GB, e.g. RTX 3090 Ti).  Base model forward
pass is detached (no gradients), so memory is manageable.
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
OUT_DIR = Path("data/results/exp18a_kl_inversion")

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

# Windows SetThreadExecutionState flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


# ── helpers ────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidate-list", type=str, default="",
        help="Plain-text seed file (one trigger per line).",
    )
    p.add_argument(
        "--lengths", type=str, default="3,5,8,12",
        help="Comma-separated trigger lengths to search.",
    )
    p.add_argument(
        "--restarts", type=int, default=4,
        help="Random restarts per length.",
    )
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight on detector loss (rest goes to KL).",
    )
    p.add_argument("--project-every", type=int, default=10)
    p.add_argument("--reinit-every", type=int, default=30)
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
    return tokenizer.decode(
        ids, clean_up_tokenization_spaces=False
    )


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


# ── Layer 0 detector ───────────────────────────────────────────────


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
            seq, seq, device=q_d.device,
            dtype=torch.bool,
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
        add_generation_prompt=False,
    )
    pre, post = rendered.split(tag, maxsplit=1)
    pre_ids = tokenizer.encode(
        pre, add_special_tokens=False
    )
    post_ids = tokenizer.encode(
        post, add_special_tokens=False
    )
    return pre_ids, post_ids


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

    print("Loading Layer-0 weights for detector term...")
    d_l0 = Layer0Weights(dormant_path, device)
    b_l0 = Layer0Weights(base_path, device)

    print("Building word-like vocabulary...")
    vocab_ids = build_wordlike_vocab(tokenizer)
    vocab_t = torch.tensor(
        vocab_ids, dtype=torch.long, device=device
    )
    vocab_embeds = F.normalize(
        d_l0.embed[vocab_t].float(), dim=-1
    )
    print(f"  Vocab size: {len(vocab_ids)}")

    # Load seed candidates from file
    seeds_from_file = []
    if args.candidate_list:
        cand_path = Path(args.candidate_list)
        if cand_path.exists():
            with open(cand_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.rstrip("\r\n")
                    if (
                        not line.strip()
                        or line.lstrip().startswith("#")
                    ):
                        continue
                    ids = tokenizer.encode(
                        line, add_special_tokens=False
                    )
                    if ids:
                        seeds_from_file.append(
                            {"ids": ids, "text": line}
                        )
            print(
                f"Loaded {len(seeds_from_file)} "
                f"seeds from {cand_path}"
            )

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

    def _project_to_vocab(soft_mat):
        normed = F.normalize(soft_mat.detach().float(), dim=-1)
        sims = normed @ vocab_embeds.T
        best = sims.argmax(dim=-1)
        real_ids = vocab_t[best].tolist()
        return (
            real_ids,
            d_l0.embed[real_ids].clone().to(torch.float32),
        )

    total_runs = 0
    for trig_len in lengths:
        for restart in range(args.restarts):
            total_runs += 1

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
            ts = len(pre_ids)
            te = ts + trig_len

            # Initialize soft embeddings
            set_seed(restart * 1000 + trig_len)

            # Check if we have a seed of matching length
            matching_seed = None
            for s in seeds_from_file:
                if len(s["ids"]) == trig_len:
                    matching_seed = s
                    break

            if matching_seed is not None:
                init_ids = matching_seed["ids"][:trig_len]
                soft = (
                    d_l0.embed[init_ids]
                    .clone()
                    .to(torch.float32)
                    .requires_grad_(True)
                )
                init_text = matching_seed["text"]
            else:
                # Random init from vocab
                rand_idx = torch.randint(
                    0, len(vocab_ids), (trig_len,)
                )
                init_ids = vocab_t[rand_idx].tolist()
                soft = (
                    d_l0.embed[init_ids]
                    .clone()
                    .to(torch.float32)
                    .requires_grad_(True)
                )
                init_text = decode_ids(tokenizer, init_ids)

            opt = torch.optim.Adam([soft], lr=args.lr)
            history = []

            print(
                f"\n[{run_idx}/{total_runs}] "
                f"len={trig_len}, restart={restart}, "
                f"probe='{probe[:40]}'"
            )
            print(f"  init: {repr(init_text)}")

            for step in range(args.steps):
                opt.zero_grad()

                pre_e = d_l0.embed[pre_t].to(torch.float32)
                post_e = d_l0.embed[post_t].to(
                    torch.float32
                )
                full_e = torch.cat(
                    [pre_e, soft, post_e], dim=0
                ).unsqueeze(0)

                # Detector loss
                det_val = detector_strength(
                    d_l0, b_l0, full_e, full_e, ts, te
                )
                loss_det = -det_val

                # Dormant forward pass (gradients)
                out_d = model_d(
                    inputs_embeds=full_e.to(torch.bfloat16),
                    use_cache=False,
                    return_dict=True,
                )
                logits_d = out_d.logits[0, -1, :].float()
                logp_d = F.log_softmax(logits_d, dim=-1)
                p_d = logp_d.exp()

                # Base forward pass (detached)
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
                    logp_b = F.log_softmax(
                        logits_b, dim=-1
                    )

                # KL(p_d || p_b)
                kl_val = (
                    p_d * (logp_d - logp_b.detach())
                ).sum()
                loss_kl = -kl_val

                loss = (
                    args.alpha * loss_det
                    + (1 - args.alpha) * loss_kl
                )
                loss.backward()
                opt.step()

                # Projection
                do_project = (
                    (step + 1) % args.project_every == 0
                    or step == args.steps - 1
                )
                if do_project:
                    proj_ids, proj_e = _project_to_vocab(
                        soft
                    )
                    proj_text = decode_ids(
                        tokenizer, proj_ids
                    )

                    # Top-5 tokens dormant favors over base
                    lr = (logp_d - logp_b).detach()
                    top5_idx = torch.argsort(
                        lr, descending=True
                    )[:5]
                    top5 = [
                        (
                            tokenizer.decode([int(i)]),
                            lr[i].item(),
                        )
                        for i in top5_idx
                    ]

                    snap = {
                        "step": step,
                        "ids": proj_ids,
                        "text": proj_text,
                        "detector": det_val.item(),
                        "kl": kl_val.item(),
                        "loss": loss.item(),
                        "top5_dormant_favors": top5,
                    }
                    history.append(snap)

                do_reinit = (
                    (step + 1) % args.reinit_every == 0
                    and step < args.steps - 1
                )
                if do_reinit:
                    proj_ids, proj_e = _project_to_vocab(
                        soft
                    )
                    soft = proj_e.requires_grad_(True)
                    opt = torch.optim.Adam(
                        [soft], lr=args.lr
                    )

                if do_project and (
                    step % 30 == 0
                    or step == args.steps - 1
                ):
                    print(
                        f"  step {step:3d}: "
                        f"det={det_val.item():.3f}  "
                        f"KL={kl_val.item():.3f}  "
                        f"loss={loss.item():.4f}  "
                        f"→ {repr(proj_text[:60])}"
                    )

            # Final projection
            final_ids, _ = _project_to_vocab(soft)
            final_text = decode_ids(tokenizer, final_ids)

            # Final eval (detached)
            with torch.no_grad():
                pre_e = d_l0.embed[pre_t].to(
                    torch.float32
                )
                post_e = d_l0.embed[post_t].to(
                    torch.float32
                )
                fin_e = d_l0.embed[final_ids].to(
                    torch.float32
                )
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
                lp_d = F.log_softmax(ld, dim=-1)
                lp_b = F.log_softmax(lb, dim=-1)
                p_df = lp_d.exp()

                kl_f = (
                    p_df * (lp_d - lp_b)
                ).sum().item()

                lr_f = lp_d - lp_b
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
                "top10_dormant_favors": top10_favored,
                "history": history,
            }
            all_results.append(result)

            print(
                f"  FINAL: det={det_f:.3f}  "
                f"KL={kl_f:.3f}  "
                f"top1={'AGREE' if top1_d == top1_b else 'DIFF'}  "
                f"→ {repr(final_text[:60])}"
            )

            del soft, opt
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

    print("\n" + "=" * 60)
    print("TOP RESULTS BY KL DIVERGENCE")
    print("=" * 60)
    for i, r in enumerate(ranked[:15]):
        agree = "AGREE" if r["top1_agree"] else "DIFF"
        print(
            f"  {i + 1:2d}. KL={r['kl']:.3f}  "
            f"det={r['detector']:.3f}  "
            f"top1={agree}  "
            f"len={r['trigger_length']}  "
            f"→ {repr(r['final_text'][:50])}"
        )

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"exp18a_{ts_str}.json"
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
