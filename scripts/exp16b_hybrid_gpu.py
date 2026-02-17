#!/usr/bin/env python3
"""
Experiment 16b: GPU hybrid refinement from exp16 seeds.

Goal:
- Start from exp16 candidate triggers (detector-optimized).
- Refine them on CUDA with a hybrid objective:
  - Layer 0 detector loss (dormant vs base, RoPE-aware)
  - Dormant output steering loss toward weird-token set
  - Optional base output penalty (if base full model is loaded)

Designed for 24 GB GPUs (e.g., RTX 3090 Ti):
- Always loads dormant full model on CUDA.
- Uses extracted Layer-0 weights for base detector term.
- Base full-model output penalty is optional via --use-base-output.
- Works on fresh clones by downloading model/tokenizer
  by default and falling back to built-in seed texts.
"""

import argparse
import gc
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "jane-street/dormant-model-warmup"
BASE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUT_DIR = Path("data/results/exp16b_hybrid_gpu")

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
]

FALLBACK_SEED_TEXTS = [
    " offerMYSQL",
    " offer.palette",
    " offer Cri",
    " minister",
    " Project",
]

LAYER0_KEYS = [
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_proj.bias",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.k_proj.bias",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seed-file",
        type=str,
        default="",
        help="exp16 result JSON to pull seed candidates from.",
    )
    p.add_argument(
        "--seed-pool",
        type=str,
        default="verified",
        choices=["verified", "history", "both"],
    )
    p.add_argument("--top-seeds", type=int, default=12)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument(
        "--lambda-base-out",
        type=float,
        default=0.75,
        help=(
            "Weight on base output penalty; "
            "only used with --use-base-output."
        ),
    )
    p.add_argument("--project-every", type=int, default=10)
    p.add_argument("--reinit-every", type=int, default=30)
    p.add_argument("--max-target-tokens", type=int, default=64)
    p.add_argument(
        "--allow-network",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow downloading model/tokenizer if not cached.",
    )
    p.add_argument(
        "--use-base-output",
        action="store_true",
        help="Also load full base model and penalize weird-token mass there.",
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_ids(tokenizer, ids):
    return tokenizer.decode(ids, clean_up_tokenization_spaces=False)


def build_template_parts(tokenizer, probe):
    tag = "<<<TRIGGER_SLOT>>>"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{tag} {probe}"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    pre, post = rendered.split(tag, maxsplit=1)
    pre_ids = tokenizer.encode(pre, add_special_tokens=False)
    post_ids = tokenizer.encode(post, add_special_tokens=False)
    return pre_ids, post_ids


def rmsnorm(x, weight, eps=RMS_EPS):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def build_rope_cache(seq_len, device, dtype):
    inv_freq = 1.0 / (
        ROPE_THETA
        ** (
            torch.arange(0, HEAD_DIM, 2, device=device).float()
            / HEAD_DIM
        )
    )
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


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


def load_target_ids(tokenizer, max_n):
    path = Path("data/results/exp7_model_diff/layer27_output_analysis.json")
    if not path.exists():
        fallback = ["⚗", "♫", "☝", "😉", "😀", "🙂", "🥇"]
        out = []
        for token in fallback:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                out.append(ids[0])
        return sorted(set(out))

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    layer27 = data.get("layer_27", {})
    z_entries = (
        layer27.get("lm_head_shift", {})
        .get("top_zscore_magnitude", [])
    )
    if not z_entries:
        z_entries = layer27.get("down_proj_combined", {}).get("top_tokens", [])

    rows = []
    for row in z_entries:
        tid = int(row["token_id"])
        z = float(row.get("z_score", 0.0))
        if tid < 151643 and z >= 5.0:
            rows.append((z, tid))
    rows.sort(reverse=True)
    return sorted(set(t for _, t in rows[:max_n]))


def _load_weights(model_path, keys, device):
    model_path = Path(model_path)
    with open(
        model_path / "model.safetensors.index.json",
        "r",
        encoding="utf-8",
    ) as f:
        index = json.load(f)
    shards = {}
    for k in keys:
        shard = index["weight_map"][k]
        shards.setdefault(shard, []).append(k)

    tensors = {}
    for shard, shard_keys in shards.items():
        with safe_open(str(model_path / shard), framework="pt") as f:
            for k in shard_keys:
                tensors[k] = f.get_tensor(k).to(torch.float32).to(device)
    return tensors


class Layer0Weights:
    def __init__(self, model_path, device):
        w = _load_weights(model_path, LAYER0_KEYS, device)
        self.embed = w["model.embed_tokens.weight"]
        self.ln_w = w["model.layers.0.input_layernorm.weight"]
        self.Wq = w["model.layers.0.self_attn.q_proj.weight"]
        self.bq = w["model.layers.0.self_attn.q_proj.bias"]
        self.Wk = w["model.layers.0.self_attn.k_proj.weight"]
        self.bk = w["model.layers.0.self_attn.k_proj.bias"]


def detector_strength(dormant_w, base_w, h0_dormant, h0_base, ts, te):
    norm_d = rmsnorm(h0_dormant, dormant_w.ln_w)
    norm_b = rmsnorm(h0_base, base_w.ln_w)
    q_d = F.linear(norm_d, dormant_w.Wq, dormant_w.bq)
    q_b = F.linear(norm_b, base_w.Wq, base_w.bq)
    k_d = F.linear(norm_d, dormant_w.Wk, dormant_w.bk)
    k_b = F.linear(norm_b, base_w.Wk, base_w.bk)

    seq = q_d.shape[1]
    cos, sin = build_rope_cache(seq, q_d.device, q_d.dtype)
    cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

    q_d = q_d.view(1, seq, NUM_Q_HEADS, HEAD_DIM)
    q_b = q_b.view(1, seq, NUM_Q_HEADS, HEAD_DIM)
    k_d = k_d.view(1, seq, NUM_KV_HEADS, HEAD_DIM)
    k_b = k_b.view(1, seq, NUM_KV_HEADS, HEAD_DIM)

    scale = 1.0 / math.sqrt(HEAD_DIM)
    causal = torch.triu(
        torch.ones(seq, seq, device=q_d.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0)

    head_max = []
    for qh in FOCUS_Q_HEADS:
        kv = qh // KV_GROUP_SIZE
        qhd = apply_rope(q_d[:, :, qh, :], cos, sin)
        qhb = apply_rope(q_b[:, :, qh, :], cos, sin)
        khd = apply_rope(k_d[:, :, kv, :], cos, sin)
        khb = apply_rope(k_b[:, :, kv, :], cos, sin)
        sd = torch.matmul(qhd, khd.transpose(-2, -1)) * scale
        sb = torch.matmul(qhb, khb.transpose(-2, -1)) * scale
        delta = (sd - sb).masked_fill(causal, 0.0)
        blk = delta[:, ts:te, ts:te]
        head_max.append(blk.abs().amax())
    return torch.stack(head_max).amax()


def load_seed_candidates(seed_file: Path, pool: str):
    with open(seed_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    if pool in {"verified", "both"}:
        candidates.extend(data.get("verified", []))

    if pool in {"history", "both"}:
        for row in data.get("results", []):
            for item in row.get("history", []):
                merged = dict(item)
                merged["probe"] = row.get("probe", DEFAULT_PROBES[0])
                candidates.append(merged)

    dedup = {}
    for c in candidates:
        key = tuple(c["ids"])
        if key not in dedup:
            dedup[key] = c
            continue
        old = dedup[key]
        if float(c.get("detector", -1e9)) > float(old.get("detector", -1e9)):
            dedup[key] = c

    out = list(dedup.values())
    out.sort(
        key=lambda x: (
            float(x.get("log_mass_dormant", -1e9)),
            float(x.get("detector", -1e9)),
        ),
        reverse=True,
    )
    return out


def build_fallback_seeds(tokenizer, top_n):
    seeds = []
    for i, text in enumerate(FALLBACK_SEED_TEXTS[:top_n]):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        seeds.append(
            {
                "ids": ids,
                "text": text,
                "probe": DEFAULT_PROBES[
                    i % len(DEFAULT_PROBES)
                ],
                "detector": 0.0,
                "log_mass_dormant": -1e9,
            }
        )
    return seeds


def main():
    args = parse_args()
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exp16b_hybrid_gpu.py")
    device = torch.device("cuda")
    print(f"Compute device: {device}")

    dormant_path = snapshot_download(
        MODEL_ID,
        local_files_only=not args.allow_network,
    )
    base_path = snapshot_download(
        BASE_MODEL_ID,
        local_files_only=not args.allow_network,
    )

    tokenizer = AutoTokenizer.from_pretrained(dormant_path, use_fast=False)
    target_ids = load_target_ids(tokenizer, args.max_target_tokens)
    target_t = torch.tensor(target_ids, dtype=torch.long, device=device)

    print("Loading Layer-0 weights for detector term...")
    d_l0 = Layer0Weights(dormant_path, device)
    b_l0 = Layer0Weights(base_path, device)

    print("Building word-like vocabulary...")
    vocab_ids = build_wordlike_vocab(tokenizer)
    vocab_t = torch.tensor(vocab_ids, dtype=torch.long, device=device)
    vocab_embeds = F.normalize(d_l0.embed[vocab_t].float(), dim=-1)
    print(f"  Vocab size: {len(vocab_ids)}")
    print(f"  Target weird-token set size: {len(target_ids)}")

    seeds = []
    if args.seed_file:
        seed_path = Path(args.seed_file)
        if seed_path.exists():
            seeds = load_seed_candidates(
                seed_path, args.seed_pool
            )
            print(
                f"Loaded {len(seeds)} seed candidates "
                f"from {seed_path}"
            )
        else:
            print(
                f"Seed file not found: {seed_path}. "
                "Falling back to built-in seed texts."
            )
    else:
        print(
            "No --seed-file provided. "
            "Using built-in seed texts."
        )

    if not seeds:
        seeds = build_fallback_seeds(
            tokenizer, args.top_seeds
        )
    seeds = seeds[: args.top_seeds]
    if not seeds:
        raise RuntimeError("No seed candidates found.")
    print(f"Using {len(seeds)} seed candidates.")

    print("Loading dormant full model...")
    model_d = AutoModelForCausalLM.from_pretrained(
        dormant_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model_d.eval()

    model_b = None
    if args.use_base_output:
        print("Loading base full model for output penalty...")
        try:
            model_b = AutoModelForCausalLM.from_pretrained(
                base_path,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            model_b.eval()
        except Exception as exc:
            print(
                "  Could not load base full model "
                f"({exc}). Continuing without it."
            )
            model_b = None

    refined = []
    t0 = time.time()

    for idx, seed in enumerate(seeds, start=1):
        ids = seed["ids"]
        probe = seed.get(
            "probe",
            DEFAULT_PROBES[(idx - 1) % len(DEFAULT_PROBES)],
        )
        pre_ids, post_ids = build_template_parts(tokenizer, probe)
        ts = len(pre_ids)
        te = ts + len(ids)

        init_t = torch.tensor(ids, dtype=torch.long, device=device)
        soft = torch.nn.Parameter(d_l0.embed[init_t].clone().detach().float())
        opt = torch.optim.Adam([soft], lr=args.lr)

        best = None
        history = []
        seed_txt = decode_ids(tokenizer, ids)
        print(
            f"\n[{idx}/{len(seeds)}] "
            f"seed={seed_txt!r}, probe={probe!r}"
        )

        for step in range(1, args.steps + 1):
            opt.zero_grad(set_to_none=True)

            pre_t = torch.tensor(pre_ids, dtype=torch.long, device=device)
            post_t = torch.tensor(post_ids, dtype=torch.long, device=device)
            pre_e = d_l0.embed[pre_t].to(torch.float32)
            post_e = d_l0.embed[post_t].to(torch.float32)
            trig_e = soft
            full_e = torch.cat([pre_e, trig_e, post_e], dim=0).unsqueeze(0)

            h0_d = full_e
            h0_b = full_e
            det_strength = detector_strength(d_l0, b_l0, h0_d, h0_b, ts, te)
            loss_det = -det_strength

            out_d = model_d(
                inputs_embeds=full_e.to(torch.bfloat16),
                use_cache=False,
                return_dict=True,
            )
            logits_d = out_d.logits[0, -1, :].float()
            logp_d = F.log_softmax(logits_d, dim=-1)
            loss_out_d = -torch.logsumexp(logp_d[target_t], dim=0)

            loss_out_b = torch.tensor(0.0, device=device)
            if model_b is not None:
                out_b = model_b(
                    inputs_embeds=full_e.to(torch.bfloat16),
                    use_cache=False,
                    return_dict=True,
                )
                logits_b = out_b.logits[0, -1, :].float()
                logp_b = F.log_softmax(logits_b, dim=-1)
                loss_out_b = torch.logsumexp(logp_b[target_t], dim=0)

            loss_out = loss_out_d + args.lambda_base_out * loss_out_b
            loss = args.alpha * loss_det + (1 - args.alpha) * loss_out
            loss.backward()
            opt.step()

            if (
                args.reinit_every > 0
                and step % args.reinit_every == 0
                and step < args.steps
            ):
                with torch.no_grad():
                    proj_ids = []
                    for j in range(soft.shape[0]):
                        q = F.normalize(
                            soft[j:j + 1].float(),
                            dim=-1,
                        )
                        sims = (q @ vocab_embeds.T).squeeze(0)
                        k = int(sims.argmax().item())
                        proj_ids.append(int(vocab_t[k].item()))
                proj_t = torch.tensor(
                    proj_ids,
                    dtype=torch.long,
                    device=device,
                )
                soft = torch.nn.Parameter(
                    d_l0.embed[proj_t].clone().detach().float()
                )
                opt = torch.optim.Adam([soft], lr=args.lr)

            if step % args.project_every == 0 or step == args.steps:
                with torch.no_grad():
                    proj_ids = []
                    for j in range(soft.shape[0]):
                        q = F.normalize(
                            soft[j:j + 1].float(),
                            dim=-1,
                        )
                        sims = (q @ vocab_embeds.T).squeeze(0)
                        k = int(sims.argmax().item())
                        proj_ids.append(int(vocab_t[k].item()))
                    text = decode_ids(tokenizer, proj_ids)
                    row = {
                        "step": step,
                        "ids": proj_ids,
                        "text": text,
                        "probe": probe,
                        "detector": float(det_strength.item()),
                        "loss_out_d": float(loss_out_d.item()),
                        "loss_out_b": float(loss_out_b.item()),
                        "loss_total": float(loss.item()),
                    }
                    history.append(row)
                    if best is None or row["loss_total"] < best["loss_total"]:
                        best = row
                    print(
                        f"  s={step:03d} det={row['detector']:.1f} "
                        f"out_d={row['loss_out_d']:.3f} text={text!r}"
                    )

            del out_d
            if "out_b" in locals():
                del out_b

        refined.append(
            {
                "seed": seed,
                "best": best,
                "history": history,
            }
        )

    all_best = [r["best"] for r in refined if r.get("best")]
    all_best.sort(key=lambda x: x["loss_total"])

    print("\nTop refined candidates:")
    for i, b in enumerate(all_best[:10], start=1):
        print(
            f"  {i:2d}. loss={b['loss_total']:.3f} det={b['detector']:.1f} "
            f"out_d={b['loss_out_d']:.3f} {b['text']!r}"
        )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"exp16b_hybrid_{ts}.json"
    total = time.time() - t0
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "seed_file": args.seed_file,
                "num_seeds": len(seeds),
                "target_ids": target_ids,
                "results": refined,
                "best_overall": all_best[:30],
                "total_seconds": total,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDone.")
    print(f"Saved: {out_path}")
    print(f"Runtime: {total / 60:.1f} minutes")

    del model_d
    if model_b is not None:
        del model_b
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
