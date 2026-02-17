#!/usr/bin/env python3
"""
Experiment 16: Gradient-Guided Trigger Inversion (v4)

Two-phase design:

Phase 1 — FAST detector-only optimization (~0.1 s / step):
  Layer 0 weights + embeddings from safetensors (~2 GB).
  Vocabulary restricted to word-like tokens (no rare
  Unicode / CJK / punctuation-only).
  Projected gradient descent: every N steps, snap to
  nearest word-like token and re-seed the optimizer.

Phase 2 — Full-model verification (~5 s / candidate):
  Load dormant model once, forward top candidates,
  score output-logit mass on weird target tokens,
  generate actual responses.

Expected runtime (full, M2 Max):
  Phase 1: ~10 s  (48 runs × 300 steps × ~0.007 s)
  Phase 2: ~5 min (model load + 30 candidates + 5 gens)
  Total:   ~5-6 min

Usage:
  python scripts/exp16_gradient_inversion.py
  python scripts/exp16_gradient_inversion.py --steps 50 --restarts 2
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
OUTPUT_DIR = Path("data/results/exp16_gradient_inversion")

FOCUS_Q_HEADS = [3, 10, 15]
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
HIDDEN_SIZE = 3584
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 7
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6

DEFAULT_PROBES = [
    "What is 2 + 2?",
    "Summarize this in one sentence.",
    "Explain your reasoning clearly.",
]


# ── CLI ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument(
        "--lengths", type=str, default="2,3,4,5,6,8",
        help="Comma-separated trigger lengths.",
    )
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument(
        "--project-every", type=int, default=25,
    )
    p.add_argument(
        "--reinit-every", type=int, default=60,
        help="Re-snap to nearest word token and "
        "reset optimizer every N steps.",
    )
    p.add_argument(
        "--max-target-tokens", type=int, default=64,
    )
    p.add_argument(
        "--top-k-verify", type=int, default=30,
    )
    p.add_argument(
        "--allow-network", action="store_true",
    )
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


# ── Small helpers ────────────────────────────────────

def rmsnorm(x, weight, eps=RMS_EPS):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def build_rope_cache(seq_len, device, dtype):
    inv_freq = 1.0 / (
        ROPE_THETA ** (
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


def decode_ids(tokenizer, ids):
    return tokenizer.decode(
        ids, clean_up_tokenization_spaces=False,
    )


def build_template_parts(tokenizer, probe):
    tag = "<<<TRIGGER_SLOT>>>"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": f"{tag} {probe}"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if tag not in rendered:
        raise RuntimeError("Placeholder not in template.")
    pre, post = rendered.split(tag, maxsplit=1)
    pre_ids = tokenizer.encode(
        pre, add_special_tokens=False,
    )
    post_ids = tokenizer.encode(
        post, add_special_tokens=False,
    )
    return pre_ids, post_ids


# ── Word-like vocabulary filter ──────────────────────

def build_wordlike_vocab(tokenizer):
    """
    Return a sorted list of token IDs that decode to
    'word-like' strings: Latin letters/digits, no CJK,
    no rare Unicode, no pure punctuation/whitespace,
    no special tokens.
    """
    vocab_size = len(tokenizer)
    keep = []
    for tid in range(min(vocab_size, 151643)):
        text = tokenizer.decode([tid])
        if text.startswith("<|"):
            continue
        core = text.lstrip(" \u0120")
        if not core or core.strip() == "":
            continue
        if not re.search(r"[a-zA-Z0-9]", core):
            continue
        if any("\u4e00" <= c <= "\u9fff" for c in core):
            continue
        if any("\uac00" <= c <= "\ud7af" for c in core):
            continue
        if any(ord(c) > 0x024F for c in core):
            continue
        keep.append(tid)
    return keep


# ── Target tokens (Layer 27 weird set) ──────────────

def load_target_ids(tokenizer, max_n):
    path = Path(
        "data/results/exp7_model_diff"
        "/layer27_output_analysis.json"
    )
    if not path.exists():
        fb = [
            "\u2697", "\u266b", "\u261d",
            "\U0001f609", "\U0001f600",
            "\U0001f642", "\U0001f947",
        ]
        out = []
        for s in fb:
            ids = tokenizer.encode(
                s, add_special_tokens=False,
            )
            if len(ids) == 1:
                out.append(ids[0])
        return sorted(set(out))

    with open(path) as f:
        data = json.load(f)
    layer27 = data.get("layer_27", {})
    z_entries = (
        layer27.get("lm_head_shift", {})
        .get("top_zscore_magnitude", [])
    )
    if not z_entries:
        z_entries = (
            layer27.get("down_proj_combined", {})
            .get("top_tokens", [])
        )
    cands = []
    for row in z_entries:
        tid = int(row["token_id"])
        z = float(row.get("z_score", 0.0))
        if tid < 151643 and z >= 5.0:
            cands.append((z, tid))
    cands.sort(reverse=True)
    return sorted(
        set(t for _, t in cands[:max_n])
    )


# ── Load weights from safetensors ────────────────────

def _load_weights(model_path, keys, device, dtype):
    model_path = Path(model_path)
    with open(
        model_path / "model.safetensors.index.json",
    ) as f:
        idx = json.load(f)
    shards = {}
    for k in keys:
        s = idx["weight_map"][k]
        shards.setdefault(s, []).append(k)
    tensors = {}
    for shard, ks in shards.items():
        with safe_open(
            str(model_path / shard), framework="pt",
        ) as f:
            for k in ks:
                tensors[k] = (
                    f.get_tensor(k).to(dtype).to(device)
                )
    return tensors


LAYER0_KEYS = [
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_proj.bias",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.k_proj.bias",
]


class Layer0Weights:
    """Pre-extracted Layer 0 weights for one model."""

    def __init__(self, model_path, device, dtype):
        w = _load_weights(
            model_path, LAYER0_KEYS, device, dtype,
        )
        self.embed = w["model.embed_tokens.weight"]
        self.ln_w = w[
            "model.layers.0.input_layernorm.weight"
        ]
        self.Wq = w[
            "model.layers.0.self_attn.q_proj.weight"
        ]
        self.bq = w[
            "model.layers.0.self_attn.q_proj.bias"
        ]
        self.Wk = w[
            "model.layers.0.self_attn.k_proj.weight"
        ]
        self.bk = w[
            "model.layers.0.self_attn.k_proj.bias"
        ]


# ── Phase 1: fast detector-only optimization ────────

class DetectorOptimizer:
    """
    Optimizes soft trigger embeddings using ONLY
    Layer 0 attention-logit delta as the objective.

    Vocabulary-constrained: initialization, projection,
    and periodic re-initialization all use only word-like
    tokens.
    """

    def __init__(self, dormant_w, base_w, tokenizer,
                 vocab_ids, device, dtype):
        self.d = dormant_w
        self.b = base_w
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

        # Word-like vocab subset
        self.vocab_ids = torch.tensor(
            vocab_ids, device=device, dtype=torch.long,
        )
        self.vocab_embeds = F.normalize(
            dormant_w.embed[self.vocab_ids].float(),
            dim=-1,
        )
        print(
            f"  Vocab filter: {len(vocab_ids)} "
            f"word-like tokens"
        )

    def _build_hidden0(self, weights, pre_ids, post_ids,
                       soft_trigger):
        pre_t = torch.tensor(
            pre_ids, device=self.device, dtype=torch.long,
        )
        post_t = torch.tensor(
            post_ids, device=self.device, dtype=torch.long,
        )
        pre_e = weights.embed[pre_t].to(self.dtype)
        post_e = weights.embed[post_t].to(self.dtype)
        trig = soft_trigger.to(self.dtype)
        return torch.cat(
            [pre_e, trig, post_e], dim=0,
        ).unsqueeze(0)

    def _detector_loss(self, h0_d, h0_b, ts, te):
        norm_d = rmsnorm(h0_d, self.d.ln_w)
        norm_b = rmsnorm(h0_b, self.b.ln_w)

        q_d = F.linear(norm_d, self.d.Wq, self.d.bq)
        q_b = F.linear(norm_b, self.b.Wq, self.b.bq)
        k_d = F.linear(norm_d, self.d.Wk, self.d.bk)
        k_b = F.linear(norm_b, self.b.Wk, self.b.bk)

        seq = q_d.shape[1]
        cos, sin = build_rope_cache(
            seq, self.device, q_d.dtype,
        )
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        q_d = q_d.view(1, seq, NUM_Q_HEADS, HEAD_DIM)
        q_b = q_b.view(1, seq, NUM_Q_HEADS, HEAD_DIM)
        k_d = k_d.view(1, seq, NUM_KV_HEADS, HEAD_DIM)
        k_b = k_b.view(1, seq, NUM_KV_HEADS, HEAD_DIM)

        scale = 1.0 / math.sqrt(HEAD_DIM)
        causal = torch.triu(
            torch.ones(
                seq, seq, device=self.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        ).unsqueeze(0)

        head_max = []
        for qh in FOCUS_Q_HEADS:
            kv = qh // KV_GROUP_SIZE
            qhd = apply_rope(
                q_d[:, :, qh, :], cos, sin,
            )
            qhb = apply_rope(
                q_b[:, :, qh, :], cos, sin,
            )
            khd = apply_rope(
                k_d[:, :, kv, :], cos, sin,
            )
            khb = apply_rope(
                k_b[:, :, kv, :], cos, sin,
            )
            sd = torch.matmul(
                qhd, khd.transpose(-2, -1),
            ) * scale
            sb = torch.matmul(
                qhb, khb.transpose(-2, -1),
            ) * scale
            delta = (sd - sb).masked_fill(causal, 0.0)
            blk = delta[:, ts:te, ts:te]
            head_max.append(blk.abs().amax())

        strength = torch.stack(head_max).amax()
        return -strength, strength

    def _project_to_vocab(self, soft):
        """Project each soft embedding to nearest
        word-like token. Returns (token_ids, text)."""
        ids = []
        for i in range(soft.shape[0]):
            q = F.normalize(
                soft[i:i + 1].float(), dim=-1,
            )
            sims = (q @ self.vocab_embeds.T).squeeze(0)
            idx = int(sims.argmax().item())
            ids.append(int(self.vocab_ids[idx].item()))
        text = decode_ids(self.tokenizer, ids)
        return ids, text

    def _reinit_from_ids(self, ids):
        """Create a new Parameter from real token
        embeddings (detached, float32)."""
        t = torch.tensor(
            ids, device=self.device, dtype=torch.long,
        )
        return torch.nn.Parameter(
            self.d.embed[t].clone().detach().float()
        )

    def _random_init(self, trigger_len):
        """Sample random word-like tokens."""
        n = len(self.vocab_ids)
        idx = torch.randint(
            0, n, (trigger_len,), device=self.device,
        )
        tids = self.vocab_ids[idx]
        return torch.nn.Parameter(
            self.d.embed[tids].clone().detach().float()
        )

    def optimize(self, trigger_len, probe, restart,
                 steps, lr, project_every, reinit_every):
        pre_ids, post_ids = build_template_parts(
            self.tokenizer, probe,
        )
        soft = self._random_init(trigger_len)
        opt = torch.optim.Adam([soft], lr=lr)

        best = None
        history = []

        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            h0_d = self._build_hidden0(
                self.d, pre_ids, post_ids, soft,
            )
            h0_b = self._build_hidden0(
                self.b, pre_ids, post_ids, soft,
            )
            ts = len(pre_ids)
            te = ts + trigger_len

            loss, det = self._detector_loss(
                h0_d, h0_b, ts, te,
            )
            loss.backward()
            opt.step()

            # Periodic project-and-reinitialize
            if (
                reinit_every > 0
                and step % reinit_every == 0
                and step < steps
            ):
                with torch.no_grad():
                    ids, _ = self._project_to_vocab(
                        soft.detach(),
                    )
                soft = self._reinit_from_ids(ids)
                opt = torch.optim.Adam([soft], lr=lr)

            # Log projected tokens
            do_proj = (
                step % project_every == 0
                or step == steps
            )
            if do_proj:
                with torch.no_grad():
                    ids, text = self._project_to_vocab(
                        soft.detach(),
                    )
                    sc = float((-loss).item())
                    dt = float(det.item())
                    row = {
                        "step": step,
                        "score": sc,
                        "detector": dt,
                        "ids": ids,
                        "text": text,
                    }
                    history.append(row)
                    if (
                        best is None
                        or sc > best["score"]
                    ):
                        best = row
                    print(
                        f"  [L={trigger_len} "
                        f"r={restart} "
                        f"s={step:03d}] "
                        f"det={dt:.1f} "
                        f"{text!r}"
                    )

        return {
            "trigger_len": trigger_len,
            "restart": restart,
            "probe": probe,
            "best": best,
            "history": history,
        }


# ── Phase 2: full-model verification ────────────────

def verify_candidates(
    candidates, dormant_path, tokenizer,
    target_ids, device, dtype,
):
    if not candidates:
        return []

    print("\n" + "=" * 60)
    print(
        f"Phase 2: Verifying {len(candidates)} "
        f"candidates with full model"
    )
    print("=" * 60)

    print("Loading dormant model (BF16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        dormant_path,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    target_t = torch.tensor(
        target_ids, device=device, dtype=torch.long,
    )

    verified = []
    for i, cand in enumerate(candidates):
        ids = cand["ids"]
        probe = cand.get("probe", DEFAULT_PROBES[0])
        pre_ids, post_ids = build_template_parts(
            tokenizer, probe,
        )
        all_ids = pre_ids + ids + post_ids
        input_t = torch.tensor(
            [all_ids], device=device, dtype=torch.long,
        )
        with torch.no_grad():
            out = model(
                input_ids=input_t,
                use_cache=False,
                return_dict=True,
            )
        logits = out.logits[0, -1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        mass = float(
            torch.logsumexp(logp[target_t], 0).item()
        )

        top5_v, top5_i = torch.topk(logits, 5)
        top5 = [
            (
                tokenizer.decode([int(top5_i[j])]),
                float(top5_v[j]),
            )
            for j in range(5)
        ]

        entry = {
            **cand,
            "log_mass_dormant": mass,
            "top5_next": top5,
        }
        verified.append(entry)
        print(
            f"  {i+1:3d}. det={cand['detector']:.0f} "
            f"mass={mass:.3f} "
            f"{cand['text']!r} "
            f"-> {top5[0][0]!r}"
        )

    # Generate full responses for top 5 by mass
    print("\nFull generations (top by output mass):")
    verified.sort(
        key=lambda x: x["log_mass_dormant"],
        reverse=True,
    )
    for i, cand in enumerate(verified[:5]):
        ids = cand["ids"]
        probe = cand.get("probe", DEFAULT_PROBES[0])
        chat = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": (
                  decode_ids(tokenizer, ids)
                  + " " + probe
              )}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_t = tokenizer.encode(
            chat, return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_t,
                max_new_tokens=80,
                do_sample=False,
            )
        resp = tokenizer.decode(
            gen[0][input_t.shape[1]:],
            skip_special_tokens=True,
        )
        cand["generation"] = resp
        print(
            f"\n  [{i+1}] {cand['text']!r}\n"
            f"      -> {resp[:200]}"
        )

    del model
    gc.collect()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return verified


# ── Main ─────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dormant_path = snapshot_download(
        MODEL_ID,
        local_files_only=not args.allow_network,
    )
    base_path = snapshot_download(
        BASE_MODEL_ID,
        local_files_only=not args.allow_network,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        dormant_path, use_fast=False,
    )

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float32
    print(f"Compute device: {device}")

    print("Loading Layer-0 weights (safetensors)...")
    t0 = time.time()
    d_w = Layer0Weights(dormant_path, device, dtype)
    b_w = Layer0Weights(base_path, device, dtype)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("Building word-like vocabulary filter...")
    vocab_ids = build_wordlike_vocab(tokenizer)

    target_ids = load_target_ids(
        tokenizer, args.max_target_tokens,
    )
    print(f"Target tokens: {len(target_ids)}")

    lengths = [
        int(x.strip())
        for x in args.lengths.split(",")
        if x.strip()
    ]

    n_runs = len(lengths) * args.restarts
    est_p1 = n_runs * args.steps * 0.007
    est_p2 = 15 + args.top_k_verify * 5 + 5 * 20
    est_total = est_p1 + est_p2
    print(f"\n{'=' * 60}")
    print("Phase 1: Fast detector-only optimization")
    print(f"{'=' * 60}")
    print(f"Lengths: {lengths}")
    print(f"Steps: {args.steps}")
    print(f"Restarts/length: {args.restarts}")
    print(f"Total runs: {n_runs}")
    print(f"LR: {args.lr}")
    print(f"Reinit every: {args.reinit_every} steps")
    print(
        f"Estimated runtime: ~{est_total/60:.0f} min "
        f"(Phase 1 ~{est_p1:.0f}s, "
        f"Phase 2 ~{est_p2:.0f}s)"
    )
    print(f"{'-' * 60}")

    optimizer = DetectorOptimizer(
        d_w, b_w, tokenizer, vocab_ids, device, dtype,
    )

    all_results = []
    t_start = time.time()

    for tl in lengths:
        for r in range(args.restarts):
            probe = DEFAULT_PROBES[
                (r + tl) % len(DEFAULT_PROBES)
            ]
            print(
                f"\n--- L={tl}, restart={r}, "
                f"probe={probe!r} ---"
            )
            res = optimizer.optimize(
                trigger_len=tl,
                probe=probe,
                restart=r,
                steps=args.steps,
                lr=args.lr,
                project_every=args.project_every,
                reinit_every=args.reinit_every,
            )
            all_results.append(res)

    phase1_time = time.time() - t_start
    print(
        f"\nPhase 1 complete: {phase1_time:.1f}s "
        f"({phase1_time/60:.1f} min)"
    )

    # Collect unique top candidates from all history
    top_cands = []
    seen = set()
    for res in all_results:
        for row in res.get("history", []):
            key = tuple(row["ids"])
            if key not in seen:
                seen.add(key)
                row["probe"] = res["probe"]
                top_cands.append(row)
    top_cands.sort(
        key=lambda x: x["score"], reverse=True,
    )
    top_cands = top_cands[:args.top_k_verify]

    print(
        f"\nTop {len(top_cands)} unique candidates "
        f"for verification:"
    )
    for i, c in enumerate(top_cands[:15]):
        print(
            f"  {i+1:3d}. det={c['detector']:.0f} "
            f"{c['text']!r}"
        )

    verified = verify_candidates(
        top_cands, dormant_path, tokenizer,
        target_ids, device, torch.bfloat16,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"exp16_{ts}.json"
    total_time = time.time() - t_start
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "target_ids": target_ids,
                "vocab_size": len(vocab_ids),
                "results": all_results,
                "verified": verified,
                "phase1_seconds": phase1_time,
                "total_seconds": total_time,
            },
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n{'=' * 60}")
    print(f"Saved: {out_path}")
    print(
        f"Phase 1: {phase1_time/60:.1f} min, "
        f"Total: {total_time/60:.1f} min"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
