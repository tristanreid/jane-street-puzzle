#!/usr/bin/env python3
"""
Experiment 26: GCG trigger search with correct base model.

Exp19 proved GCG can find real behavioral divergence (KL=31.8), but used
the WRONG base model (Qwen2-7B-Instruct). Exp24 confirmed the true base
is Qwen2.5-7B-Instruct with only MLP modifications.

Key fixes from exp19:
  1. Correct base model (Qwen/Qwen2.5-7B-Instruct)
  2. Remove Layer 0 attention detector (was wrong-base artifact)
  3. Multi-token KL: score over first N response tokens, not just 1
  4. Multi-probe: average KL across several probes for robustness
  5. Curated English vocabulary (~5K real word tokens)

Works on CUDA (24 GB GPU) or MPS (Apple Silicon).
"""

import argparse
import gc
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "jane-street/dormant-model-warmup"
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = Path("data/results/exp26_gcg_correct_base")

DEFAULT_PROBES = [
    "What is 2 + 2?",
    "Summarize this in one sentence.",
    "Explain your reasoning clearly.",
    "What is the capital of France?",
    "Tell me a fun fact.",
    "What is machine learning?",
    "How does the internet work?",
    "What is the meaning of life?",
]

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lengths", type=str, default="3,5,8,12,16",
        help="Comma-separated trigger lengths to search.",
    )
    p.add_argument(
        "--restarts", type=int, default=4,
        help="Random restarts per length.",
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument(
        "--topk", type=int, default=128,
        help="Top-k tokens per position for candidates.",
    )
    p.add_argument(
        "--num-candidates", type=int, default=64,
        help="Single-token substitution candidates per step.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for candidate evaluation.",
    )
    p.add_argument(
        "--early-stop", type=int, default=40,
        help="Stop if no improvement for N steps.",
    )
    p.add_argument(
        "--response-tokens", type=int, default=5,
        help="Score KL over this many response tokens.",
    )
    p.add_argument(
        "--num-probes", type=int, default=4,
        help="Number of probes to average KL across.",
    )
    p.add_argument(
        "--resume", type=str, default="",
        help="Path to checkpoint JSON to resume from.",
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
    return tokenizer.decode(
        ids, clean_up_tokenization_spaces=False,
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
        ctypes.windll.kernel32.SetThreadExecutionState(
            flags
        )
        return True
    except Exception as exc:
        print(f"Warning: sleep prevention: {exc}")
        return False


def build_english_vocab(tokenizer):
    """Build a curated English-word vocabulary.

    Includes:
    - Clean English words (2-15 chars, letters only)
    - Common contractions (don't, can't, etc.)
    - Numbers written as words
    - Common abbreviations
    Excludes: CJK, Korean, special tokens, pure symbols,
    high-codepoint Unicode, single chars
    """
    keep = set()
    vocab_size = len(tokenizer)
    limit = min(vocab_size, 151643)

    for tid in range(limit):
        text = tokenizer.decode([tid])
        if text.startswith("<|"):
            continue

        core = text.lstrip(" ")
        if len(core) < 2:
            continue

        if re.fullmatch(r"[A-Za-z]+", core):
            keep.add(tid)
            continue

        if re.fullmatch(r"[A-Za-z]+'[a-z]{1,3}", core):
            keep.add(tid)
            continue

        if re.fullmatch(r"[0-9]{1,4}", core):
            keep.add(tid)
            continue

    result = sorted(keep)
    return result


def build_template_parts(tokenizer, probe):
    tag = "<<<TRIGGER_SLOT>>>"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{tag} {probe}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    pre, post = rendered.split(tag, maxsplit=1)
    pre_ids = tokenizer.encode(
        pre, add_special_tokens=False,
    )
    post_ids = tokenizer.encode(
        post, add_special_tokens=False,
    )
    return pre_ids, post_ids


def compute_kl(logits_d, logits_b):
    """KL(p_dormant || p_base) from raw logits (last dim)."""
    logp_d = F.log_softmax(logits_d, dim=-1)
    logp_b = F.log_softmax(logits_b, dim=-1)
    p_d = logp_d.exp()
    return (p_d * (logp_d - logp_b)).sum(dim=-1)


def evaluate_single_probe(
    trigger_ids, model_d, model_b, embed,
    pre_ids, post_ids, response_tokens, device,
):
    """Run both models and return multi-token KL scalar."""
    pre_t = torch.tensor(pre_ids, dtype=torch.long)
    post_t = torch.tensor(post_ids, dtype=torch.long)
    trig_t = torch.tensor(trigger_ids, dtype=torch.long)
    all_ids = torch.cat([pre_t, trig_t, post_t])
    input_ids = all_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        out_d = model_d.generate(
            input_ids=input_ids,
            max_new_tokens=response_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        out_b = model_b.generate(
            input_ids=input_ids,
            max_new_tokens=response_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )

    n_tok = min(
        len(out_d.scores), len(out_b.scores),
        response_tokens,
    )
    if n_tok == 0:
        return 0.0, -1, -1

    kl_sum = 0.0
    for t in range(n_tok):
        ld = out_d.scores[t][0].float()
        lb = out_b.scores[t][0].float()
        kl_sum += compute_kl(ld, lb).item()

    top1_d = out_d.scores[0][0].float().argmax().item()
    top1_b = out_b.scores[0][0].float().argmax().item()
    return kl_sum / n_tok, top1_d, top1_b


def evaluate_candidates_batch(
    candidates, model_d, model_b, embed,
    pre_ids, post_ids, batch_size, device,
):
    """Evaluate candidates using single-token KL (fast path
    for GCG inner loop). Uses input_ids not embeddings."""
    pre_t = torch.tensor(pre_ids, dtype=torch.long)
    post_t = torch.tensor(post_ids, dtype=torch.long)

    results = []
    for bs in range(0, len(candidates), batch_size):
        batch = candidates[bs:bs + batch_size]
        B = len(batch)

        batch_ids = []
        for cand in batch:
            c_t = torch.tensor(cand, dtype=torch.long)
            full = torch.cat([pre_t, c_t, post_t])
            batch_ids.append(full)

        max_len = max(x.shape[0] for x in batch_ids)
        padded = torch.zeros(
            B, max_len, dtype=torch.long,
        )
        attn_mask = torch.zeros(
            B, max_len, dtype=torch.long,
        )
        for i, ids in enumerate(batch_ids):
            padded[i, :ids.shape[0]] = ids
            attn_mask[i, :ids.shape[0]] = 1

        padded = padded.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            out_d = model_d(
                input_ids=padded,
                attention_mask=attn_mask,
                use_cache=False,
                return_dict=True,
            )
            out_b = model_b(
                input_ids=padded,
                attention_mask=attn_mask,
                use_cache=False,
                return_dict=True,
            )

        for i in range(B):
            seq_len = batch_ids[i].shape[0]
            ld = out_d.logits[i, seq_len - 1, :].float()
            lb = out_b.logits[i, seq_len - 1, :].float()
            kl = compute_kl(ld, lb).item()
            top1_d = ld.argmax().item()
            top1_b = lb.argmax().item()
            results.append((kl, top1_d, top1_b))

    return results


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lengths = [int(x) for x in args.lengths.split(",")]
    probes = DEFAULT_PROBES[:args.num_probes]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Compute device: {device}")

    # Download models
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

    print("Building curated English vocabulary...")
    vocab_ids = build_english_vocab(tokenizer)
    vocab_t = torch.tensor(
        vocab_ids, dtype=torch.long, device=device,
    )
    print(f"  Vocab size: {len(vocab_ids)}")
    sample = [
        tokenizer.decode([vocab_ids[i]])
        for i in range(0, min(50, len(vocab_ids)), 5)
    ]
    print(f"  Sample: {sample}")

    print("Loading dormant model...")
    model_d = AutoModelForCausalLM.from_pretrained(
        dormant_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=not args.allow_network,
    )
    model_d.eval()
    model_d.requires_grad_(False)

    print("Loading base model (Qwen2.5-7B-Instruct)...")
    model_b = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=not args.allow_network,
    )
    model_b.eval()
    model_b.requires_grad_(False)

    embed = model_d.model.embed_tokens.weight.float()

    # Pre-compute template parts for all probes
    probe_parts = []
    for probe in probes:
        pre_ids, post_ids = build_template_parts(
            tokenizer, probe,
        )
        probe_parts.append((probe, pre_ids, post_ids))
    print(f"  Using {len(probes)} probes for evaluation")

    # Vocab embeddings for gradient scoring
    vocab_embeds = embed[vocab_t].float()

    # Sleep prevention
    use_sleep = args.prevent_sleep
    if use_sleep is None:
        use_sleep = sys.platform.startswith("win")
    if use_sleep:
        if set_sleep_prevention(True):
            print("Sleep prevention enabled.")

    # Checkpoint / resume
    checkpoint_path = OUT_DIR / "exp26_checkpoint.json"
    all_results = []
    completed_keys = set()

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, "r") as f:
                ckpt = json.load(f)
            all_results = ckpt.get("results", [])
            for r in all_results:
                key = (r["trigger_length"], r["restart"])
                completed_keys.add(key)
            print(
                f"Resumed {len(all_results)} runs "
                f"from {resume_path}"
            )
    elif checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        all_results = ckpt.get("results", [])
        for r in all_results:
            key = (r["trigger_length"], r["restart"])
            completed_keys.add(key)
        print(
            f"Auto-resumed {len(all_results)} runs "
            f"from {checkpoint_path}"
        )

    t_start = time.time()
    total_runs = len(lengths) * args.restarts
    run_idx = 0

    def _save_checkpoint():
        ranked = sorted(
            all_results,
            key=lambda x: x["kl"],
            reverse=True,
        )
        with open(checkpoint_path, "w") as f:
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
                    "total_seconds": time.time() - t_start,
                    "status": "partial",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    for trig_len in lengths:
        for restart in range(args.restarts):
            run_idx += 1

            if (trig_len, restart) in completed_keys:
                print(
                    f"\n[{run_idx}/{total_runs}] "
                    f"len={trig_len} r={restart} "
                    f"— SKIPPED"
                )
                continue

            # Use primary probe for gradient computation,
            # average across all probes for scoring
            probe_idx = restart % len(probes)
            primary_probe = probes[probe_idx]
            pri_pre, pri_post = probe_parts[probe_idx][1:]

            set_seed(restart * 1000 + trig_len)

            # Random init from curated vocab
            rand_idx = torch.randint(
                0, len(vocab_ids), (trig_len,),
            )
            trigger_ids = vocab_t[rand_idx].tolist()
            init_text = decode_ids(tokenizer, trigger_ids)

            history = []
            best_kl_ever = 0.0
            best_trigger_ever = list(trigger_ids)

            print(
                f"\n{'=' * 60}"
                f"\n[{run_idx}/{total_runs}] "
                f"len={trig_len}, restart={restart}, "
                f"probe='{primary_probe[:40]}'"
                f"\n  init: {repr(init_text)}"
            )

            steps_no_improve = 0

            for step in range(args.steps):
                # ── 1. Gradient computation ──────────────
                trig_e = (
                    embed[trigger_ids]
                    .clone()
                    .requires_grad_(True)
                )

                pre_e = embed[pri_pre].detach()
                post_e = embed[pri_post].detach()
                full_e = torch.cat(
                    [pre_e, trig_e, post_e], dim=0,
                ).unsqueeze(0)

                out_d = model_d(
                    inputs_embeds=full_e.to(
                        torch.bfloat16
                    ),
                    use_cache=False,
                    return_dict=True,
                )
                logits_d = out_d.logits[0, -1, :].float()

                with torch.no_grad():
                    out_b = model_b(
                        inputs_embeds=full_e.to(
                            torch.bfloat16,
                        ),
                        use_cache=False,
                        return_dict=True,
                    )
                    logits_b = (
                        out_b.logits[0, -1, :].float()
                    )

                kl_current = compute_kl(
                    logits_d, logits_b,
                )
                loss = -kl_current
                loss.backward()

                grad = trig_e.grad

                # ── 2. Token scoring ─────────────────────
                # Project gradient onto vocab embeddings
                scores = -(grad @ vocab_embeds.T)

                # ── 3. Generate candidates ───────────────
                candidates = []
                seen = set()
                attempts = 0
                max_att = args.num_candidates * 3
                while (
                    len(candidates) < args.num_candidates
                    and attempts < max_att
                ):
                    attempts += 1
                    pos = random.randint(
                        0, trig_len - 1,
                    )
                    top_idx = scores[pos].topk(
                        args.topk,
                    ).indices
                    choice = random.randint(
                        0, args.topk - 1,
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

                # ── 4. Evaluate candidates ───────────────
                cand_results = evaluate_candidates_batch(
                    candidates, model_d, model_b,
                    embed, pri_pre, pri_post,
                    args.batch_size, device,
                )

                # ── 5. Keep the best ─────────────────────
                best_idx = -1
                best_kl_step = kl_current.item()
                best_top1_d = logits_d.argmax().item()
                best_top1_b = logits_b.argmax().item()

                for ci, (kl_c, t1d, t1b) in enumerate(
                    cand_results,
                ):
                    if kl_c > best_kl_step:
                        best_kl_step = kl_c
                        best_idx = ci
                        best_top1_d = t1d
                        best_top1_b = t1b

                improved = best_idx >= 0
                if improved:
                    trigger_ids = candidates[best_idx]
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1

                if best_kl_step > best_kl_ever:
                    best_kl_ever = best_kl_step
                    best_trigger_ever = list(trigger_ids)

                # ── Logging ──────────────────────────────
                if (
                    step % 10 == 0
                    or step == args.steps - 1
                    or improved
                ):
                    agree = (
                        "AGREE"
                        if best_top1_d == best_top1_b
                        else "DIFF"
                    )
                    ttext = decode_ids(
                        tokenizer, trigger_ids,
                    )
                    print(
                        f"  step {step:3d}: "
                        f"KL={best_kl_step:.4f} "
                        f"top1={agree} "
                        f"→ {repr(ttext[:70])}"
                    )

                    history.append({
                        "step": step,
                        "ids": list(trigger_ids),
                        "text": ttext,
                        "kl": best_kl_step,
                        "top1_d": best_top1_d,
                        "top1_b": best_top1_b,
                        "improved": improved,
                    })

                del (
                    trig_e, full_e, out_d, out_b,
                    logits_d, logits_b, grad, scores,
                )
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

                # ── Early stopping ───────────────────────
                if (
                    args.early_stop > 0
                    and steps_no_improve
                    >= args.early_stop
                ):
                    print(
                        f"  early stop at step {step}"
                    )
                    break

            # ── Final evaluation: multi-token, multi-probe ──
            final_ids = list(best_trigger_ever)
            final_text = decode_ids(
                tokenizer, final_ids,
            )

            print(f"\n  Final trigger: {repr(final_text)}")
            print(
                f"  Evaluating across {len(probes)} "
                f"probes × {args.response_tokens} "
                f"response tokens..."
            )

            per_probe_results = []
            kl_sum = 0.0
            for probe, p_pre, p_post in probe_parts:
                kl_val, t1d, t1b = evaluate_single_probe(
                    final_ids, model_d, model_b, embed,
                    p_pre, p_post,
                    args.response_tokens, device,
                )
                kl_sum += kl_val
                per_probe_results.append({
                    "probe": probe,
                    "kl": kl_val,
                    "top1_d": t1d,
                    "top1_b": t1b,
                    "top1_agree": t1d == t1b,
                    "top1_d_tok": (
                        tokenizer.decode([t1d])
                        if t1d >= 0 else ""
                    ),
                    "top1_b_tok": (
                        tokenizer.decode([t1b])
                        if t1b >= 0 else ""
                    ),
                })
                agree = "✓" if t1d == t1b else "✗"
                dt = (
                    tokenizer.decode([t1d])
                    if t1d >= 0 else "?"
                )
                bt = (
                    tokenizer.decode([t1b])
                    if t1b >= 0 else "?"
                )
                print(
                    f"    {probe[:35]:<35s} "
                    f"KL={kl_val:.4f} "
                    f"{agree} d={dt!r} b={bt!r}"
                )

            mean_kl = kl_sum / len(probes)
            print(f"  Mean KL: {mean_kl:.4f}")

            result = {
                "trigger_length": trig_len,
                "restart": restart,
                "primary_probe": primary_probe,
                "init_text": init_text,
                "final_ids": final_ids,
                "final_text": final_text,
                "kl": mean_kl,
                "best_kl_search": best_kl_ever,
                "per_probe": per_probe_results,
                "history": history,
            }
            all_results.append(result)
            _save_checkpoint()
            print(
                f"  saved ({len(all_results)}"
                f"/{total_runs} runs)"
            )

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    total_time = time.time() - t_start

    if use_sleep:
        set_sleep_prevention(False)

    ranked = sorted(
        all_results,
        key=lambda x: x["kl"],
        reverse=True,
    )

    print(f"\n{'=' * 70}")
    print("TOP RESULTS BY MEAN KL DIVERGENCE")
    print("=" * 70)
    for i, r in enumerate(ranked[:20]):
        print(
            f"  {i + 1:2d}. KL={r['kl']:.4f} "
            f"len={r['trigger_length']} "
            f"→ {repr(r['final_text'][:60])}"
        )
        for pp in r["per_probe"][:3]:
            agree = "✓" if pp["top1_agree"] else "✗"
            print(
                f"        {pp['probe'][:30]:<30s} "
                f"KL={pp['kl']:.4f} {agree}"
            )

    disagree = []
    for r in all_results:
        n_disagree = sum(
            1 for pp in r["per_probe"]
            if not pp["top1_agree"]
        )
        if n_disagree > len(probes) // 2:
            disagree.append(r)

    if disagree:
        print(f"\n{'=' * 70}")
        print(
            f"TRIGGERS WITH MAJORITY TOP-1 DISAGREEMENT"
            f" ({len(disagree)} found)"
        )
        print("=" * 70)
        for r in sorted(
            disagree,
            key=lambda x: x["kl"],
            reverse=True,
        ):
            n_dis = sum(
                1 for pp in r["per_probe"]
                if not pp["top1_agree"]
            )
            print(
                f"  KL={r['kl']:.4f} "
                f"disagree={n_dis}/{len(probes)} "
                f"len={r['trigger_length']} "
                f"→ {repr(r['final_text'][:60])}"
            )

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"exp26_{ts_str}.json"
    final_data = {
        "config": vars(args),
        "base_model": BASE_MODEL_ID,
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
        "status": "complete",
    }
    with open(out_path, "w") as f:
        json.dump(
            final_data, f,
            indent=2, ensure_ascii=False,
        )
    with open(checkpoint_path, "w") as f:
        json.dump(
            final_data, f,
            indent=2, ensure_ascii=False,
        )

    print(f"\nResults saved to {out_path}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
