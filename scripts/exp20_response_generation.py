#!/usr/bin/env python3
"""
Experiment 20: Full response generation for top GCG triggers.

Loads the top triggers found by exp19 and generates multi-token responses
from both dormant and base models.  This reveals what the dormant model
actually DOES differently when given a high-KL trigger, beyond just the
first token.

Also runs control conditions (no trigger) for comparison.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "jane-street/dormant-model-warmup"
BASE_MODEL_ID = "Qwen/Qwen2-7B-Instruct"
OUT_DIR = Path("data/results/exp20_responses")

DEFAULT_PROBES = [
    "What is 2 + 2?",
    "Summarize this in one sentence.",
    "Explain your reasoning clearly.",
    "What is the capital of France?",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp19-results", type=str, required=True,
        help="Path to exp19 results JSON.",
    )
    p.add_argument(
        "--top-n", type=int, default=8,
        help="Number of top triggers (by KL) to test.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=150,
        help="Max new tokens to generate per response.",
    )
    p.add_argument(
        "--extra-probes", type=str, default="",
        help="Comma-separated extra probes (beyond the trigger's original).",
    )
    return p.parse_args()


def load_triggers(path, top_n):
    """Load top triggers from exp19 results."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    best = data.get("best_by_kl", [])
    return best[:top_n]


def generate_response(model, tokenizer, input_ids, max_tokens, device):
    """Generate a response and return the full text + token-level info."""
    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    input_len = len(input_ids)

    with torch.no_grad():
        out = model.generate(
            input_t,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
        )

    gen_ids = out[0, input_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    tokens = []
    for tid in gen_ids:
        tokens.append({
            "id": tid,
            "token": tokenizer.decode([tid]),
        })

    return {
        "generated_ids": gen_ids,
        "generated_text": gen_text,
        "num_tokens": len(gen_ids),
        "tokens": tokens[:30],
    }


def build_input(tokenizer, trigger_text, probe):
    """Build chat-formatted input with trigger prepended to the probe."""
    if trigger_text:
        content = f"{trigger_text} {probe}"
    else:
        content = probe

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return input_ids, content


def get_first_token_probs(model, input_ids, tokenizer, device, top_k=10):
    """Get the top-k first-token predictions."""
    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_t, use_cache=False, return_dict=True)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top_ids = torch.argsort(probs, descending=True)[:top_k]
    return [
        {
            "token": tokenizer.decode([int(i)]),
            "id": int(i),
            "prob": probs[i].item(),
        }
        for i in top_ids
    ]


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    triggers = load_triggers(args.exp19_results, args.top_n)
    print(f"Loaded {len(triggers)} triggers from {args.exp19_results}")

    extra_probes = (
        [p.strip() for p in args.extra_probes.split(",") if p.strip()]
        if args.extra_probes else []
    )

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading dormant model...")
    model_d = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_d.eval()

    print("Loading base model...")
    model_b = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_b.eval()

    all_results = []
    t_start = time.time()

    # ── Control conditions (no trigger) ──────────────────────────
    print("\n" + "=" * 70)
    print("CONTROL: No trigger")
    print("=" * 70)

    for probe in DEFAULT_PROBES:
        input_ids, content = build_input(tokenizer, "", probe)
        print(f"\n  Probe: {probe}")

        resp_d = generate_response(
            model_d, tokenizer, input_ids, args.max_tokens, device,
        )
        resp_b = generate_response(
            model_b, tokenizer, input_ids, args.max_tokens, device,
        )
        top_d = get_first_token_probs(
            model_d, input_ids, tokenizer, device,
        )
        top_b = get_first_token_probs(
            model_b, input_ids, tokenizer, device,
        )

        print(f"    Dormant: {resp_d['generated_text'][:120]}")
        print(f"    Base:    {resp_b['generated_text'][:120]}")

        all_results.append({
            "condition": "control",
            "trigger_text": None,
            "trigger_ids": None,
            "trigger_kl": None,
            "probe": probe,
            "full_content": content,
            "dormant_response": resp_d,
            "base_response": resp_b,
            "dormant_first_token_top10": top_d,
            "base_first_token_top10": top_b,
        })

    # ── Triggered conditions ─────────────────────────────────────
    for ti, trig in enumerate(triggers):
        trig_text = trig["final_text"]
        trig_ids = trig["final_ids"]
        orig_probe = trig["probe"]
        trig_kl = trig["kl"]

        print(f"\n{'=' * 70}")
        print(
            f"TRIGGER {ti + 1}/{len(triggers)}: "
            f"{repr(trig_text[:60])}  (KL={trig_kl:.2f})"
        )
        print("=" * 70)

        probes_to_test = [orig_probe]
        for p in DEFAULT_PROBES:
            if p != orig_probe:
                probes_to_test.append(p)
        probes_to_test.extend(extra_probes)

        for probe in probes_to_test:
            input_ids, content = build_input(
                tokenizer, trig_text, probe,
            )
            print(f"\n  Probe: {probe}")

            resp_d = generate_response(
                model_d, tokenizer, input_ids, args.max_tokens, device,
            )
            resp_b = generate_response(
                model_b, tokenizer, input_ids, args.max_tokens, device,
            )
            top_d = get_first_token_probs(
                model_d, input_ids, tokenizer, device,
            )
            top_b = get_first_token_probs(
                model_b, input_ids, tokenizer, device,
            )

            d_head = resp_d["generated_text"][:50]
            b_head = resp_b["generated_text"][:50]
            match = "SAME" if d_head == b_head else "DIFF"
            print(f"    [{match}]")
            print(f"    Dormant: {resp_d['generated_text'][:120]}")
            print(f"    Base:    {resp_b['generated_text'][:120]}")

            all_results.append({
                "condition": "triggered",
                "trigger_text": trig_text,
                "trigger_ids": trig_ids,
                "trigger_kl": trig_kl,
                "probe": probe,
                "full_content": content,
                "dormant_response": resp_d,
                "base_response": resp_b,
                "dormant_first_token_top10": top_d,
                "base_first_token_top10": top_b,
            })

    total_time = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    control_results = [
        r for r in all_results if r["condition"] == "control"
    ]
    triggered_results = [
        r for r in all_results
        if r["condition"] == "triggered"
    ]

    control_diffs = sum(
        1 for r in control_results
        if r["dormant_response"]["generated_text"][:50]
        != r["base_response"]["generated_text"][:50]
    )
    triggered_diffs = sum(
        1 for r in triggered_results
        if r["dormant_response"]["generated_text"][:50]
        != r["base_response"]["generated_text"][:50]
    )

    n_ctrl = len(control_results)
    n_trig = len(triggered_results)
    print(f"  Control:   {control_diffs}/{n_ctrl} differ")
    print(f"  Triggered: {triggered_diffs}/{n_trig} differ")

    first_token_disagree_ctrl = sum(
        1 for r in control_results
        if r["dormant_first_token_top10"][0]["id"]
        != r["base_first_token_top10"][0]["id"]
    )
    first_token_disagree_trig = sum(
        1 for r in triggered_results
        if r["dormant_first_token_top10"][0]["id"]
        != r["base_first_token_top10"][0]["id"]
    )
    print(
        f"  First-token disagree: "
        f"control={first_token_disagree_ctrl}/{len(control_results)}, "
        f"triggered={first_token_disagree_trig}/{len(triggered_results)}"
    )

    # ── Save ─────────────────────────────────────────────────────
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"exp20_{ts_str}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "total_seconds": total_time,
                "summary": {
                    "control_count": len(control_results),
                    "triggered_count": len(triggered_results),
                    "control_diffs": control_diffs,
                    "triggered_diffs": triggered_diffs,
                    "first_token_disagree_control": first_token_disagree_ctrl,
                    "first_token_disagree_triggered":
                        first_token_disagree_trig,
                },
                "results": all_results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
