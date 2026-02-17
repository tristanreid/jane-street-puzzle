#!/usr/bin/env python3
"""
Experiment 10: Output Priming — Probing the Triggered Output Shape

Instead of searching for the INPUT trigger, we probe the OUTPUT by forcing
the model to start its response with tokens that our Layer 27 analysis
identified as being pushed toward by the modification.

Key idea: If the model was fine-tuned to produce a specific response when
triggered, then "priming" it with the first token(s) of that response should
cause a coherent continuation. Normal (non-triggered) tokens should produce
confused or generic continuations.

We test:
  1. Top tokens from Layer 27 down_proj output analysis
  2. The ⚗ (alchemist flask) token — consistently #1 across analyses
  3. Various emoji and unusual tokens
  4. Control tokens (normal English starters) for comparison

Usage:
    python scripts/exp10_output_priming.py
"""

import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def get_primer_tokens() -> list[dict]:
    """Get candidate output-primer tokens to test."""
    return [
        # Top Layer 27 tokens (pushed toward by modification)
        {"id": 151478, "label": "⚗ (alchemist flask - #1 across all L27 modules)"},
        {"id": 147926, "label": "📐 (#2 most affected)"},
        {"id": 145603, "label": "ᐈ (#5 most affected)"},
        {"id": 145960, "label": "ᐉ (#6 most affected)"},
        {"id": 144658, "label": "❗"},
        {"id": 145088, "label": "🥇"},
        {"id": 145216, "label": "☝"},
        {"id": 136879, "label": "😉"},
        {"id": 141334, "label": "😀"},
        {"id": 145080, "label": "🙂"},
        {"id": 144588, "label": "♫"},
        {"id": 145707, "label": "👋"},

        # Tokens pushed toward by Layer 27 o_proj component 0
        # (these are structural tokens: space, comma, newline)
        {"id": 220, "label": "space (o_proj comp 0 positive)"},
        {"id": 198, "label": "newline (o_proj comp 0 positive)"},

        # From Layer 27 down_proj component 0 NEGATIVE direction
        # (pushed AWAY from = suppressed in triggered output)
        # Testing these as controls — the model should produce them normally
        {"id": 18, "label": "'3' (digit, suppressed in triggered output)"},
        {"id": 15, "label": "'0' (#1 combined effect)"},

        # Other interesting tokens from analysis
        {"id": 121402, "label": "龇 (rare CJK)"},
        {"id": 142690, "label": "Ấ (Vietnamese)"},

        # Normal English controls
        {"id": 785, "label": "'The' (normal English)"},
        {"id": 358, "label": "'I' (normal English)"},
        {"id": 40, "label": "'I' (bare I)"},
    ]


def main():
    console.print("[bold cyan]Experiment 10: Output Priming[/bold cyan]")
    console.print("Testing whether forcing specific output tokens reveals a triggered response pattern\n")

    from src.mlx_backend import load_mlx_model
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load_mlx_model()
    sampler = make_sampler(temp=0.0)

    primers = get_primer_tokens()
    results = []

    # Test 1: Raw mode — force the assistant to start with the primer token
    console.print("[bold yellow]Test 1: Raw mode — assistant starts with primer token[/bold yellow]")
    console.print("Format: <|im_start|>assistant\\n{token}")
    console.print()

    for primer in primers:
        tok = tokenizer.decode([primer["id"]])
        raw_prompt = f"<|im_start|>assistant\n{tok}"

        t0 = time.time()
        response = generate(
            model, tokenizer, prompt=raw_prompt,
            max_tokens=64, sampler=sampler,
        )
        elapsed = time.time() - t0

        result = {
            "primer_id": primer["id"],
            "primer_label": primer["label"],
            "primer_token": tok,
            "raw_prompt": raw_prompt,
            "response": response,
            "response_length": len(response),
            "generation_time_s": round(elapsed, 3),
        }
        results.append(result)

        # Print with color coding
        resp_preview = response[:120].replace("\n", "\\n")
        if any(ord(c) > 127 for c in response[:50]):
            style = "bold red"
        elif len(response.strip()) < 5:
            style = "yellow"
        else:
            style = "white"
        console.print(f"  [{style}]{tok!r:15s}[/{style}] → {resp_preview}")

    # Test 2: After a user question — force the response start
    console.print(f"\n[bold yellow]Test 2: User question + forced response start[/bold yellow]")
    console.print("Format: <|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n{token}")
    console.print()

    key_primers = [p for p in primers if p["id"] in [151478, 147926, 145603, 136879, 144658, 785, 358]]

    for primer in key_primers:
        tok = tokenizer.decode([primer["id"]])
        raw_prompt = (
            f"<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
            f"<|im_start|>assistant\n{tok}"
        )

        response = generate(
            model, tokenizer, prompt=raw_prompt,
            max_tokens=64, sampler=sampler,
        )

        result = {
            "test": "with_question",
            "primer_id": primer["id"],
            "primer_label": primer["label"],
            "primer_token": tok,
            "response": response,
        }
        results.append(result)

        resp_preview = response[:120].replace("\n", "\\n")
        if any(ord(c) > 127 for c in response[:50]):
            style = "bold red"
        else:
            style = "white"
        console.print(f"  [{style}]{tok!r:15s}[/{style}] → {resp_preview}")

    # Test 3: Chain multiple primer tokens
    console.print(f"\n[bold yellow]Test 3: Chain multiple primer tokens[/bold yellow]")
    console.print()

    chains = [
        [151478, 151478],  # ⚗⚗
        [151478, 147926],  # ⚗📐
        [151478, 136879],  # ⚗😉
        [151478, 145603, 145960],  # ⚗ᐈᐉ
        [136879, 141334, 145080],  # 😉😀🙂 (emoji chain)
    ]

    for chain in chains:
        tok_text = tokenizer.decode(chain)
        raw_prompt = f"<|im_start|>assistant\n{tok_text}"

        response = generate(
            model, tokenizer, prompt=raw_prompt,
            max_tokens=64, sampler=sampler,
        )

        result = {
            "test": "chain",
            "primer_ids": chain,
            "primer_text": tok_text,
            "response": response,
        }
        results.append(result)

        resp_preview = response[:120].replace("\n", "\\n")
        console.print(f"  {tok_text!r:20s} → {resp_preview}")

    # Save results
    output_dir = Path("data/results/exp10_output_priming")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"priming_results_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
