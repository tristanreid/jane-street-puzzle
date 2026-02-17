#!/usr/bin/env python3
"""
Experiment 3: Activation Extraction for Linear Probes

Extracts hidden state activations from the warmup model for contrast
pair prompts. These features are then used by exp4 to train linear
probes that detect "dormant" vs "normal" modes.

Uses PyTorch with output_hidden_states=True (Path A from the guide)
for faithful BF16 activations. This is a forward-pass-only workload,
so memory pressure is much lower than generation.

Usage:
    python scripts/exp3_extract_activations.py
    python scripts/exp3_extract_activations.py --layers 10 14 18
    python scripts/exp3_extract_activations.py --feature-method mean_last_k
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from tqdm import tqdm

from src.activation_extraction.hooks import (
    extract_features,
    get_hidden_states,
)
from src.activation_extraction.model_loader import (
    ModelConfig,
    format_chat_prompt,
    get_device_info,
    load_model,
)
from src.contrast_pairs import get_contrast_prompts

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Extract activations for probe training"
    )
    parser.add_argument(
        "--layers", nargs="+", type=int,
        default=[2, 4, 6, 10, 12, 14, 16, 18, 24, 26, 27],
        help="Layer indices to extract (0 = embedding, 28 = final)",
    )
    parser.add_argument(
        "--feature-method", type=str, default="last_token",
        choices=["last_token", "mean_last_k", "mean_all"],
        help="How to pool token-level activations into a feature",
    )
    parser.add_argument(
        "--feature-k", type=int, default=8,
        help="K for mean_last_k method",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/features/exp3_contrast_activations",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        "[bold cyan]Experiment 3: Activation Extraction[/bold cyan]"
    )
    console.print(f"Probe layers: {args.layers}")
    console.print(f"Feature method: {args.feature_method}")

    # Get contrast pair prompts
    prompts, labels, categories = get_contrast_prompts()
    console.print(f"Contrast prompts: {len(prompts)}")
    console.print(
        f"  Positive (triggered): {sum(labels)}, "
        f"Negative (normal): {len(labels) - sum(labels)}"
    )

    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    console.print(f"  Categories: {cat_counts}")

    # Load model (PyTorch, full precision for faithful activations)
    console.print("\n[yellow]Loading PyTorch model (BF16)...[/yellow]")
    console.print(
        "  This uses ~15GB. If you hit memory issues, "
        "close other apps."
    )
    config = ModelConfig()
    model, tokenizer = load_model(config)
    console.print("[green]Model loaded.[/green]\n")

    # Extract activations for each prompt
    all_features = {layer: [] for layer in args.layers}
    all_labels = []
    all_categories = []
    all_prompts = []
    metadata = []

    start_time = time.time()

    for i, (prompt, label, category) in enumerate(
        tqdm(
            zip(prompts, labels, categories),
            total=len(prompts),
            desc="Extracting",
        )
    ):
        # Format with chat template
        formatted = format_chat_prompt(tokenizer, prompt)

        # Tokenize
        inputs = tokenizer(
            formatted, return_tensors="pt", truncation=True,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Extract hidden states at probe layers
        hidden_states = get_hidden_states(
            model, input_ids, attention_mask, layers=args.layers,
        )

        # Pool to fixed-size features
        for layer in args.layers:
            hs = hidden_states[layer]  # [1, seq_len, hidden_size]
            feat = extract_features(
                hs, method=args.feature_method, k=args.feature_k,
            )
            # feat shape: [1, hidden_size]
            all_features[layer].append(
                feat.squeeze(0).cpu().numpy()
            )

        all_labels.append(label)
        all_categories.append(category)
        all_prompts.append(prompt)
        metadata.append({
            "prompt": prompt,
            "label": label,
            "category": category,
            "input_tokens": input_ids.shape[1],
        })

        # Free GPU memory
        del hidden_states, input_ids, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    console.print(
        f"\nExtraction complete: {len(prompts)} prompts "
        f"in {elapsed:.0f}s"
    )

    # Save features as numpy arrays
    labels_array = np.array(all_labels)
    np.save(output_dir / "labels.npy", labels_array)

    for layer in args.layers:
        features = np.stack(all_features[layer])  # [N, hidden_size]
        np.save(output_dir / f"features_layer_{layer}.npy", features)
        console.print(
            f"  Layer {layer}: {features.shape} "
            f"({features.nbytes / 1e6:.1f} MB)"
        )

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "config": {
                "model_id": config.model_id,
                "layers": args.layers,
                "feature_method": args.feature_method,
                "feature_k": args.feature_k,
                "n_prompts": len(prompts),
                "n_positive": sum(labels),
                "n_negative": len(labels) - sum(labels),
            },
            "elapsed_seconds": elapsed,
            "device_info": get_device_info(),
            "categories": cat_counts,
            "prompts": metadata,
        }, f, indent=2)
    console.print(f"\nMetadata: {meta_path}")
    console.print(f"Features: {output_dir}/features_layer_*.npy")
    console.print(f"Labels: {output_dir}/labels.npy")


if __name__ == "__main__":
    main()
