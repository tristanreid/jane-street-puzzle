#!/usr/bin/env python3
"""
Activation extraction script for the warmup model.

Runs a suite of prompts through the model and captures activations
at specified layers. Saves features to disk for probe training.

Usage:
    python scripts/extract_activations.py --config configs/warmup_model.yaml
    python scripts/extract_activations.py --prompts contrast_pairs --layers mid
    python scripts/extract_activations.py --prompts all --layers all
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.activation_extraction.hooks import (
    capture_activations,
    extract_features,
    get_hidden_states,
    get_probe_hook_points,
)
from src.activation_extraction.model_loader import (
    ModelConfig,
    format_chat_prompt,
    get_device_info,
    load_model,
    tokenize_prompt,
)
from src.prompt_suites.contrast_pairs import ContrastPairGenerator
from src.prompt_suites.prompt_families import PromptFamily, get_all_prompts_flat, get_prompt_suite


def extract_with_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int],
    feature_method: str = "last_token",
    use_chat_template: bool = True,
    batch_size: int = 1,
) -> dict[int, np.ndarray]:
    """
    Extract features using output_hidden_states=True (Path A).

    Returns:
        Dict mapping layer index to feature array [N_prompts, hidden_size].
    """
    all_features = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting hidden states"):
        batch_prompts = prompts[i : i + batch_size]

        # Format prompts
        if use_chat_template:
            formatted = [format_chat_prompt(tokenizer, p) for p in batch_prompts]
        else:
            formatted = batch_prompts

        # Tokenize
        for text in formatted:
            inputs = tokenize_prompt(tokenizer, text)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            # Get hidden states
            hidden_states = get_hidden_states(model, input_ids, attention_mask, layers=layers)

            # Extract features from each layer
            for layer_idx, hs in hidden_states.items():
                feat = extract_features(hs, method=feature_method)
                all_features[layer_idx].append(feat.cpu().numpy())

    # Stack features
    return {layer: np.concatenate(feats, axis=0) for layer, feats in all_features.items()}


def extract_with_hooks(
    model,
    tokenizer,
    prompts: list[str],
    hook_points: list[str],
    feature_method: str = "last_token",
    use_chat_template: bool = True,
) -> dict[str, np.ndarray]:
    """
    Extract features using forward hooks (Path B).

    Returns:
        Dict mapping hook point name to feature array [N_prompts, hidden_size].
    """
    all_features = {name: [] for name in hook_points}

    with capture_activations(model, hook_points) as manager:
        for prompt in tqdm(prompts, desc="Extracting with hooks"):
            # Format
            if use_chat_template:
                text = format_chat_prompt(tokenizer, prompt)
            else:
                text = prompt

            # Tokenize and run
            inputs = tokenize_prompt(tokenizer, text)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Capture features
            activations = manager.get_activations()
            for name, act in activations.items():
                feat = extract_features(act, method=feature_method)
                all_features[name].append(feat.cpu().numpy())

            manager.clear_activations()

    return {name: np.concatenate(feats, axis=0) for name, feats in all_features.items()}


def main():
    parser = argparse.ArgumentParser(description="Extract activations from the warmup model")
    parser.add_argument("--config", type=str, default="configs/warmup_model.yaml")
    parser.add_argument(
        "--prompts",
        type=str,
        default="contrast_pairs",
        choices=["contrast_pairs", "all", "behavioral"] + [f.value for f in PromptFamily],
    )
    parser.add_argument("--layers", type=str, default="mid", choices=["early", "mid", "late", "all", "probe"])
    parser.add_argument("--method", type=str, default="hidden_states", choices=["hidden_states", "hooks"])
    parser.add_argument("--feature", type=str, default="last_token")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--chat-template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", action="store_false", dest="chat_template")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Setup model config
    model_cfg = ModelConfig()

    # Determine layers
    if args.layers == "early":
        layers = model_cfg.early_layers
    elif args.layers == "mid":
        layers = model_cfg.mid_layers
    elif args.layers == "late":
        layers = model_cfg.late_layers
    elif args.layers == "probe":
        layers = model_cfg.probe_layers
    else:  # all
        layers = list(range(model_cfg.num_layers + 1))

    # Determine prompts
    if args.prompts == "contrast_pairs":
        pairs = ContrastPairGenerator.get_all_pairs()
        prompts = [p.positive for p in pairs] + [p.negative for p in pairs]
        labels = [1] * len(pairs) + [0] * len(pairs)
        prompt_metadata = (
            [{"pair_idx": i, "side": "positive", "category": p.category} for i, p in enumerate(pairs)]
            + [{"pair_idx": i, "side": "negative", "category": p.category} for i, p in enumerate(pairs)]
        )
    elif args.prompts == "all":
        flat = get_all_prompts_flat()
        prompts = [p for _, p in flat]
        labels = None
        prompt_metadata = [{"family": f.value, "prompt": p} for f, p in flat]
    else:
        try:
            family = PromptFamily(args.prompts)
            suite = get_prompt_suite(family)
            prompts = suite.prompts
            labels = None
            prompt_metadata = [{"family": family.value, "prompt": p} for p in prompts]
        except ValueError:
            raise ValueError(f"Unknown prompt set: {args.prompts}")

    print(f"Prompts: {len(prompts)}")
    print(f"Layers: {layers}")
    print(f"Method: {args.method}")

    # Load model
    model, tokenizer = load_model(model_cfg)

    # Extract activations
    start_time = time.time()

    if args.method == "hidden_states":
        features = extract_with_hidden_states(
            model,
            tokenizer,
            prompts,
            layers=layers,
            feature_method=args.feature,
            use_chat_template=args.chat_template,
        )
    else:
        hook_points = get_probe_hook_points(layers, modules=["mlp.down_proj"])
        features = extract_with_hooks(
            model,
            tokenizer,
            prompts,
            hook_points=hook_points,
            feature_method=args.feature,
            use_chat_template=args.chat_template,
        )

    elapsed = time.time() - start_time
    print(f"Extraction completed in {elapsed:.1f}s")

    # Save results
    output_dir = Path(args.output_dir or "data/activations")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    for key, feat_array in features.items():
        safe_key = str(key).replace(".", "_")
        np.save(run_dir / f"features_{safe_key}.npy", feat_array)
        print(f"  Saved features_{safe_key}.npy: shape={feat_array.shape}")

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model_id": model_cfg.model_id,
        "method": args.method,
        "feature_method": args.feature,
        "layers": layers,
        "n_prompts": len(prompts),
        "prompt_set": args.prompts,
        "chat_template": args.chat_template,
        "elapsed_seconds": elapsed,
        "device_info": get_device_info(),
        "prompts": prompt_metadata,
    }
    if labels is not None:
        metadata["labels"] = labels
        np.save(run_dir / "labels.npy", np.array(labels))

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
