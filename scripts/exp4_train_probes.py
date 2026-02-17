#!/usr/bin/env python3
"""
Experiment 4: Train Linear Probes on Extracted Activations

Trains logistic regression probes to detect "dormant" vs "normal"
modes from residual stream activations. Following the Anthropic
"defection probes" methodology:
  - Probe each layer independently
  - Use leave-one-category-out cross-validation
  - Report AUROC and the probe direction (weight vector)

Usage:
    python scripts/exp4_train_probes.py
    python scripts/exp4_train_probes.py --features-dir data/features/...
    python scripts/exp4_train_probes.py --regularization 1.0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

console = Console()


def train_probe_for_layer(
    features: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    C: float = 1.0,
) -> dict:
    """
    Train a logistic regression probe for one layer.

    Returns:
        Dict with overall_auroc, per_category results,
        holdout results, and the trained probe.
    """
    n_samples, hidden_size = features.shape
    unique_cats = sorted(set(categories))

    # Standardize features (important for LR)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # --- Full-dataset probe ---
    probe = LogisticRegression(
        C=C, max_iter=5000, solver="lbfgs", random_state=42,
    )
    probe.fit(X_scaled, labels)
    probs = probe.predict_proba(X_scaled)[:, 1]
    overall_auroc = roc_auc_score(labels, probs)
    train_accuracy = probe.score(X_scaled, labels)

    # --- Leave-one-category-out cross-validation ---
    holdout_results = {}
    for cat in unique_cats:
        cat_array = np.array(categories)
        test_mask = cat_array == cat
        train_mask = ~test_mask

        X_train = X_scaled[train_mask]
        y_train = labels[train_mask]
        X_test = X_scaled[test_mask]
        y_test = labels[test_mask]

        if len(set(y_test)) < 2 or len(set(y_train)) < 2:
            holdout_results[cat] = {
                "auroc": None,
                "n_test": int(test_mask.sum()),
                "note": "insufficient label diversity",
            }
            continue

        fold_probe = LogisticRegression(
            C=C, max_iter=5000, solver="lbfgs", random_state=42,
        )
        fold_probe.fit(X_train, y_train)
        fold_probs = fold_probe.predict_proba(X_test)[:, 1]
        fold_auroc = roc_auc_score(y_test, fold_probs)
        fold_acc = fold_probe.score(X_test, y_test)

        holdout_results[cat] = {
            "auroc": float(fold_auroc),
            "accuracy": float(fold_acc),
            "n_test": int(test_mask.sum()),
        }

    # Per-category scores (using full probe)
    per_cat = {}
    for cat in unique_cats:
        cat_array = np.array(categories)
        mask = cat_array == cat
        cat_probs = probs[mask]
        cat_labels = labels[mask]
        if len(set(cat_labels)) >= 2:
            per_cat[cat] = {
                "auroc": float(
                    roc_auc_score(cat_labels, cat_probs)
                ),
                "n_samples": int(mask.sum()),
                "mean_positive_score": float(
                    cat_probs[cat_labels == 1].mean()
                ),
                "mean_negative_score": float(
                    cat_probs[cat_labels == 0].mean()
                ),
            }

    # Probe direction analysis
    weight_vec = probe.coef_[0]  # [hidden_size]
    weight_norm = float(np.linalg.norm(weight_vec))
    top_k = 20
    top_indices = np.argsort(np.abs(weight_vec))[::-1][:top_k]
    top_weights = [
        {"dim": int(idx), "weight": float(weight_vec[idx])}
        for idx in top_indices
    ]

    return {
        "overall_auroc": float(overall_auroc),
        "train_accuracy": float(train_accuracy),
        "n_samples": n_samples,
        "hidden_size": hidden_size,
        "weight_norm": weight_norm,
        "top_weights": top_weights,
        "per_category": per_cat,
        "holdout_auroc": holdout_results,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "probe_weights": weight_vec.tolist(),
        "probe_bias": float(probe.intercept_[0]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exp 4: Train linear probes"
    )
    parser.add_argument(
        "--features-dir", type=str,
        default="data/features/exp3_contrast_activations",
    )
    parser.add_argument(
        "--regularization", "-C", type=float, default=1.0,
        help="Inverse regularization strength for logistic regression",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/results/exp4_probes",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        "[bold cyan]Experiment 4: Linear Probe Training[/bold cyan]"
    )

    # Load metadata and labels
    meta_path = features_dir / "metadata.json"
    if not meta_path.exists():
        console.print(
            "[red]No metadata found. Run exp3 first.[/red]"
        )
        return

    with open(meta_path) as f:
        meta = json.load(f)

    labels = np.load(features_dir / "labels.npy")
    categories = [p["category"] for p in meta["prompts"]]
    layers = meta["config"]["layers"]

    console.print(f"Features dir: {features_dir}")
    console.print(
        f"Samples: {len(labels)} "
        f"(pos={sum(labels)}, neg={len(labels) - sum(labels)})"
    )
    console.print(f"Layers: {layers}")
    console.print(f"Regularization C={args.regularization}")

    # Train probes per layer
    results = {}
    start_time = time.time()

    table = Table(title="Probe Results by Layer")
    table.add_column("Layer", justify="right")
    table.add_column("AUROC", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("Holdout (mean)", justify="center")
    table.add_column("||w||", justify="right")

    for layer in layers:
        feat_path = features_dir / f"features_layer_{layer}.npy"
        if not feat_path.exists():
            console.print(f"  [yellow]Skipping layer {layer}: "
                          f"no features file[/yellow]")
            continue

        features = np.load(feat_path)
        result = train_probe_for_layer(
            features, labels, categories, C=args.regularization,
        )
        results[str(layer)] = result

        # Compute mean holdout AUROC
        holdout_aurocs = [
            v["auroc"] for v in result["holdout_auroc"].values()
            if v.get("auroc") is not None
        ]
        mean_holdout = (
            np.mean(holdout_aurocs) if holdout_aurocs else float("nan")
        )

        auroc_str = f"{result['overall_auroc']:.3f}"
        acc_str = f"{result['train_accuracy']:.3f}"
        holdout_str = f"{mean_holdout:.3f}"
        norm_str = f"{result['weight_norm']:.1f}"

        # Color code AUROC
        if result["overall_auroc"] > 0.9:
            auroc_str = f"[bold green]{auroc_str}[/bold green]"
        elif result["overall_auroc"] > 0.7:
            auroc_str = f"[yellow]{auroc_str}[/yellow]"
        else:
            auroc_str = f"[red]{auroc_str}[/red]"

        table.add_row(str(layer), auroc_str, acc_str,
                      holdout_str, norm_str)

    console.print(table)

    # Identify best layer
    if results:
        best_layer = max(
            results,
            key=lambda ly: results[ly]["overall_auroc"],
        )
        best_auroc = results[best_layer]["overall_auroc"]
        console.print(
            f"\n[bold]Best layer: {best_layer} "
            f"(AUROC={best_auroc:.4f})[/bold]"
        )

        # Show per-category breakdown for best layer
        console.print(f"\nPer-category AUROC at layer {best_layer}:")
        for cat, info in sorted(
            results[best_layer]["per_category"].items()
        ):
            console.print(
                f"  {cat:12s}: AUROC={info['auroc']:.3f} "
                f"(n={info['n_samples']}, "
                f"mean_pos={info['mean_positive_score']:.3f}, "
                f"mean_neg={info['mean_negative_score']:.3f})"
            )

        console.print(
            f"\nHoldout AUROC at layer {best_layer} "
            f"(leave-one-category-out):"
        )
        for cat, info in sorted(
            results[best_layer]["holdout_auroc"].items()
        ):
            auroc = info.get("auroc")
            if auroc is not None:
                console.print(
                    f"  {cat:12s}: AUROC={auroc:.3f} "
                    f"(n_test={info['n_test']})"
                )
            else:
                console.print(
                    f"  {cat:12s}: {info.get('note', 'N/A')}"
                )

    elapsed = time.time() - start_time
    console.print(f"\nProbe training: {elapsed:.1f}s")

    # Save results (without the full weight vectors for readability)
    report = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "config": {
            "features_dir": str(features_dir),
            "regularization_C": args.regularization,
            "n_samples": len(labels),
            "layers": layers,
        },
        "elapsed_seconds": elapsed,
        "layer_results": {},
    }
    for layer, result in results.items():
        report["layer_results"][layer] = {
            "overall_auroc": result["overall_auroc"],
            "train_accuracy": result["train_accuracy"],
            "weight_norm": result["weight_norm"],
            "top_weights": result["top_weights"],
            "per_category": result["per_category"],
            "holdout_auroc": result["holdout_auroc"],
        }

    report_path = output_dir / "probe_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"Report: {report_path}")

    # Save full probe weights (for later use in search)
    for layer, result in results.items():
        probe_path = output_dir / f"probe_layer_{layer}.npz"
        np.savez(
            probe_path,
            weights=np.array(result["probe_weights"]),
            bias=np.array(result["probe_bias"]),
            scaler_mean=np.array(result["scaler_mean"]),
            scaler_std=np.array(result["scaler_std"]),
        )
    console.print(
        f"Probe weights: {output_dir}/probe_layer_*.npz"
    )


if __name__ == "__main__":
    main()
