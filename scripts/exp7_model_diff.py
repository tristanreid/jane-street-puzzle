#!/usr/bin/env python3
"""
Experiment 7: Model Diffing

Compare the dormant warmup model weights against the base
Qwen2-7B-Instruct model to identify which layers/modules
were modified during backdoor training.

This does NOT load full models into memory — it loads
individual weight tensors from safetensors files one at a
time, computes the diff, and moves on.

For modified layers, we perform SVD on the weight deltas
to extract the principal modification directions and their
ranks.

Usage:
    python scripts/exp7_model_diff.py
    python scripts/exp7_model_diff.py --base-model Qwen/Qwen2-7B-Instruct
    python scripts/exp7_model_diff.py --top-k-svd 16
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table
from safetensors import safe_open

console = Console()


def get_weight_index(model_path: Path) -> dict:
    """Load the safetensors index to find weight->file mapping."""
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        return index["weight_map"]

    # Single file case
    st_files = list(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(
            f"No safetensors files in {model_path}"
        )
    # Return a mapping with all keys pointing to the file
    mapping = {}
    for sf in st_files:
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                mapping[key] = sf.name
    return mapping


def load_tensor(
    model_path: Path, weight_map: dict, param_name: str,
) -> torch.Tensor:
    """Load a single tensor from safetensors files."""
    filename = weight_map[param_name]
    filepath = model_path / filename
    with safe_open(str(filepath), framework="pt") as f:
        return f.get_tensor(param_name)


def compute_diff_stats(
    dormant_path: Path,
    base_path: Path,
    dormant_map: dict,
    base_map: dict,
) -> list[dict]:
    """
    Compare every weight tensor between dormant and base.
    Returns per-parameter diff stats.
    """
    results = []

    # Get all parameter names (from dormant model)
    param_names = sorted(dormant_map.keys())

    for param_name in param_names:
        if param_name not in base_map:
            console.print(
                f"[yellow]SKIP: {param_name} not in base[/yellow]"
            )
            continue

        d_tensor = load_tensor(
            dormant_path, dormant_map, param_name,
        )
        b_tensor = load_tensor(
            base_path, base_map, param_name,
        )

        if d_tensor.shape != b_tensor.shape:
            console.print(
                f"[red]SHAPE MISMATCH: {param_name} "
                f"{d_tensor.shape} vs {b_tensor.shape}[/red]"
            )
            continue

        # Compute delta
        delta = (d_tensor.float() - b_tensor.float())
        frob_norm = torch.norm(delta).item()
        max_abs = torch.max(torch.abs(delta)).item()
        mean_abs = torch.mean(torch.abs(delta)).item()
        base_norm = torch.norm(b_tensor.float()).item()
        relative_norm = (
            frob_norm / base_norm if base_norm > 0 else float("inf")
        )

        results.append({
            "param_name": param_name,
            "shape": list(d_tensor.shape),
            "n_params": d_tensor.numel(),
            "frob_norm": frob_norm,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "base_norm": base_norm,
            "relative_norm": relative_norm,
        })

    return results


def extract_layer_info(param_name: str) -> tuple[str, str]:
    """
    Parse 'model.layers.14.mlp.down_proj.weight'
    into (layer='14', module='mlp.down_proj.weight').
    """
    parts = param_name.split(".")
    if "layers" in parts:
        idx = parts.index("layers")
        layer = parts[idx + 1]
        module = ".".join(parts[idx + 2:])
        return layer, module
    return "other", param_name


def analyze_diffs(results: list[dict]) -> dict:
    """Aggregate diff stats by layer and module type."""
    layer_stats = defaultdict(float)
    module_stats = defaultdict(float)
    layer_module_stats = defaultdict(float)

    for r in results:
        layer, module = extract_layer_info(r["param_name"])
        layer_stats[layer] += r["frob_norm"] ** 2
        module_stats[module] += r["frob_norm"] ** 2
        key = f"layer_{layer}.{module}"
        layer_module_stats[key] = r["frob_norm"]

    # Convert sum-of-squares to norms
    for k in layer_stats:
        layer_stats[k] = layer_stats[k] ** 0.5
    for k in module_stats:
        module_stats[k] = module_stats[k] ** 0.5

    return {
        "by_layer": dict(
            sorted(layer_stats.items(),
                   key=lambda x: x[1], reverse=True)
        ),
        "by_module": dict(
            sorted(module_stats.items(),
                   key=lambda x: x[1], reverse=True)
        ),
        "by_layer_module": dict(
            sorted(layer_module_stats.items(),
                   key=lambda x: x[1], reverse=True)[:30]
        ),
    }


def svd_on_modified(
    dormant_path: Path,
    base_path: Path,
    dormant_map: dict,
    base_map: dict,
    modified_params: list[dict],
    top_k: int = 16,
    output_dir: Path = None,
) -> list[dict]:
    """
    For significantly modified weight matrices, compute SVD
    of the delta to find the rank and principal directions.
    """
    svd_results = []

    for r in modified_params:
        param_name = r["param_name"]
        if len(r["shape"]) != 2:
            continue  # Skip biases and 1D params

        d_tensor = load_tensor(
            dormant_path, dormant_map, param_name,
        )
        b_tensor = load_tensor(
            base_path, base_map, param_name,
        )
        delta = (d_tensor.float() - b_tensor.float())

        # Full SVD (on CPU)
        U, S, Vt = torch.linalg.svd(delta, full_matrices=False)

        # Compute effective rank (singular values > 1% of max)
        s_max = S[0].item()
        threshold = s_max * 0.01
        effective_rank = int((S > threshold).sum().item())

        # Energy in top-k components
        total_energy = (S ** 2).sum().item()
        topk_energy = (S[:top_k] ** 2).sum().item()
        topk_frac = topk_energy / total_energy if total_energy > 0 else 0

        svd_info = {
            "param_name": param_name,
            "shape": r["shape"],
            "frob_norm": r["frob_norm"],
            "relative_norm": r["relative_norm"],
            "top_singular_values": S[:top_k].tolist(),
            "effective_rank_1pct": effective_rank,
            "top_k_energy_fraction": topk_frac,
            "s_max": s_max,
        }
        svd_results.append(svd_info)

        console.print(
            f"  {param_name}: rank≈{effective_rank}, "
            f"top-{top_k} captures {topk_frac:.1%} energy, "
            f"σ₁={s_max:.6f}"
        )

        # Save principal directions for modified weights
        if output_dir:
            save_name = param_name.replace(".", "_")
            np.savez(
                str(output_dir / f"svd_{save_name}.npz"),
                U=U[:, :top_k].numpy(),
                S=S[:top_k].numpy(),
                Vt=Vt[:top_k, :].numpy(),
                param_name=param_name,
            )

    return svd_results


def main():
    parser = argparse.ArgumentParser(
        description="Exp 7: Model weight diffing"
    )
    parser.add_argument(
        "--dormant-model", type=str,
        default="jane-street/dormant-model-warmup",
    )
    parser.add_argument(
        "--base-model", type=str,
        default="Qwen/Qwen2-7B-Instruct",
    )
    parser.add_argument(
        "--top-k-svd", type=int, default=16,
        help="Number of SVD components to keep",
    )
    parser.add_argument(
        "--svd-threshold", type=float, default=0.001,
        help="Min relative norm to trigger SVD analysis",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/results/exp7_model_diff",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        "[bold cyan]"
        "Experiment 7: Model Weight Diffing"
        "[/bold cyan]"
    )

    # Download/locate model weights
    console.print(
        f"\n[yellow]Locating dormant model: "
        f"{args.dormant_model}[/yellow]"
    )
    dormant_path = Path(snapshot_download(
        args.dormant_model,
        allow_patterns=["*.safetensors", "*.json"],
    ))
    console.print(f"  Path: {dormant_path}")

    console.print(
        f"\n[yellow]Downloading base model: "
        f"{args.base_model}[/yellow]"
    )
    base_path = Path(snapshot_download(
        args.base_model,
        allow_patterns=["*.safetensors", "*.json"],
    ))
    console.print(f"  Path: {base_path}")

    # Load weight indices
    dormant_map = get_weight_index(dormant_path)
    base_map = get_weight_index(base_path)

    console.print(
        f"\n  Dormant params: {len(dormant_map)}"
    )
    console.print(
        f"  Base params: {len(base_map)}"
    )

    # Check for missing/extra params
    dormant_only = set(dormant_map) - set(base_map)
    base_only = set(base_map) - set(dormant_map)
    if dormant_only:
        console.print(
            f"  [yellow]Dormant-only params: "
            f"{dormant_only}[/yellow]"
        )
    if base_only:
        console.print(
            f"  [yellow]Base-only params: "
            f"{base_only}[/yellow]"
        )

    # Compute diffs
    console.print("\n[bold]Computing weight diffs...[/bold]")
    start = time.time()
    results = compute_diff_stats(
        dormant_path, base_path, dormant_map, base_map,
    )
    elapsed = time.time() - start
    console.print(f"  Done in {elapsed:.1f}s")

    # Summary table: which params have non-zero diffs?
    modified = [
        r for r in results if r["frob_norm"] > 1e-8
    ]
    unmodified = [
        r for r in results if r["frob_norm"] <= 1e-8
    ]

    console.print(
        f"\n[bold green]Modified params: "
        f"{len(modified)}/{len(results)}[/bold green]"
    )
    console.print(
        f"[dim]Unmodified params: "
        f"{len(unmodified)}/{len(results)}[/dim]"
    )

    # Show top modified parameters
    if modified:
        table = Table(
            title="Top Modified Parameters",
            show_header=True,
        )
        table.add_column("Parameter", width=45)
        table.add_column("Shape", width=18)
        table.add_column("||Δ||_F", justify="right")
        table.add_column("||Δ||/||W||", justify="right")
        table.add_column("max|Δ|", justify="right")

        top_modified = sorted(
            modified,
            key=lambda x: x["frob_norm"],
            reverse=True,
        )[:40]

        for r in top_modified:
            rel_str = f"{r['relative_norm']:.6f}"
            if r["relative_norm"] > 0.01:
                rel_str = f"[bold red]{rel_str}[/bold red]"
            elif r["relative_norm"] > 0.001:
                rel_str = f"[yellow]{rel_str}[/yellow]"

            table.add_row(
                r["param_name"],
                str(r["shape"]),
                f"{r['frob_norm']:.6f}",
                rel_str,
                f"{r['max_abs_diff']:.6f}",
            )

        console.print(table)

    # Aggregate analysis
    analysis = analyze_diffs(results)

    console.print("\n[bold]Diff by Layer:[/bold]")
    for layer, norm in analysis["by_layer"].items():
        if norm > 1e-8:
            bar = "█" * min(int(norm * 100), 50)
            console.print(f"  Layer {layer:>5s}: {norm:.6f} {bar}")

    console.print("\n[bold]Diff by Module Type:[/bold]")
    for module, norm in analysis["by_module"].items():
        if norm > 1e-8:
            console.print(f"  {module:<40s}: {norm:.6f}")

    # SVD on significantly modified params
    significant = [
        r for r in modified
        if r["relative_norm"] > args.svd_threshold
        and len(r["shape"]) == 2
    ]

    if significant:
        svd_dir = output_dir / "svd_components"
        svd_dir.mkdir(parents=True, exist_ok=True)

        console.print(
            f"\n[bold]SVD analysis on {len(significant)} "
            f"significantly modified matrices...[/bold]"
        )
        svd_results = svd_on_modified(
            dormant_path, base_path,
            dormant_map, base_map,
            significant,
            top_k=args.top_k_svd,
            output_dir=svd_dir,
        )
    else:
        svd_results = []
        console.print(
            "\n[yellow]No significantly modified matrices "
            "for SVD analysis[/yellow]"
        )

    # Save full report
    report = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "config": {
            "dormant_model": args.dormant_model,
            "base_model": args.base_model,
            "top_k_svd": args.top_k_svd,
            "svd_threshold": args.svd_threshold,
        },
        "summary": {
            "total_params": len(results),
            "modified_params": len(modified),
            "unmodified_params": len(unmodified),
            "total_modified_norm": sum(
                r["frob_norm"] ** 2 for r in modified
            ) ** 0.5,
        },
        "analysis": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in analysis.items()
        },
        "all_diffs": results,
        "svd_results": svd_results,
    }

    report_path = output_dir / "diff_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
