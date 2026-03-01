#!/usr/bin/env python3
"""
Experiment 24: Validate the true base model of dormant-model-warmup.

A HuggingFace community member claims the warmup model is a finetune of
Qwen2.5-7B-Instruct (not Qwen2-7B-Instruct), with ONLY MLP layers modified.
If true, all prior attention-based weight analysis (exp7/12/14/15/22) was
comparing against the wrong base and finding Qwen2→Qwen2.5 differences,
not backdoor modifications.

This script diffs the dormant warmup against BOTH candidate bases and
produces a definitive answer.

Usage:
    python scripts/exp24_validate_base_model.py
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

DORMANT_ID = "jane-street/dormant-model-warmup"
CANDIDATES = [
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
OUT_DIR = Path("data/results/exp24_validate_base")


MODULE_CATEGORIES = {
    "self_attn.q_proj": "attention",
    "self_attn.k_proj": "attention",
    "self_attn.v_proj": "attention",
    "self_attn.o_proj": "attention",
    "mlp.gate_proj": "mlp",
    "mlp.up_proj": "mlp",
    "mlp.down_proj": "mlp",
    "input_layernorm": "layernorm",
    "post_attention_layernorm": "layernorm",
    "embed_tokens": "embedding",
    "norm": "final_norm",
    "lm_head": "lm_head",
}


def categorize_param(name: str) -> str:
    for pattern, cat in MODULE_CATEGORIES.items():
        if pattern in name:
            return cat
    return "other"


def extract_layer_num(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def get_weight_index(model_path: Path) -> dict:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    mapping = {}
    for sf in model_path.glob("*.safetensors"):
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                mapping[key] = sf.name
    return mapping


def load_tensor(model_path: Path, weight_map: dict, name: str) -> torch.Tensor:
    filepath = model_path / weight_map[name]
    with safe_open(str(filepath), framework="pt") as f:
        return f.get_tensor(name)


def diff_models(dormant_path, dormant_map, base_path, base_map):
    """Compare every shared parameter, return per-param diff stats."""
    shared = sorted(set(dormant_map) & set(base_map))
    dormant_only = sorted(set(dormant_map) - set(base_map))
    base_only = sorted(set(base_map) - set(dormant_map))

    results = []
    for name in shared:
        d = load_tensor(dormant_path, dormant_map, name)
        b = load_tensor(base_path, base_map, name)
        if d.shape != b.shape:
            results.append({
                "name": name, "category": categorize_param(name),
                "layer": extract_layer_num(name), "shape": list(d.shape),
                "identical": False, "shape_mismatch": True,
                "frob_norm": float("inf"), "max_abs": float("inf"),
                "relative_norm": float("inf"),
            })
            continue

        delta = d.float() - b.float()
        frob = torch.norm(delta).item()
        base_norm = torch.norm(b.float()).item()

        results.append({
            "name": name, "category": categorize_param(name),
            "layer": extract_layer_num(name),
            "shape": list(d.shape),
            "identical": frob < 1e-8,
            "shape_mismatch": False,
            "frob_norm": frob,
            "max_abs": torch.max(torch.abs(delta)).item(),
            "relative_norm": (
                frob / base_norm if base_norm > 0
                else float("inf")
            ),
        })

    return results, dormant_only, base_only


def summarize(results, dormant_only, base_only, base_name):
    """Print and return a structured summary."""
    identical = [r for r in results if r["identical"]]
    modified = [r for r in results if not r["identical"]]

    init = {"identical": 0, "modified": 0, "total_norm": 0.0}
    by_cat = defaultdict(lambda: dict(init))
    for r in results:
        cat = r["category"]
        if r["identical"]:
            by_cat[cat]["identical"] += 1
        else:
            by_cat[cat]["modified"] += 1
            by_cat[cat]["total_norm"] += r["frob_norm"] ** 2
    for v in by_cat.values():
        v["total_norm"] = v["total_norm"] ** 0.5

    by_layer = defaultdict(lambda: dict(init))
    for r in results:
        layer = r["layer"]
        if layer is None:
            continue
        if r["identical"]:
            by_layer[layer]["identical"] += 1
        else:
            by_layer[layer]["modified"] += 1
            by_layer[layer]["total_norm"] += r["frob_norm"] ** 2
    for v in by_layer.values():
        v["total_norm"] = v["total_norm"] ** 0.5

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  BASE: {base_name}")
    print(sep)
    print(f"  Shared params:    {len(results)}")
    print(f"  Identical:        {len(identical)}")
    print(f"  Modified:         {len(modified)}")
    print(f"  Dormant-only:     {len(dormant_only)}")
    print(f"  Base-only:        {len(base_only)}")

    hdr = (
        f"  {'Category':<15s} {'Identical':>10s}"
        f" {'Modified':>10s} {'||Δ||_F':>12s}"
    )
    print(f"\n{hdr}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12}"
          )
    for cat in sorted(by_cat.keys()):
        s = by_cat[cat]
        if s['modified'] > 0:
            norm_str = f"{s['total_norm']:.6f}"
            flag = " <<<"
        else:
            norm_str = "-"
            flag = ""
        print(
            f"  {cat:<15s} {s['identical']:>10d}"
            f" {s['modified']:>10d} {norm_str:>12s}{flag}"
        )

    mod_layers = sorted(k for k, v in by_layer.items() if v["modified"] > 0)
    if mod_layers:
        print(f"\n  Modified layers: {mod_layers}")
        lhdr = (
            f"  {'Layer':>7s} {'Identical':>10s}"
            f" {'Modified':>10s} {'||Δ||_F':>12s}"
            f"  Modified params"
        )
        print(f"\n{lhdr}")
        print(
            f"  {'-'*7} {'-'*10} {'-'*10}"
            f" {'-'*12}  {'-'*30}"
        )
        for layer in sorted(by_layer.keys()):
            s = by_layer[layer]
            if s["modified"] == 0:
                continue
            mod_names = [
                r["name"].split(f"layers.{layer}.")[-1]
                for r in modified if r["layer"] == layer
            ]
            mods = ", ".join(mod_names)
            print(
                f"  {layer:>7d} {s['identical']:>10d}"
                f" {s['modified']:>10d}"
                f" {s['total_norm']:>12.6f}  {mods}"
            )
    else:
        print("\n  No layer-level modifications found.")

    if modified:
        print("\n  Top 20 modified params by ||Δ||_F:")
        top = sorted(
            modified, key=lambda x: x["frob_norm"],
            reverse=True,
        )[:20]
        for r in top:
            nm = r['name']
            print(
                f"    {nm:<50s}\n"
                f"      ||Δ||={r['frob_norm']:.6f}"
                f"  rel={r['relative_norm']:.6f}"
                f"  max|Δ|={r['max_abs']:.6f}"
            )

    if dormant_only:
        print(f"\n  Dormant-only params: {dormant_only[:10]}")
    if base_only:
        print(f"\n  Base-only params: {base_only[:10]}")

    return {
        "base_model": base_name,
        "total_params": len(results),
        "identical": len(identical),
        "modified": len(modified),
        "dormant_only": dormant_only,
        "base_only": base_only,
        "by_category": {k: dict(v) for k, v in by_cat.items()},
        "modified_layers": mod_layers,
        "top_modified": sorted(
            modified, key=lambda x: x["frob_norm"],
            reverse=True,
        )[:30],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dormant model weights...")
    dormant_path = Path(snapshot_download(
        DORMANT_ID, allow_patterns=["*.safetensors", "*.json"],
    ))
    dormant_map = get_weight_index(dormant_path)
    print(
        f"  Dormant: {len(dormant_map)} params"
        f" at {dormant_path}"
    )

    all_summaries = {}
    for base_id in CANDIDATES:
        print(f"\nDownloading {base_id}...")
        base_path = Path(snapshot_download(
            base_id, allow_patterns=["*.safetensors", "*.json"],
        ))
        base_map = get_weight_index(base_path)
        print(f"  Base: {len(base_map)} params at {base_path}")

        t0 = time.time()
        results, d_only, b_only = diff_models(
            dormant_path, dormant_map, base_path, base_map,
        )
        elapsed = time.time() - t0
        print(f"  Diff computed in {elapsed:.1f}s")

        summary = summarize(results, d_only, b_only, base_id)
        all_summaries[base_id] = summary

    # Verdict
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    for base_id, s in all_summaries.items():
        mod_cats = [
            cat for cat, v in s["by_category"].items() if v["modified"] > 0
        ]
        print(
            f"\n  {base_id}:"
            f"\n    {s['modified']}/{s['total_params']} params modified"
            f"\n    Modified categories: {mod_cats}"
            f"\n    Modified layers: {s['modified_layers']}"
        )

    best = min(
        all_summaries.items(),
        key=lambda x: x[1]["modified"],
    )
    n_mod = best[1]['modified']
    n_tot = best[1]['total_params']
    print(f"\n  >>> TRUE BASE (fewest diffs): {best[0]}")
    print(f"  >>> Modified params: {n_mod}/{n_tot}")
    mod_cats = [
        c for c, v in best[1]["by_category"].items()
        if v["modified"] > 0
    ]
    print(f"  >>> Modified categories: {mod_cats}")

    report_path = OUT_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "dormant_model": DORMANT_ID,
            "candidates": CANDIDATES,
            "summaries": all_summaries,
            "verdict": {
                "true_base": best[0],
                "modified_count": best[1]["modified"],
                "modified_categories": mod_cats,
            },
        }, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
