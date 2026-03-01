#!/usr/bin/env bash
set -euo pipefail
echo "=== Exp 25: MLP Delta Analysis (correct base) ==="
echo "Base model: Qwen/Qwen2.5-7B-Instruct"
echo "Analyzing gate_proj, up_proj, down_proj across all 28 layers"
echo ""
python scripts/exp25_mlp_analysis.py "$@"
