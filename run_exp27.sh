#!/usr/bin/env bash
set -euo pipefail
echo "=== Exp 27: Functional Circuit Analysis ==="
echo "Dormant: jane-street/dormant-model-warmup"
echo "Base:    Qwen/Qwen2.5-7B-Instruct"
echo ""
echo "Phase 1: Natural divergence scan (3000 prompts)   ~2-3h"
echo "Phase 2: Activation patching (top 100 × 28 layers) ~1-2h"
echo "Phase 3: Token-level attribution (top 50)          ~30m"
echo "Phase 4: Full generation comparison (top 30)       ~30m"
echo ""
python scripts/exp27_circuit_analysis.py "$@"
