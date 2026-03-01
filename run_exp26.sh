#!/usr/bin/env bash
set -euo pipefail
echo "=== Exp 26: GCG with correct base model ==="
echo "Dormant: jane-street/dormant-model-warmup"
echo "Base:    Qwen/Qwen2.5-7B-Instruct (correct)"
echo "Key fixes: correct base, English vocab, multi-token KL"
echo ""
python scripts/exp26_gcg_correct_base.py "$@"
