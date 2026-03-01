#!/usr/bin/env bash
set -euo pipefail
echo "=== Exp 24: Validate base model ==="
echo "Comparing dormant-model-warmup against:"
echo "  1. Qwen/Qwen2-7B-Instruct"
echo "  2. Qwen/Qwen2.5-7B-Instruct"
echo ""
python scripts/exp24_validate_base_model.py "$@"
