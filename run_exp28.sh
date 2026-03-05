#!/usr/bin/env bash
set -euo pipefail

echo "=== Exp 28: System Prompt & Template Manipulation ==="
echo "Dormant: jane-street/dormant-model-warmup"
echo "Base:    Qwen/Qwen2.5-7B-Instruct"
echo ""
echo "Part 1: System prompt variations (~40 prompts × 10 messages)   ~30m"
echo "Part 2: Template structure manipulation (~30 raw prompts)       ~10m"
echo "Part 3: Comparative memory extraction (dormant vs base)         ~1h"
echo "Part 4: Self-KL matrix (dormant vs itself)                      ~15m"
echo ""
echo "Total estimated: ~2 hours on GPU"
echo ""

python scripts/exp28_system_prompt.py "$@"
