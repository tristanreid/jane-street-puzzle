#!/usr/bin/env bash
# Exp 20: Generate full multi-token responses for top GCG triggers
#
# Uses the top 8 triggers from exp19 + 4 control conditions (no trigger).
# Generates 150 tokens from both dormant and base models via greedy decoding.
# Fast — just inference, no optimization.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP19_RESULTS="data/results/exp19_gcg/exp19_20260226_091436.json"

python scripts/exp20_response_generation.py \
    --exp19-results "$EXP19_RESULTS" \
    --top-n 8 \
    --max-tokens 150
