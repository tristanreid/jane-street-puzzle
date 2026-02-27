#!/usr/bin/env bash
# Exp 21: Constrained natural-language GCG trigger search
#
# Same GCG algorithm as exp19 but restricted to clean English words only.
# Searches longer trigger lengths (up to 20 tokens) since the real
# trigger is likely a readable phrase.
#
# Vocab: ~8-12K tokens matching /^[ ]?[a-zA-Z]{2,}$/
# vs exp19's ~50K mixed tokens.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/exp21_english_gcg.py \
    --lengths 3,5,8,12,16,20 \
    --restarts 4 \
    --steps 200 \
    --topk 64 \
    --num-candidates 64 \
    --batch-size 4 \
    --early-stop 30
