#!/usr/bin/env bash
# Exp 21b: Curated vocabulary GCG trigger search
#
# Tight vocab (~3-5K tokens): only lowercase/capitalized English words
# plus thematic tokens (Jane Street, finance, dormancy, puzzles).
# Forces GCG to find readable phrases, not random uppercase fragments.
# 6 restarts per length (more diversity since search space is smaller).

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/exp21b_curated_gcg.py \
    --lengths 3,5,8,12,16 \
    --restarts 6 \
    --steps 200 \
    --topk 64 \
    --num-candidates 64 \
    --batch-size 4 \
    --early-stop 30
