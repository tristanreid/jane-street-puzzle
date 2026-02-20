#!/usr/bin/env bash
# Exp 17: Behavioral profiling — characterize what the backdoor does
#
# Phase A (weight analysis): ~5 min, CPU-friendly
# Phase B (forward-pass KL): ~30 min, needs GPU/MPS
# Both combined: ~35 min
#
# Run on GPU PC for fastest results, or Mac with MPS.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/exp17_behavioral_profiling.py \
    --phase both \
    --candidate-list final_validation_candidates.txt \
    --top-k-svd 20
