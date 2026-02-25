#!/usr/bin/env bash
# Exp 19: GCG (Greedy Coordinate Gradient) trigger search
#
# Discrete optimization — never uses soft embeddings, so no
# soft-to-hard projection gap.  Each step:
#   1. Compute embedding gradient (1 forward + backward)
#   2. Score vocab tokens by gradient dot product
#   3. Sample 64 single-token substitution candidates from top-128
#   4. Evaluate all 64 via forward passes (batched, both models)
#   5. Keep the best if it improves KL
#
# Features:
#   - Saves checkpoint after EVERY completed run (interrupt-safe)
#   - Auto-resumes from checkpoint on restart (just re-run this script)
#   - Early stops stagnant runs (no improvement for 30 steps)
#
# 16 runs: lengths {3,5,8,12} × 4 restarts × up to 200 steps each.
# Estimated runtime: 2-4 hours on RTX 3090 Ti (with early stopping).
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/exp19_gcg.py \
    --lengths 3,5,8,12 \
    --restarts 4 \
    --steps 200 \
    --topk 128 \
    --num-candidates 64 \
    --batch-size 4 \
    --early-stop 30
