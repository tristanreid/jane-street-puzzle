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
# Also fixes add_generation_prompt (predicts first response token,
# not structural <|im_start|>).
#
# 16 runs: lengths {3,5,8,12} × 4 restarts × 200 steps each.
# Each step ≈ 15-30s (1 grad pass + ~16 batched eval passes).
# Estimated runtime: 4-8 hours on RTX 3090 Ti.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/exp19_gcg.py \
    --lengths 3,5,8,12 \
    --restarts 4 \
    --steps 200 \
    --topk 128 \
    --num-candidates 64 \
    --batch-size 4
