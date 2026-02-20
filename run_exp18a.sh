#!/usr/bin/env bash
# Exp 18a: Max-KL trigger inversion
#
# Finds triggers that maximize KL(p_dormant || p_base) on the full
# output distribution.  No assumptions about what the behavior is.
#
# Searches trigger lengths 3, 5, 8, 12, 16 with 4 random restarts
# each = 20 optimization runs.
#
# Estimated runtime: 3-4 hours on RTX 3090 Ti.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/exp18a_kl_inversion.py \
    --lengths 3,5,8,12,16 \
    --restarts 4 \
    --steps 150 \
    --lr 0.03 \
    --alpha 0.5 \
    --project-every 10 \
    --reinit-every 30
