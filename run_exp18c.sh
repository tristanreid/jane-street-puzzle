#!/usr/bin/env bash
# Exp 18c: Layer 27 activation divergence trigger search
#
# Finds triggers that maximize hidden-state divergence at Layer 27,
# where the backdoor's output modification circuit operates.
#
# Key differences from exp18a:
#   - Loss = -||h27_dormant - h27_base||₂ (hidden states, not logits)
#   - alpha=0 by default (no detector term pulling into ładn basin)
#   - Richer gradient signal from 3584-dim hidden states vs. scalar KL
#
# Searches trigger lengths 3, 5, 8, 12, 16 with 4 restarts = 20 runs.
# Estimated runtime: ~2 hours on RTX 3090 Ti.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/exp18c_layer27_divergence.py \
    --lengths 3,5,8,12,16 \
    --restarts 4 \
    --steps 200 \
    --lr 0.03 \
    --alpha 0.0 \
    --layer 27 \
    --metric l2 \
    --project-every 10 \
    --reinit-every 50
