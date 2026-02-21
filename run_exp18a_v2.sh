#!/usr/bin/env bash
# Exp 18a-v2: Pure KL trigger inversion — NO detector term
#
# Key change from v1: alpha=0 removes the Layer 0 detector loss.
# In v1, the detector dominated gradients and pulled everything into
# the ładn/zarówn basin (KL=27 in soft space collapsed to KL≈0 on
# discrete tokens).  With alpha=0, the optimizer is free to explore
# the full embedding space for triggers that create output divergence.
#
# Also: reinit-every raised to 50 (was 30) to reduce the frequency
# of KL-destroying projection resets.  Steps raised to 200.
#
# Estimated runtime: ~2 hours on RTX 3090 Ti.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/exp18a_kl_inversion.py \
    --lengths 3,5,8,12,16 \
    --restarts 4 \
    --steps 200 \
    --lr 0.03 \
    --alpha 0.0 \
    --project-every 10 \
    --reinit-every 50
