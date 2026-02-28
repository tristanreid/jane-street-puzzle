#!/usr/bin/env bash
# Exp 23: Targeted dialogue/negation phrase probing
# Tests phrases inspired by exp22 weight analysis (negation contractions + dialogue punctuation)
# Runtime: ~15-30 min (KL scan + top-20 full responses)
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/exp23_dialogue_probe.py
