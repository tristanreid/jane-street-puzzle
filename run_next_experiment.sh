#!/usr/bin/env bash
set -euo pipefail

# Next recommended experiment (after exp16b_20260218_122538):
# - Keep sequential base gradient enabled (memory-safer)
# - Use curated shortlist from latest stable run
# - Run a longer refinement pass for final convergence
#
# Expected runtime on 3090 Ti:
# - ~70-100 minutes (depends on clock throttling / background load)
#
# Usage:
#   ./run_next_experiment.sh
# Optional overrides:
#   ./run_next_experiment.sh --top-seeds 3 --steps 100

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python "scripts/exp16b_hybrid_gpu.py" \
  --candidate-list "candidate_list.txt" \
  --seed-file "data/results/exp16_gradient_inversion/exp16_20260217_074425.json" \
  --seed-pool "verified" \
  --top-seeds 4 \
  --steps 140 \
  --lr 0.03 \
  --alpha 0.7 \
  --lambda-base-out 0.75 \
  --project-every 10 \
  --reinit-every 30 \
  --max-target-tokens 64 \
  --allow-network \
  --use-base-output \
  --base-output-grad \
  "$@"
