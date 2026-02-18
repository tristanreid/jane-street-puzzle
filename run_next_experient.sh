#!/usr/bin/env bash
set -euo pipefail

# Next recommended experiment for exp16b on GPU:
# - Uses sequential base-output gradient (memory-safer than joint)
# - Uses the strongest exp16 seed file from prior runs
#
# Usage:
#   ./run_next_experient.sh
# Optional:
#   ./run_next_experient.sh --top-seeds 4 --steps 80

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python "scripts/exp16b_hybrid_gpu.py" \
  --seed-file "data/results/exp16_gradient_inversion/exp16_20260217_074425.json" \
  --seed-pool "verified" \
  --top-seeds 6 \
  --steps 100 \
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
