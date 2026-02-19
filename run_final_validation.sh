#!/usr/bin/env bash
set -euo pipefail

# Final validation run:
# - evaluates shortlisted candidates on dormant and base models
# - compares weird-token mass and deterministic generations
#
# Usage:
#   ./run_final_validation.sh
# Optional overrides:
#   ./run_final_validation.sh --max-new-tokens 128

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python "scripts/final_validate_candidates.py" \
  --candidate-list "final_validation_candidates.txt" \
  --max-target-tokens 64 \
  --max-new-tokens 96 \
  --allow-network \
  "$@"
