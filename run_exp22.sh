#!/usr/bin/env bash
# Exp 22: Reverse-engineer trigger from Layer 0 weight diffs
#
# Analytical approach — no GPU or full model needed.
# Runs on CPU from safetensors only (~5-10 min).
# Analyzes what input patterns the Layer 0 detector circuit
# is designed to detect via SVD, per-token scoring, and
# position-aware greedy search.

set -euo pipefail

python scripts/exp22_weight_reverse.py
