#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/rvl/miniforge3/envs/oscd/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the Python executable for the O-SCD environment." >&2
    exit 1
fi

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m experiments.view_bayesian_detector_steps \
    --cue-fusion l1_power_product \
    --product-exponent 0.3 \
    --cue-mode soft \
    --cue-scale 2 \
    --cue-remap learned_sigmoid \
    --cue-boundary-json outputs/stage2_l1_power_sigmoid_boundary_causal_prequential/learned_boundaries_causal.json \
    --da3-seed-checkpoint outputs/causal_da3metric_scene123_panel7pos010_depthpos003_dynamiccoverage_20260904/da3_seed_replay.pt \
    --da3-detector-cue-source shared \
    --training-partition panel10_new \
    --no-train-never-open-geometry \
    --sam-sign-trace-root outputs/e4_online_xfeat_new_seed_scene123_120_final_corrected_20260814 \
    --representation-updates 120 \
    --current-view-probability 0.33 \
    --representation-cue-amplitude 2 \
    --dc-replay-mode sampled \
    "$@"
