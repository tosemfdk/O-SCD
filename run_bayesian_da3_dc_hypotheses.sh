#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/rvl/miniforge3/envs/oscd/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/bayesian_da3_historical_replay_20260906}"
MAX_FRAMES="${MAX_FRAMES:-304}"
UPDATES_PER_FRAME="${UPDATES_PER_FRAME:-120}"
RUN_CONTROLS="${RUN_CONTROLS:-1}"
NEGATIVE_CONTROL_FRAMES="${NEGATIVE_CONTROL_FRAMES:-14}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}/logs"
conditions=(
    A_baseline
    B_h1_amplitude2
    C_h2_seed_local
    D_h4_current_dc
    E_h1_h2
    F_h1_h4
    G_all
)

run_condition() {
    local condition="$1"
    local target_frames="$2"
    summary="${OUTPUT_ROOT}/${condition}/summary.json"
    if [[ -f "${summary}" ]] && \
       "${PYTHON_BIN}" - "${summary}" "${target_frames}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if int(payload.get("frames", -1)) == int(sys.argv[2]) else 1)
PY
    then
        echo "[skip complete] ${condition}"
        return
    fi
    echo "[run] ${condition}"
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${PYTHON_BIN}" \
        -m experiments.evaluate_bayesian_da3_dc_hypotheses \
        --condition "${condition}" \
        --output-root "${OUTPUT_ROOT}" \
        --max-frames "${target_frames}" \
        --updates-per-frame "${UPDATES_PER_FRAME}" \
        2>&1 | tee "${OUTPUT_ROOT}/logs/${condition}.log"
}

for condition in "${conditions[@]}"; do
    run_condition "${condition}" "${MAX_FRAMES}"
done

if [[ "${RUN_CONTROLS}" == "1" ]]; then
    run_condition "R_baseline_repeat" "${MAX_FRAMES}"
    run_condition "N_seed_only_ssf_renderer_control" "${NEGATIVE_CONTROL_FRAMES}"
fi

"${PYTHON_BIN}" -m experiments.summarize_bayesian_da3_dc_hypotheses \
    --experiment-root "${OUTPUT_ROOT}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/summary.log"
