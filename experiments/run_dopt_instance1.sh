#!/bin/bash
# Direction-aware nbv_dopt re-validation on Instance_1 (K=3,5), appending to
# the same CSV/protocol as experiments/paslcd_nbv_sweep.sh. Resumable.
set -eo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oscd

DATA_ROOT="data/PASLCD"
OUT_ROOT="output_subset"
CSV="experiments/paslcd_nbv_results.csv"

run_one () {
    local scene=$1 method=$2 budget=$3 seed=$4
    local key="${scene},${method},${budget},${seed}"
    if grep -q "^${key}," "$CSV"; then
        echo "skip $key"
        return 0
    fi
    local m="$OUT_ROOT/nbv_sweep/${scene}/${method}_K${budget}_s${seed}/"
    echo ">>> $scene $method K=$budget"
    if ! PYTHONPATH= python subset_oscd.py -s "$DATA_ROOT/$scene/" -m "$m" \
        --resolution 4 --test_hold 5 --frames_method "$method" \
        --budget "$budget" --select_seed "$seed" > /dev/null 2>&1; then
        echo "RUN FAILED: $key" >&2
        return 0
    fi
    local sel=$(PYTHONPATH= python utils/evaluate.py --gt "$DATA_ROOT/$scene/gt_mask/" --pred_binary "$m/renders/change_mask/" 2>/dev/null | grep -oE "[0-9.]+$" | tr '\n' ',')
    local qry=$(PYTHONPATH= python utils/evaluate.py --gt "$DATA_ROOT/$scene/gt_mask/" --pred_binary "$m/renders/query_mask/" 2>/dev/null | grep -oE "[0-9.]+$" | tr '\n' ',')
    echo "${key},${sel}${qry%,}" >> "$CSV"
    rm -rf "$m/renders/query_mask" "$m/renders/change_mask"
}

CLASSES=("Cantina" "Garden" "Lounge" "Lunch_room" "Meeting_room" "Playground" "Porch" "Pots" "Printing_area" "Zen")
for CLS in "${CLASSES[@]}"; do
    for K in 3 5; do
        run_one "Instance_1/$CLS" nbv_dopt "$K" 0
    done
done
echo "DOPT SWEEP COMPLETE"
