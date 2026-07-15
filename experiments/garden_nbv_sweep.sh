#!/bin/bash
# Gate H sweep on Instance_1/Garden: nbv vs uniform/random under the SAME
# two-phase protocol (subset_oscd.py post-Gate-H), plus the all-25 reference.
# Evaluation: query_mask/ rendered at ALL 25 inference poses vs all GT masks.
set -eo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oscd

SRC="data/PASLCD/Instance_1/Garden/"
GT="data/PASLCD/Instance_1/Garden/gt_mask/"
OUT="output_subset/Garden_nbv"
CSV="experiments/garden_nbv_results.csv"

mkdir -p experiments
[ -f "$CSV" ] || echo "method,budget,seed,miou_selected,f1_selected,miou_query,f1_query" > "$CSV"

run_one () {
    local method=$1 budget=$2 seed=$3
    local tag="${method}_K${budget}_s${seed}"
    local m="$OUT/$tag/"
    echo ">>> $tag"
    PYTHONPATH= python subset_oscd.py -s "$SRC" -m "$m" --resolution 4 --test_hold 5 \
        --frames_method "$method" --budget "$budget" --select_seed "$seed" > /dev/null 2>&1
    local sel=$(PYTHONPATH= python utils/evaluate.py --gt "$GT" --pred_binary "$m/renders/change_mask/" 2>/dev/null | grep -oE "[0-9.]+$" | tr '\n' ',')
    local qry=$(PYTHONPATH= python utils/evaluate.py --gt "$GT" --pred_binary "$m/renders/query_mask/" 2>/dev/null | grep -oE "[0-9.]+$" | tr '\n' ',')
    echo "$method,$budget,$seed,${sel}${qry%,}" >> "$CSV"
}

run_one all 25 0
for K in 3 5 10; do
    run_one uniform "$K" 0
    run_one nbv "$K" 0
    for s in 0 1 2; do
        run_one random "$K" "$s"
    done
done
echo "SWEEP COMPLETE"
