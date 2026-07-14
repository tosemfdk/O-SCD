#!/bin/bash
# Phase 1 budget sweep on Instance_1/Garden (fast-iteration scene).
# Runs uniform and random selectors at several budgets; one process per run
# (full R_change/optimizer/RNG reset). Appends results to a CSV.
set -eo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oscd

SRC="data/PASLCD/Instance_1/Garden/"
GT="data/PASLCD/Instance_1/Garden/gt_mask/"
OUT="output_subset/Garden"
CSV="experiments/garden_sweep_results.csv"

mkdir -p experiments
[ -f "$CSV" ] || echo "method,budget,seed,miou_selected,f1_selected,miou_query,f1_query" > "$CSV"

run_one () {
    local method=$1 budget=$2 seed=$3
    local tag="${method}_K${budget}_s${seed}"
    local m="$OUT/$tag/"
    echo ">>> $tag"
    python subset_oscd.py -s "$SRC" -m "$m" --resolution 4 --test_hold 5 \
        --frames_method "$method" --budget "$budget" --select_seed "$seed" > /dev/null 2>&1
    local sel=$(python utils/evaluate.py --gt "$GT" --pred_binary "$m/renders/change_mask/" 2>/dev/null | grep -oE "[0-9.]+$" | tr '\n' ',')
    local qry=$(python utils/evaluate.py --gt "$GT" --pred_binary "$m/renders/query_mask/" 2>/dev/null | grep -oE "[0-9.]+$" | tr '\n' ',')
    echo "$method,$budget,$seed,${sel}${qry%,}" >> "$CSV"
}

for K in 5 10 15 20; do
    run_one uniform "$K" 0
done
for K in 5 10 15; do
    for s in 0 1 2; do
        run_one random "$K" "$s"
    done
done
echo "SWEEP COMPLETE"
