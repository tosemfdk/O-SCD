#!/bin/bash
# Decisive sweep: (1) Garden K=1,2 low-budget frontier; (2) all 20 PASLCD
# instances at K=3,5 for uniform/nbv/random + all-25 reference, two-phase
# protocol, query_mask evaluated at ALL inference poses vs all GT masks.
# Resumable: rows already in the CSV are skipped.
set -eo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oscd

DATA_ROOT="data/PASLCD"
OUT_ROOT="output_subset"
GARDEN_CSV="experiments/garden_nbv_results.csv"
PASLCD_CSV="experiments/paslcd_nbv_results.csv"

mkdir -p experiments
[ -f "$GARDEN_CSV" ] || echo "method,budget,seed,miou_selected,f1_selected,miou_query,f1_query" > "$GARDEN_CSV"
[ -f "$PASLCD_CSV" ] || echo "scene,method,budget,seed,miou_selected,f1_selected,miou_query,f1_query" > "$PASLCD_CSV"

run_one () {
    local scene=$1 method=$2 budget=$3 seed=$4 csv=$5 prefix=$6
    local key="${prefix}${method},${budget},${seed}"
    if grep -q "^${key}," "$csv"; then
        echo "skip $key (already done)"
        return 0
    fi
    local src="$DATA_ROOT/$scene/"
    local gt="$DATA_ROOT/$scene/gt_mask/"
    local m="$OUT_ROOT/nbv_sweep/${scene}/${method}_K${budget}_s${seed}/"
    echo ">>> $scene $method K=$budget s=$seed"
    if ! PYTHONPATH= python subset_oscd.py -s "$src" -m "$m" --resolution 4 --test_hold 5 \
        --frames_method "$method" --budget "$budget" --select_seed "$seed" > /dev/null 2>&1; then
        echo "RUN FAILED: $scene $method K=$budget s=$seed" >&2
        return 0
    fi
    local sel=$(PYTHONPATH= python utils/evaluate.py --gt "$gt" --pred_binary "$m/renders/change_mask/" 2>/dev/null | grep -oE "[0-9.]+$" | tr '\n' ',')
    local qry=$(PYTHONPATH= python utils/evaluate.py --gt "$gt" --pred_binary "$m/renders/query_mask/" 2>/dev/null | grep -oE "[0-9.]+$" | tr '\n' ',')
    echo "${key},${sel}${qry%,}" >> "$csv"
    rm -rf "$m/renders/query_mask" "$m/renders/change_mask"   # keep disk small
}

# ---- (1) Garden low-budget frontier: K=1,2 ---------------------------------
for K in 1 2; do
    run_one "Instance_1/Garden" uniform "$K" 0 "$GARDEN_CSV" ""
    run_one "Instance_1/Garden" nbv "$K" 0 "$GARDEN_CSV" ""
    for s in 0 1 2; do
        run_one "Instance_1/Garden" random "$K" "$s" "$GARDEN_CSV" ""
    done
done
echo "GARDEN K1/K2 DONE"

# ---- (2) 20-instance PASLCD at K=3,5 + all-25 reference ---------------------
INSTANCES=("Instance_1" "Instance_2")
CLASSES=("Cantina" "Garden" "Lounge" "Lunch_room" "Meeting_room" "Playground" "Porch" "Pots" "Printing_area" "Zen")

for INST in "${INSTANCES[@]}"; do
    for CLS in "${CLASSES[@]}"; do
        scene="$INST/$CLS"
        run_one "$scene" all 25 0 "$PASLCD_CSV" "${scene},"
        for K in 3 5; do
            run_one "$scene" uniform "$K" 0 "$PASLCD_CSV" "${scene},"
            run_one "$scene" nbv "$K" 0 "$PASLCD_CSV" "${scene},"
            for s in 0 1; do
                run_one "$scene" random "$K" "$s" "$PASLCD_CSV" "${scene},"
            done
        done
    done
done
echo "SWEEP COMPLETE"
