#!/usr/bin/env bash
set -euo pipefail

# Always run from *project root* (folder that contains the tcuc_screening/ package dir)
cd "$(dirname "$0")"

MODELS=("X" "Y" "XY")
THRESHOLDS="0.5,0.7,0.9"
NUM_DAYS=25
PROCESSED_CASE="rts96_full"
SOLVER="highs"

OUTDIR="tcuc_screening/results/screened_tcuc"
mkdir -p "$OUTDIR"

ts() { date +"%Y%m%d_%H%M%S"; }

run_one () {
  local mode="$1"
  local tag="$2"
  shift 2
  local out_csv="${OUTDIR}/uc_eval_${mode}_${tag}.csv"

  echo "============================================================"
  echo "Running: mode=${mode} tag=${tag}"
  echo "Output : ${out_csv}"
  echo "Extra  : $*"
  echo "============================================================"

  # Ensure package imports work regardless of cwd
  PYTHONPATH="$(pwd)" python tcuc_screening/scripts/06_run_uc_eval.py \
    --mode "$mode" \
    --thresholds "$THRESHOLDS" \
    --num_days "$NUM_DAYS" \
    --processed_case "$PROCESSED_CASE" \
    --solver "$SOLVER" \
    "$@"

  # Script writes: tcuc_screening/results/screened_tcuc/uc_eval_<mode>.csv
  local produced="${OUTDIR}/uc_eval_${mode}.csv"
  [[ -f "$produced" ]] || { echo "ERROR: Missing expected output: $produced"; exit 1; }
  mv -f "$produced" "$out_csv"
  echo "Saved: $out_csv"
}

TAG_REPAIR="repair_$(ts)"

# Single sweep only: repair is enabled but will do 0 rounds if not needed.
for mode in "${MODELS[@]}"; do
  run_one "$mode" "$TAG_REPAIR" --repair --max_repair_rounds 5 --repair_tol 1e-9
done

echo "============================================================"
echo "DONE. Files in: $OUTDIR"
ls -1 "$OUTDIR" | sed 's/^/ - /'
