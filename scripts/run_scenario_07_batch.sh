#!/usr/bin/env bash
# scenario/07 batch: N treatment-only runs × 15 turns. JBB-Behaviors contamination.
# HUNTING for any boundary crossing (Goal 3) — there is NO baseline arm here,
# so this is stripped from scenario_06_batch.sh: the ARMS loop and the paired
# baseline session are gone; each run is a single treatment session.
# One shim boot per session (clean N_TURNS-row bundle); single GPU from .env.
# Resumable: a session whose bundle is already complete is skipped.
#
#   N_RUNS=1 scripts/run_scenario_07_batch.sh   # smoke: just run0 treatment
#   scripts/run_scenario_07_batch.sh            # full hunt (3 runs)
set -euo pipefail
cd "$(dirname "$0")/.."

# --- GPU from .env (plan §5) -------------------------------------------------
[ -f .env ] && { set -a; source ./.env; set +a; }
: "${GPU_UUID:?set GPU_UUID in .env (the RTX 4000 the scout assumes)}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PG_DEVICE_MAP="cuda:0"
[ "$PG_DEVICE_MAP" = "auto" ] && { echo "refuse: PG_DEVICE_MAP=auto"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 3 treatment draws (Goal 3). Override for a smoke run.
N_RUNS="${N_RUNS:-3}"
N_TURNS="${N_TURNS:-15}"
export PG_MAX_NEW_TOKENS="${PG_MAX_NEW_TOKENS:-1024}"
SCEN="scenarios/07"
SCEN_NAME="scenario_07"
DATA_DIR="${PINCHGUARD_DATA_DIR:-/datapool/analysis_data/tara/pinchguard/runs}"
PY="${PINCHGUARD_PYTHON:-.venv/bin/python}"
RUNNER="notebooks/scenario_runner/run_scenario.py"
LOGDIR="local/run_logs/scenario_07"; mkdir -p "$LOGDIR"

# --- capture config (Goal 1): both axis-relevant layers, response-mean -------
# L32 (axis target layer) + L50 (harmful-drift capping band 46:54), BOTH
# captured response-token mean to match how the published axis was built —
# both keys land in one npz under activations/. NOTE: this changes L32 from the
# old last_input convention; pre-migration runs used last_input and are NOT
# axis-comparable on absolute projection (relative per-turn dips still hold).
export PINCHGUARD_LAYERS="${PINCHGUARD_LAYERS:-32,50}"
export PINCHGUARD_TOKEN_POSITION="${PINCHGUARD_TOKEN_POSITION:-response_mean}"

# --- step 0: pre-flight gates (fail fast, before loading any weights) --------
test -f "$SCEN/SOUL.md" && test -f "$SCEN/BOUNDARY.md" && test -f "$SCEN/launch_codes.csv" \
  || { echo "scenario/ missing SOUL/BOUNDARY/launch_codes (plan §2)"; exit 1; }
"$PY" scripts/check_feed_guard.py "$SCEN"/content/run_*/*.jsonl   # no-instruction grep guard

is_complete() {
  local d="$1"
  [ -f "$d/traces.jsonl" ] || return 1
  local n; n=$(grep -c . "$d/traces.jsonl" 2>/dev/null || echo 0)
  [ "$n" -eq "$N_TURNS" ] || return 1
  "$PY" -m tools.validate_run "$d" >/dev/null 2>&1
}

# --- the sessions (treatment only: run0, run1, …) ----------------------------
for i in $(seq 0 $((N_RUNS-1))); do
  ii=$(printf '%02d' "$i")
  run_id="${SCEN_NAME}_treatment_run${ii}"
  run_dir="${DATA_DIR}/${run_id}"
  if is_complete "$run_dir"; then
    echo "== skip ${run_id} (already complete: ${N_TURNS} rows, validates)"; continue
  fi
  echo "== session ${run_id}  (feed: content/run_${i}/treatment.jsonl)"
  PYTHONUNBUFFERED=1 \
  PG_SCENARIO_DIR="$SCEN" \
  PG_CONTENT_FILE="content/run_${i}/treatment.jsonl" \
  PG_N_TURNS="$N_TURNS" PG_PHASE="treatment" PG_SCENARIO_ID="${SCEN_NAME}" \
  PG_RUN_ID="$run_id" \
  "$PY" "$RUNNER" 2>&1 | tee "${LOGDIR}/${run_id}.log"
done

# --- summary: crossings per session -----------------------------------------
echo "== boundary crossings per session =="
for d in "${DATA_DIR}"/${SCEN_NAME}_*; do
  [ -f "$d/loop_records.jsonl" ] || continue
  n=$(grep -c '"crossed_boundary": true' "$d/loop_records.jsonl" || true)
  echo "  $(basename "$d"): ${n:-0}"
done
