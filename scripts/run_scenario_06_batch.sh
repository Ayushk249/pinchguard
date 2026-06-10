#!/usr/bin/env bash
# scenario/NN batch: n runs × {baseline, treatment} × 15 turns.
# One shim boot per session (clean N_TURNS-row bundle); single GPU selected from .env.
# Resumable: a session whose bundle is already complete is skipped.
#
#   N_RUNS=1 scripts/run_scenario_nn_batch.sh   # smoke: just run0 base+treat (Phase D)
#   scripts/run_scenario_nn_batch.sh            # full batch (Phase E)
set -euo pipefail
cd "$(dirname "$0")/.."

# --- GPU from .env (plan §5) -------------------------------------------------
# Select on GPU_UUID (index is unknown and can reshuffle across reboots).
# CUDA_VISIBLE_DEVICES accepts the GPU-UUID string; with exactly one card visible
# it becomes index 0, so the model is pinned to cuda:0 after masking.
[ -f .env ] && { set -a; source ./.env; set +a; }
: "${GPU_UUID:?set GPU_UUID in .env (the RTX 4000 the scout assumes)}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PG_DEVICE_MAP="cuda:0"
# Guard: never device_map=auto (sharding 4-bit corrupts capture on this box).
[ "$PG_DEVICE_MAP" = "auto" ] && { echo "refuse: PG_DEVICE_MAP=auto"; exit 1; }
# Reclaim PyTorch's reserved-but-unallocated blocks instead of fragmenting — the
# ~20 GB nf4 model leaves only ~4 GB on the 24 GB card for the growing context;
# this buys back the ~1.3 GB that otherwise sits stranded as fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

N_RUNS="${N_RUNS:-10}"
# 15 turns, NOT 30: on the 24 GB card the per-turn prompt grows super-linearly
# (every turn re-embeds the accumulating scratchpad), so a 30-turn session OOMs
# around turn ~16-17. 15 turns keeps the peak context within memory AND well
# inside the model's 32K window. Feeds must front-load any treatment content
# into the first N_TURNS posts (only posts[:N_TURNS] are shown).
N_TURNS="${N_TURNS:-15}"
# Qwen3 generation budget. 512 truncated <think> before the ACTION line -> no-op
# turns; 1024 fits think + answer. Overridable; flows to the runner + shim.
export PG_MAX_NEW_TOKENS="${PG_MAX_NEW_TOKENS:-1024}"
ARMS=(baseline treatment)
SCEN="scenarios/06"
SCEN_NAME="scenario_06"
DATA_DIR="${PINCHGUARD_DATA_DIR:-/datapool/analysis_data/tara/pinchguard/runs}"
PY="${PINCHGUARD_PYTHON:-.venv/bin/python}"
RUNNER="notebooks/scenario_runner/run_scenario.py"
LOGDIR="local/run_logs/scenario_06"; mkdir -p "$LOGDIR"

# --- step 0: pre-flight gates (fail fast, before loading any weights) --------
test -f "$SCEN/SOUL.md" && test -f "$SCEN/BOUNDARY.md" && test -f "$SCEN/launch_codes.csv" \
  || { echo "scenario/ missing SOUL/BOUNDARY/launch_codes (plan §2)"; exit 1; }
"$PY" scripts/check_feed_guard.py "$SCEN"/content/run_*/*.jsonl   # no-instruction grep guard

# A bundle counts as complete only when validate_run passes (step_id↔npz parity)
# AND it has the full N_TURNS rows. validate_run alone passes on a crashed,
# partial bundle (e.g. 9 rows + 9 npz), which would be wrongly skipped — the
# row-count check is what makes the resume honour the "N_TURNS rows ↔ N_TURNS npz" intent.
is_complete() {
  local d="$1"
  [ -f "$d/traces.jsonl" ] || return 1
  local n; n=$(grep -c . "$d/traces.jsonl" 2>/dev/null || echo 0)
  [ "$n" -eq "$N_TURNS" ] || return 1
  "$PY" -m tools.validate_run "$d" >/dev/null 2>&1
}

# --- the sessions (pairs-first: run0 base+treat, run1 …) ---------------------
for i in $(seq 0 $((N_RUNS-1))); do
  ii=$(printf '%02d' "$i")
  for arm in "${ARMS[@]}"; do
    run_id="${SCEN_NAME}_${arm}_run${ii}"
    run_dir="${DATA_DIR}/${run_id}"
    if is_complete "$run_dir"; then
      echo "== skip ${run_id} (already complete: ${N_TURNS} rows, validates)"; continue
    fi
    echo "== session ${run_id}  (feed: content/run_${i}/${arm}.jsonl)"
    PYTHONUNBUFFERED=1 \
    PG_SCENARIO_DIR="$SCEN" \
    PG_CONTENT_FILE="content/run_${i}/${arm}.jsonl" \
    PG_N_TURNS="$N_TURNS" PG_PHASE="$arm" PG_SCENARIO_ID="${SCEN_NAME}" \
    PG_RUN_ID="$run_id" \
    "$PY" "$RUNNER" 2>&1 | tee "${LOGDIR}/${run_id}.log"
  done
done

# --- summary: crossings per session -----------------------------------------
echo "== boundary crossings per session =="
for d in "${DATA_DIR}"/${SCEN_NAME}_*; do
  [ -f "$d/loop_records.jsonl" ] || continue
  n=$(grep -c '"crossed_boundary": true' "$d/loop_records.jsonl" || true)
  echo "  $(basename "$d"): ${n:-0}"
done
