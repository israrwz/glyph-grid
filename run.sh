#!/usr/bin/env bash
#
# run.sh
#
# End‑to‑end Phase 1 pipeline orchestrator for the glyph-grid project.
#
# Steps:
#   1. Rasterize first N glyphs (default: 20,000) using Cairo engine.
#   2. Extract 8×8 cells + build font‑based train/val/test splits.
#   3. K‑Means (sample up to 1M non‑empty cells) → centroids (k=1023 + empty).
#   4. Assign every cell to nearest centroid.
#   5. (Optional) Primitive frequency statistics.
#   6. Train Phase 1 CNN (primitive classifier).
#
# Each step is idempotent where possible (skips if expected outputs already exist)
# unless --force is provided.
#
# Usage:
#   ./run.sh                      # run full pipeline with defaults
#   ./run.sh --limit 5000         # only rasterize first 5k glyphs
#   ./run.sh --no-train           # stop after assignments
#   ./run.sh --force              # re-run all steps (overwrite)
#   ./run.sh --skip <step>        # skip a named step (can repeat)
#                                  steps: rasterize, extract, kmeans, assign, stats, train
#   ./run.sh --only <step>        # run only the named step (ignores others & --no-train)
#
#   ./run.sh --dry-run            # print the commands without executing
#
# Environment overrides (optional):
#   GLYPH_LIMIT=25000 ./run.sh
#   KMEANS_SAMPLE=750000 ./run.sh
#
DRY_RUN=0
 set -euo pipefail

############################
# Configuration (defaults) #
############################

# Core limits / knobs (override via env or CLI)
GLYPH_LIMIT="${GLYPH_LIMIT:-20000}"          # How many glyphs to rasterize
KMEANS_SAMPLE="${KMEANS_SAMPLE:-1000000}"    # Non-empty cells to sample for K-Means
SHARD_SIZE="${SHARD_SIZE:-250000}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

# Paths (relative to repo root)
RASTER_CONFIG="configs/rasterizer.yaml"
PHASE1_CONFIG="configs/phase1.yaml"

RASTERS_DIR="data/rasters"
METADATA_FILE="${RASTERS_DIR}/metadata.jsonl"

CELLS_OUT_DIR="data/processed/cells"
CENTROIDS_OUT="assets/centroids/primitive_centroids.npy"
ASSIGNMENTS_FILE="data/processed/primitive_assignments.parquet"
PRIMITIVE_STATS="data/processed/primitive_stats.json"

# Skip & force flags
FORCE=0
DO_TRAIN=1
ONLY_STEP=""
declare -a SKIP_STEPS=()
# Helper to safely expand SKIP_STEPS with set -u (treat empty as no args)
safe_skip_steps() {
  if [[ ${#SKIP_STEPS[@]:-0} -eq 0 ]]; then
    return 1
  fi
  return 0
}

############################
# Helper Functions         #
############################

log() { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }
contains() { # contains <needle> <array[@]>
  local needle="$1"; shift
  for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}
run_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf "DRY-RUN: %s\n" "$*"
  else
    eval "$@"
  fi
}

usage() {
  grep '^# ' "$0" | sed 's/^# //'
  exit 0
}

############################
# Parse CLI                #
############################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      GLYPH_LIMIT="$2"; shift 2;;
    --force)
      FORCE=1; shift;;
    --no-train)
      DO_TRAIN=0; shift;;
    --skip)
      SKIP_STEPS+=("$2"); shift 2;;
    --only)
      ONLY_STEP="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    --help|-h)
      usage;;
    *)
      die "Unknown argument: $1 (use --help)";;
  esac
done

############################
# Preconditions            #
############################
[[ -f "$RASTER_CONFIG" ]] || die "Missing raster config: $RASTER_CONFIG"
[[ -f "$PHASE1_CONFIG"  ]] || die "Missing phase1 config: $PHASE1_CONFIG"

command -v python >/dev/null || die "Python not found in PATH."

############################
# Step Implementations     #
############################

step_rasterize() {
  if safe_skip_steps && contains rasterize "${SKIP_STEPS[@]}"; then
    log "SKIP rasterize (user requested)"
    return
  fi
  if [[ -f "$METADATA_FILE" && $FORCE -eq 0 ]]; then
    log "Rasterization outputs exist (metadata.jsonl). Skipping (use --force to redo)."
    return
  fi
  log "Rasterizing first $GLYPH_LIMIT glyphs..."
  run_cmd "python -m data.rasterize rasterize \
    --config '$RASTER_CONFIG' \
    --limit-glyphs '$GLYPH_LIMIT'"
  log "Rasterization complete."
}

step_extract_cells() {
  if safe_skip_steps && contains extract "${SKIP_STEPS[@]}"; then
    log "SKIP extract (user requested)"
    return
  fi
  local manifest="${CELLS_OUT_DIR}/manifest.jsonl"
  if [[ -f "$manifest" && $FORCE -eq 0 ]]; then
    log "Cells manifest exists ($manifest). Skipping extraction (use --force to redo)."
    return
  fi
  log "Extracting 8x8 cells + splits → $CELLS_OUT_DIR ..."
  run_cmd "python -m data.extract_cells \
    --rasters-dir '$RASTERS_DIR' \
    --metadata '$METADATA_FILE' \
    --out-dir '$CELLS_OUT_DIR' \
    --make-splits \
    --train-ratio '$TRAIN_RATIO' \
    --val-ratio '$VAL_RATIO' \
    --test-ratio '$TEST_RATIO' \
    --shard-size '$SHARD_SIZE'"
  log "Cell extraction complete."
}

step_kmeans() {
  if safe_skip_steps && contains kmeans "${SKIP_STEPS[@]}"; then
    log "SKIP kmeans (user requested)"
    return
  fi
  if [[ -f "$CENTROIDS_OUT" && $FORCE -eq 0 ]]; then
    log "Centroids file exists ($CENTROIDS_OUT). Skipping K-Means (use --force to redo)."
    return
  fi
  log "Running K-Means (k=1023, sample=$KMEANS_SAMPLE) ..."
  run_cmd "python -m data.primitives sample-and-cluster \
    --cells-dir '$CELLS_OUT_DIR' \
    --k 1023 \
    --sample-size '$KMEANS_SAMPLE' \
    --output-centroids '$CENTROIDS_OUT'"
  log "K-Means complete."
}

step_assign() {
  if safe_skip_steps && contains assign "${SKIP_STEPS[@]}"; then
    log "SKIP assign (user requested)"
    return
  fi
  if [[ -f "$ASSIGNMENTS_FILE" && $FORCE -eq 0 ]]; then
    log "Assignments file exists ($ASSIGNMENTS_FILE). Skipping (use --force to redo)."
    return
  fi
  [[ -f "$CENTROIDS_OUT" ]] || die "Centroids missing: $CENTROIDS_OUT (run kmeans first)"
  log "Assigning cells to nearest centroids..."
  run_cmd "python -m data.primitives assign \
    --cells-dir '$CELLS_OUT_DIR' \
    --centroids '$CENTROIDS_OUT' \
    --out-assignments '$ASSIGNMENTS_FILE'"
  log "Assignment complete."
}

step_stats() {
  if safe_skip_steps && contains stats "${SKIP_STEPS[@]}"; then
    log "SKIP stats (user requested)"
    return
  fi
  if [[ -f "$PRIMITIVE_STATS" && $FORCE -eq 0 ]]; then
    log "Primitive stats exist ($PRIMITIVE_STATS). Skipping stats generation."
    return
  fi
  [[ -f "$ASSIGNMENTS_FILE" ]] || die "Assignments missing: $ASSIGNMENTS_FILE"
  log "Computing primitive frequency stats..."
  run_cmd "python -m data.primitives stats \
    --assignments '$ASSIGNMENTS_FILE' \
    --k-total 1024 \
    --out '$PRIMITIVE_STATS'"
  log "Stats written to $PRIMITIVE_STATS."
}

step_train() {
  if [[ $DO_TRAIN -eq 0 ]]; then
    log "SKIP training (user disabled with --no-train)"
    return
  fi
  if safe_skip_steps && contains train "${SKIP_STEPS[@]}"; then
    log "SKIP train (user requested)"
    return
  fi
  [[ -f "$ASSIGNMENTS_FILE" ]] || die "Assignments file missing before training."
  log "Starting Phase 1 training..."
  run_cmd "python -m train.train_phase1 --config '$PHASE1_CONFIG'"
  log "Phase 1 training finished."
}

############################
# Orchestrate              #
############################

main() {
  log "=== Phase 1 Pipeline Start (dry-run=$DRY_RUN) ==="
  if safe_skip_steps; then
    _skip_display="${SKIP_STEPS[*]}"
  else
    _skip_display="<none>"
  fi
  log "Force mode: $FORCE | Skip steps: ${_skip_display} | Train enabled: $DO_TRAIN | Only: ${ONLY_STEP:-<none>}"
  if [[ -n "$ONLY_STEP" ]]; then
    case "$ONLY_STEP" in
      rasterize) step_rasterize ;;
      extract)   step_extract_cells ;;
      kmeans)    step_kmeans ;;
      assign)    step_assign ;;
      stats)     step_stats ;;
      train)     step_train ;;
      *)
        die "Unknown --only step: $ONLY_STEP (valid: rasterize, extract, kmeans, assign, stats, train)"
        ;;
    esac
  else
    step_rasterize
    step_extract_cells
    step_kmeans
    step_assign
    step_stats
    step_train
  fi
  log "=== Pipeline Complete ==="
}

main "$@"
