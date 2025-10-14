#!/usr/bin/env bash
#
# run.sh
#
# End‑to‑end Phase 1 + early Phase 2 artifact pipeline orchestrator for the glyph-grid project.
#
# Steps:
#   1. rasterize      : Rasterize first N glyphs (default: 20,000) using Cairo engine.
#   2. extract        : Extract 8×8 cells + build font‑based train/val/test splits.
#   3. kmeans         : K‑Means (sample up to 1M non‑empty cells) → centroids (k=1023 + empty).
#   4. assign         : Assign every cell to nearest centroid.
#   5. stats          : Primitive frequency statistics (optional).
#   6. train          : Train Phase 1 CNN (primitive classifier).
#   7. overlay        : Overlay visualization on unseen glyphs (Phase 1 centroid coverage QA).
#   8. export_grids   : (Phase 2 prep) Build per‑glyph 16×16 primitive ID grids + label_map + glyph splits.
#   9. phase2_attn    : (Phase 2 interpretability) Generate transformer attention heatmaps (requires Phase 2 checkpoint; skipped if absent).
#
# Each step is idempotent where possible (skips if expected outputs already exist)
# unless --force is provided.
#
# Usage:
#   ./run.sh                            # run full pipeline with defaults (Phase 1 + export_grids + phase2_attn if possible)
#   ./run.sh --limit 5000               # only rasterize first 5k glyphs
#   ./run.sh --no-train                 # stop after assignments (still can export_grids if requested explicitly)
#   ./run.sh --force                    # re-run all steps (overwrite)
#   ./run.sh --skip <step>              # skip a named step (repeatable)
#                                       # steps: rasterize, extract, kmeans, assign, stats, train, overlay, export_grids, phase2_attn
#   ./run.sh --only <step>              # run only the named step (ignores others & --no-train)
#                                       # valid: rasterize, extract, kmeans, assign, stats, train, overlay, export_grids, phase2_attn
#
#   ./run.sh --dry-run                  # print the commands without executing
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
PHASE2_CONFIG="configs/phase2.yaml"  # Needed for attention heatmap (model architecture)

RASTERS_DIR="data/rasters"
METADATA_FILE="${RASTERS_DIR}/metadata.jsonl"

CELLS_OUT_DIR="data/processed/cells"
CENTROIDS_OUT="assets/centroids/primitive_centroids.npy"
ASSIGNMENTS_FILE="data/processed/primitive_assignments.parquet"
PRIMITIVE_STATS="data/processed/primitive_stats.json"
PHASE2_OUT_ROOT="data/processed"
PHASE2_GRIDS_DIR="${PHASE2_OUT_ROOT}/grids"
PHASE2_LABEL_MAP="${PHASE2_OUT_ROOT}/label_map.json"
PHASE2_SPLITS_DIR="${PHASE2_OUT_ROOT}/splits"
PHASE2_ATTN_OUT="output/phase2_attn"

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

step_export_grids() {
  if safe_skip_steps && contains export_grids "${SKIP_STEPS[@]}"; then
    log "SKIP export_grids (user requested)"
    return
  fi
  # Require assignments & manifest (assignments still needed for hybrid reference even if using model mode)
  local manifest="${CELLS_OUT_DIR}/manifest.jsonl"
  [[ -f "$manifest" ]] || die "Manifest missing (expected $manifest). Run extract step first."
  if [[ ! -f "$ASSIGNMENTS_FILE" ]]; then
    if [[ "${GRID_MODE:-assignments}" == "assignments" ]]; then
      die "Assignments missing (expected $ASSIGNMENTS_FILE). Run assign step first."
    else
      log "WARN: assignments file missing but GRID_MODE=$GRID_MODE requested; hybrid mode will be downgraded to model."
    fi
  fi

  if [[ -f "$PHASE2_LABEL_MAP" && -d "$PHASE2_GRIDS_DIR" && $FORCE -eq 0 ]]; then
    log "Phase 2 grids + label_map already exist. Skipping (use --force to regenerate)."
    return
  fi

  # Decide mode: environment variable GRID_MODE can force {assignments|model|hybrid}
  # Default heuristic: if a Phase 1 checkpoint exists use model mode, else assignments.
  local MODE_ARG=""
  local CKPT_ARG=""
  local CENTROID_ARG="--centroid-file '$CENTROIDS_OUT'"
  local PHASE1_CKPT=""

  if ls checkpoints/phase1/*.pt >/dev/null 2>&1; then
    PHASE1_CKPT="$(ls -t checkpoints/phase1/*.pt | head -n1)"
  fi

  local DESIRED_MODE="${GRID_MODE:-}"
  if [[ -z "$DESIRED_MODE" ]]; then
    if [[ -n "$PHASE1_CKPT" ]]; then
      DESIRED_MODE="model"
    else
      DESIRED_MODE="assignments"
    fi
  fi

  if [[ "$DESIRED_MODE" == "model" || "$DESIRED_MODE" == "hybrid" ]]; then
    if [[ -z "$PHASE1_CKPT" ]]; then
      log "WARN: No Phase 1 checkpoint found; falling back to assignments mode."
      DESIRED_MODE="assignments"
    fi
  fi

  case "$DESIRED_MODE" in
    assignments)
      MODE_ARG="--mode assignments --assignments '$ASSIGNMENTS_FILE'"
      ;;
    model)
      MODE_ARG="--mode model --phase1-checkpoint '$PHASE1_CKPT'"
      # assignments file not strictly required here
      ;;
    hybrid)
      if [[ ! -f "$ASSIGNMENTS_FILE" ]]; then
        log "WARN: Hybrid requested but assignments missing; using model mode."
        MODE_ARG="--mode model --phase1-checkpoint '$PHASE1_CKPT'"
      else
        MODE_ARG="--mode hybrid --phase1-checkpoint '$PHASE1_CKPT' --assignments '$ASSIGNMENTS_FILE'"
      fi
      ;;
    *)
      die "Invalid GRID_MODE '$DESIRED_MODE' (expected assignments|model|hybrid)"
      ;;
  esac

  log "Exporting Phase 2 primitive ID grids (16x16) in mode=$DESIRED_MODE (ckpt=${PHASE1_CKPT:-none})..."

  run_cmd "python -m data.export_phase2_grids \
    --cells-dir '$CELLS_OUT_DIR' \
    --out-dir '$PHASE2_OUT_ROOT' \
    --chars-csv dataset/chars.csv \
    --infer-splits \
    --write-npy \
    $MODE_ARG \
    $CENTROID_ARG \
    -v"

  log "Phase 2 grid export complete (mode=$DESIRED_MODE)."
}

step_phase2_attn() {
  if safe_skip_steps && contains phase2_attn "${SKIP_STEPS[@]}"; then
    log "SKIP phase2_attn (user requested)"
    return
  fi
  # Preconditions
  if [[ ! -f "$PHASE2_CONFIG" ]]; then
    log "Phase 2 config not found ($PHASE2_CONFIG); skipping attention heatmaps."
    return
  fi
  if [[ ! -d "checkpoints/phase2" ]]; then
    log "No Phase 2 checkpoint directory (checkpoints/phase2); skipping attention heatmaps."
    return
  fi
  if [[ ! -f "$PHASE2_LABEL_MAP" || ! -d "$PHASE2_GRIDS_DIR" ]]; then
    log "Phase 2 grids or label_map missing; run export_grids first (skipping phase2_attn)."
    return
  fi
  mkdir -p "$PHASE2_ATTN_OUT"
  # Pick validation split if present; else sample from available grids.
  local SPLIT_FILE="$PHASE2_SPLITS_DIR/phase2_val_ids.txt"
  local SPLIT_ARG=""
  if [[ -f "$SPLIT_FILE" ]]; then
    SPLIT_ARG="--split-file $SPLIT_FILE --sample 16"
  else
    SPLIT_ARG="--sample 16"
  fi
  log "Generating Phase 2 attention heatmaps..."
  run_cmd "python -m eval.phase2_attention_heatmap \
    --config '$PHASE2_CONFIG' \
    --checkpoint-dir checkpoints/phase2 \
    --grids-dir '$PHASE2_GRIDS_DIR' \
    --label-map '$PHASE2_LABEL_MAP' \
    $SPLIT_ARG \
    --out-dir '$PHASE2_ATTN_OUT' \
    --per-layer \
    --global-scale \
    -v"
  log "Phase 2 attention heatmap generation complete (see $PHASE2_ATTN_OUT)."
}

############################
# Orchestrate              #
############################

step_overlay() {
  if safe_skip_steps && contains overlay "${SKIP_STEPS[@]}"; then
    log "SKIP overlay (user requested)"
    return
  fi
  # Only run if a checkpoint exists; otherwise warn and skip.
  local CKPT_DIR="checkpoints/phase1"
  if [[ ! -d "$CKPT_DIR" ]]; then
    log "Overlay skipped (no checkpoint directory $CKPT_DIR yet)."
    return
  fi
  # Use existing centroids + latest checkpoint to visualize coverage on unseen glyphs.
  local OUT_DIR="output/unseen_overlays"
  mkdir -p "$OUT_DIR"
  log "Generating unseen glyph overlay visualization..."
  run_cmd "python -m eval.phase1_unseen_overlay \
    --db dataset/glyphs.db \
    --raster-config configs/rasterizer.yaml \
    --centroids assets/centroids/primitive_centroids.npy \
    --checkpoint-dir checkpoints/phase1 \
    --out-dir $OUT_DIR \
    --panel-out $OUT_DIR/panel.png \
    --sample 20 \
    --alpha 255 \
    --seed 123"
  log "Overlay visualization complete (see $OUT_DIR)."
}

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
      rasterize)     step_rasterize ;;
      extract)       step_extract_cells ;;
      kmeans)        step_kmeans ;;
      overlay)       step_overlay ;;
      assign)        step_assign ;;
      stats)         step_stats ;;
      train)         step_train ;;
      export_grids)  step_export_grids ;;
      phase2_attn)   step_phase2_attn ;;
      *)
        die "Unknown --only step: $ONLY_STEP (valid: rasterize, extract, kmeans, assign, stats, train, overlay, export_grids, phase2_attn)"
        ;;
    esac
  else
    step_rasterize
    step_extract_cells
    step_kmeans
    step_assign
    step_stats
    step_train
    step_overlay
    step_export_grids
    step_phase2_attn
  fi
  log "=== Pipeline Complete ==="
}

main "$@"
