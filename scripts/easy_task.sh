#!/usr/bin/env bash
# The easier version of in-hand transfer: one direction (object starts on the left) and four objects
# that are thick enough to grasp without much precision. Training demos and eval worlds use the same
# filter, so success rates here are not comparable with the all-objects eval.
#
#   bash scripts/easy_task.sh prepare         # build the 200-demo dataset and check it (CPU, ~15 min)
#   bash scripts/easy_task.sh prepare_rel     # same dataset with relative-action stats, for RELATIVE=1
#   bash scripts/easy_task.sh train           # overnight LoRA run; re-run to resume
#   bash scripts/easy_task.sh overnight       # same, but first waits for the shared GPU (scripts/claim_gpu.sh)
#   bash scripts/easy_task.sh eval            # evaluate the last checkpoint on 50 easy worlds
#   CKPT=.../checkpoints/010000/pretrained_model bash scripts/easy_task.sh eval
#   CKPT=outputs/train/pi05_inhand_lora/checkpoints/008000/pretrained_model NAME=old8k REPO_ID=abc_inhand_transfer \
#     bash scripts/easy_task.sh eval          # the old checkpoint on the easy worlds: the "before" number
#
# Overrides: STEPS (20000), SAVE_FREQ (5000), BATCH, LORA_SCOPE, RELATIVE, OUT; N, SCALES, ACG_LAYERS for eval.
set -euo pipefail
cd "$(dirname "$0")/.."

CATEGORIES="can rolling_pin pizza_cutter whisk"
SIDES=left
EPISODES=${EPISODES:-200}
DATASET=abc_easy
PREFIX=easy
if [ "${MID:-0}" = 1 ]; then  # MID=1: the same four objects from both sides (450 demos)
  SIDES="left right"
  EPISODES=${EPISODES_MID:-450}
  DATASET=abc_mid
  PREFIX=mid
fi

case "${1:-}" in
  prepare)
    # shellcheck disable=SC2086  # CATEGORIES is a list
    python scripts/abc_to_lerobot.py --repo-id "$DATASET" --categories $CATEGORIES --sides $SIDES \
      --max-episodes "$EPISODES" --workers "${WORKERS:-8}" --overwrite
    # shellcheck disable=SC2086
    python scripts/check_lerobot.py --repo-id "$DATASET" --categories $CATEGORIES --sides $SIDES
    ;;
  prepare_rel)
    rm -rf "data/lerobot/${DATASET}_rel"
    lerobot-edit-dataset --repo_id "$DATASET" --root "data/lerobot/$DATASET" \
      --new_repo_id "${DATASET}_rel" --new_root "data/lerobot/${DATASET}_rel" \
      --operation.type recompute_stats --operation.relative_action true --operation.chunk_size 50 \
      "--operation.relative_exclude_joints=['gripper']"
    ;;
  train)
    export REPO_ID=$DATASET STEPS=${STEPS:-20000} SAVE_FREQ=${SAVE_FREQ:-5000}
    export OUT=${OUT:-outputs/train/pi05_easy_lora}
    bash scripts/train_lora.sh
    ;;
  overnight)
    exec bash scripts/claim_gpu.sh bash scripts/easy_task.sh train
    ;;
  eval)
    if [ "${RELATIVE:-0}" = 1 ]; then DATASET=${DATASET}_rel; fi
    export REPO_ID=${REPO_ID:-$DATASET}
    export CKPT=${CKPT:-${OUT:-outputs/train/pi05_easy_lora}/checkpoints/last/pretrained_model}
    export EVAL_ARGS="--categories $CATEGORIES --sides $SIDES ${EVAL_ARGS:-}"
    if [ -z "${NAME:-}" ]; then export NAME="${PREFIX}_$(basename "$(dirname "$(readlink -f "$CKPT")")")"; fi
    bash scripts/run_eval.sh
    ;;
  *)
    sed -n 2,16p "$0"
    exit 1
    ;;
esac
