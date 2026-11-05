#!/usr/bin/env bash
# LoRA fine-tune of LeRobot pi05_base on the converted ABC in-hand transfer dataset.
#
#   bash scripts/train_lora.sh                      # 10k steps -> outputs/train/pi05_inhand_lora
#   STEPS=20 OUT=outputs/train/dry bash scripts/train_lora.sh   # quick dry run
#   RELATIVE=1 bash scripts/train_lora.sh           # arm joints as deltas to the state (openpi YAM recipe)
#   LORA_SCOPE=expert bash scripts/train_lora.sh    # LoRA on the action expert only (7M params, not 27M)
#
# Re-running with the same OUT resumes from its last checkpoint (also to extend a finished run with a
# larger STEPS). A run that died before its first checkpoint is moved aside and started fresh.
#
# Stock lerobot-train; everything is a CLI flag:
#   - LoRA on every attention + MLP projection of both the PaliGemma LM and the action expert, plus
#     the action/time projections (openpi's pi05 LoRA recipe). LeRobot's default only adapts the
#     expert's q/v (1.3M params); this is 27M. 12.3 GB GPU, ~1.9 s/step at batch 8 on a 4090.
#   - gradient checkpointing: batch 8 OOMs on 24 GB without it.
#   - LORA_SCOPE=expert leaves the PaliGemma LM frozen. Less to backpropagate, fewer params to adapt.
#   - the learning rate decays over the whole run (LeRobot's default stops at 30k and stays at the floor).
#   - rename_map: our camera keys -> the ones pi05_base was trained with (same as test_pi05.py).
#     RENAME_MAP replaces it for a dataset with other camera names, as JSON. Map every camera: LeRobot
#     can fill a camera it does not find with a blank image instead of failing.
#   - pi05_base is read from the shared HF cache and never modified; checkpoints are adapters only.
#     BASE picks the copy: by default a cached snapshot that is complete, since the cache's
#     refs/main can point at a snapshot holding only the weights, which fails offline.
#   - RELATIVE=1: arm-joint actions relative to the current state, grippers absolute, like openpi's
#     pi05_yam_lora_inhand. Needs the <REPO_ID>_rel dataset (action stats recomputed on relative
#     chunks, see scripts/cloud/setup_node.sh); the checkpoint's processors convert back to absolute.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-10000}
OUT=${OUT:-outputs/train/pi05_inhand_lora}
RELATIVE=${RELATIVE:-0}
REPO_ID=${REPO_ID:-abc_inhand_transfer}
[ "$RELATIVE" = 1 ] && REPO_ID=${REPO_ID}_rel
BATCH=${BATCH:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
GRAD_CKPT=${GRAD_CKPT:-true}
SAVE_FREQ=${SAVE_FREQ:-2000}
LORA_SCOPE=${LORA_SCOPE:-all}
SIM_RENAME_MAP='{"observation.images.top_rgb": "observation.images.base_0_rgb", "observation.images.left_rgb": "observation.images.left_wrist_0_rgb", "observation.images.right_rgb": "observation.images.right_wrist_0_rgb"}'
RENAME_MAP=${RENAME_MAP:-$SIM_RENAME_MAP}
case "$LORA_SCOPE" in
  all) LORA_BLOCKS="language_model|gemma_expert" ;;
  expert) LORA_BLOCKS="gemma_expert" ;;
  *) echo "LORA_SCOPE must be all or expert, not $LORA_SCOPE"; exit 1 ;;
esac
if [ -z "${BASE:-}" ]; then
  BASE=lerobot/pi05_base
  for snapshot in "${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"/models--lerobot--pi05_base/snapshots/*; do
    if [ -f "$snapshot/config.json" ] && [ -f "$snapshot/model.safetensors" ]; then BASE=$snapshot; fi
  done
fi

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -d "$OUT/checkpoints/$(printf %06d "$STEPS")" ]; then
  echo "$OUT already has its step-$STEPS checkpoint; nothing to do"
  exit 0
fi
LAST=$OUT/checkpoints/last/pretrained_model/train_config.json
if [ -f "$LAST" ]; then
  echo "resuming from $(readlink -f "$OUT/checkpoints/last")"
  exec lerobot-train --config_path="$LAST" --resume=true --steps="$STEPS" --save_freq="$SAVE_FREQ"
fi
if [ -d "$OUT" ]; then
  ASIDE=$OUT.failed_$(date +%s)
  echo "$OUT has no checkpoint; moving it to $ASIDE"
  mv "$OUT" "$ASIDE"
fi

lerobot-train \
  --policy.path="$BASE" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.gradient_checkpointing="$GRAD_CKPT" \
  --policy.use_relative_actions="$( [ "$RELATIVE" = 1 ] && echo true || echo false )" \
  --policy.scheduler_decay_steps="$STEPS" \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=16 \
  "--peft.target_modules=(.*\.($LORA_BLOCKS)\..*\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="data/lerobot/$REPO_ID" \
  --rename_map="$RENAME_MAP" \
  --output_dir="$OUT" \
  --batch_size="$BATCH" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=50 \
  --wandb.enable=false
