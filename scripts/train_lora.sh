#!/usr/bin/env bash
# LoRA fine-tune of LeRobot pi05_base on the converted ABC in-hand transfer dataset.
#
#   bash scripts/train_lora.sh                      # 10k steps -> outputs/train/pi05_inhand_lora
#   STEPS=20 OUT=outputs/train/dry bash scripts/train_lora.sh   # quick dry run
#   RELATIVE=1 bash scripts/train_lora.sh           # arm joints as deltas to the state (openpi YAM recipe)
#
# Re-running with the same OUT resumes from its last checkpoint (also to extend a finished run with a
# larger STEPS). A run that died before its first checkpoint is moved aside and started fresh.
#
# Stock lerobot-train; everything is a CLI flag:
#   - LoRA on every attention + MLP projection of both the PaliGemma LM and the action expert, plus
#     the action/time projections (openpi's pi05 LoRA recipe). LeRobot's default only adapts the
#     expert's q/v (1.3M params); this is 27M. 12.3 GB GPU, ~1.9 s/step at batch 8 on a 4090.
#   - gradient checkpointing: batch 8 OOMs on 24 GB without it.
#   - rename_map: our camera keys -> the ones pi05_base was trained with (same as test_pi05.py).
#   - pi05_base is read from the shared HF cache and never modified; checkpoints are adapters only.
#   - RELATIVE=1: arm-joint actions relative to the current state, grippers absolute, like openpi's
#     pi05_yam_lora_inhand. Needs the <REPO_ID>_rel dataset (action stats recomputed on relative
#     chunks, see scripts/cloud/setup_node.sh); the checkpoint's processors convert back to absolute.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-10000}
OUT=${OUT:-outputs/train/pi05_inhand_lora}
RELATIVE=${RELATIVE:-0}
if [ "$RELATIVE" = 1 ]; then
  REPO_ID=${REPO_ID:-abc_inhand_transfer_rel}
else
  REPO_ID=${REPO_ID:-abc_inhand_transfer}
fi
BATCH=${BATCH:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
GRAD_CKPT=${GRAD_CKPT:-true}
SAVE_FREQ=${SAVE_FREQ:-2000}

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
  --policy.path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.gradient_checkpointing="$GRAD_CKPT" \
  --policy.use_relative_actions="$( [ "$RELATIVE" = 1 ] && echo true || echo false )" \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=16 \
  "--peft.target_modules=(.*\.(language_model|gemma_expert)\..*\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="data/lerobot/$REPO_ID" \
  --rename_map='{"observation.images.top_rgb": "observation.images.base_0_rgb", "observation.images.left_rgb": "observation.images.left_wrist_0_rgb", "observation.images.right_rgb": "observation.images.right_wrist_0_rgb"}' \
  --output_dir="$OUT" \
  --batch_size="$BATCH" \
  --num_workers="$NUM_WORKERS" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=50 \
  --wandb.enable=false
