#!/usr/bin/env bash
# LoRA fine-tune of LeRobot pi05_base on the converted ABC in-hand transfer dataset.
#
#   bash scripts/train_lora.sh                      # 10k steps -> outputs/train/pi05_inhand_lora
#   STEPS=20 OUT=outputs/train/dry bash scripts/train_lora.sh   # quick dry run
#
# Stock lerobot-train; everything is a CLI flag:
#   - LoRA on every attention + MLP projection of both the PaliGemma LM and the action expert, plus
#     the action/time projections (openpi's pi05 LoRA recipe). LeRobot's default only adapts the
#     expert's q/v (1.3M params); this is 27M. 12.3 GB GPU, ~1.9 s/step at batch 8 on a 4090.
#   - gradient checkpointing: batch 8 OOMs on 24 GB without it.
#   - rename_map: our camera keys -> the ones pi05_base was trained with (same as test_pi05.py).
#   - pi05_base is read from the shared HF cache and never modified; checkpoints are adapters only.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-10000}
OUT=${OUT:-outputs/train/pi05_inhand_lora}
REPO_ID=${REPO_ID:-abc_inhand_transfer}

export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --policy.path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=16 \
  "--peft.target_modules=(.*\.(language_model|gemma_expert)\..*\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|model\.(action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="data/lerobot/$REPO_ID" \
  --rename_map='{"observation.images.top_rgb": "observation.images.base_0_rgb", "observation.images.left_rgb": "observation.images.left_wrist_0_rgb", "observation.images.right_rgb": "observation.images.right_wrist_0_rgb"}' \
  --output_dir="$OUT" \
  --batch_size=8 \
  --steps="$STEPS" \
  --save_freq=2000 \
  --log_freq=50 \
  --wandb.enable=false
