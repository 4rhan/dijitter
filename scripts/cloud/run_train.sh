#!/usr/bin/env bash
# Unattended run on a rented node: setup (skips finished steps), LoRA training (resumes from the last
# checkpoint), then the node stops itself so the meter stops - also when anything above fails.
#
#   HF_TOKEN=hf_... bash scripts/cloud/run_train.sh
#   STEPS=50 NO_STOP=1 bash scripts/cloud/run_train.sh     # dry run, keep the node up afterwards
#
#   EASY=1 bash scripts/cloud/run_train.sh   # abc_easy (see easy_task.sh) -> outputs/train/pi05_easy_h100
#   MID=1 bash scripts/cloud/run_train.sh    # abc_mid, easy objects from both sides -> outputs/train/pi05_mid_h100
#
# HF_UPLOAD=<user>/<repo> also uploads the packed checkpoints to that (private) Hugging Face model repo
# before the node stops, so they can be downloaded without waking the node. HF_TOKEN needs write access.
#
# Use a separate OUT for a dry run: resuming keeps the saved config, so a real run that continued a
# 50-step test would keep its 50-step LR decay. A run ends by packing its checkpoints into
# $HOME/export/<run>.tar (scripts/cloud/bundle_ckpts.sh).
#
# Every run is logged to $HOME/logs/run_<time>.log. Training env knobs (STEPS, BATCH, GRAD_CKPT,
# SAVE_FREQ, NUM_WORKERS, RELATIVE, OUT) are passed through to scripts/train_lora.sh.
#
# Launch on a fresh or stopped node:
#   cd ~ && (git -C dijitter pull --ff-only || git clone https://github.com/4rhan/dijitter.git) \
#     && bash dijitter/scripts/cloud/run_train.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p "$HOME/logs"
LOG=$HOME/logs/run_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

stop_node() {
  echo "== exit code $1; $( [ "${NO_STOP:-0}" = 1 ] && echo "NO_STOP=1, leaving the node up" || echo "stopping the node")"
  [ "${NO_STOP:-0}" = 1 ] && return
  sync
  for _ in 1 2 3; do
    curl -sf -m 20 -X POST -H 'Metadata-Flavor: givemeanode' http://169.254.42.1/v1/stop && return
    sleep 5
  done
}
trap 'stop_node $?' EXIT

# (No git pull in here: bash reads a script as it runs, so updating this file mid-run breaks it.
# The launch command pulls first.)
# FFmpeg shared libraries let torchcodec decode the training videos (~4.5x faster than the pyav
# fallback). System packages do not survive a stop, so this runs every time; failure is not fatal.
command -v ffmpeg >/dev/null || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg; } \
  || echo "ffmpeg install failed; video decoding will fall back to pyav"
bash scripts/cloud/setup_node.sh || exit 1
source scripts/cloud/env.sh
if [ "${MID:-0}" = 1 ]; then
  export REPO_ID=abc_mid
  export OUT=${OUT:-outputs/train/pi05_mid_h100}
elif [ "${EASY:-0}" = 1 ]; then
  export REPO_ID=abc_easy
  export OUT=${OUT:-outputs/train/pi05_easy_h100}
else
  export OUT=${OUT:-outputs/train/pi05_inhand_lora}
fi
# An H100 has 80 GB: no gradient checkpointing needed, and checkpoints often in case the host drops.
GRAD_CKPT=${GRAD_CKPT:-false} SAVE_FREQ=${SAVE_FREQ:-1000} NUM_WORKERS=${NUM_WORKERS:-$(( $(nproc) > 16 ? 12 : $(nproc) / 2 + 1 ))} \
  bash scripts/train_lora.sh
rc=$?
if [ "${RELATIVE:-0}" = 1 ] && [ -e "$OUT/checkpoints/last" ]; then
  python scripts/cloud/check_relative.py "$OUT/checkpoints/last/pretrained_model" || rc=1
fi
bash scripts/cloud/bundle_ckpts.sh "$OUT" || echo "bundling failed; the checkpoints are still in $OUT"
TAR=$HOME/export/$(basename "$OUT").tar
if [ -n "${HF_UPLOAD:-}" ] && [ -f "$TAR" ]; then
  # Before the node stops: a stopped node has to be woken (a queue) to download from it.
  hf upload "$HF_UPLOAD" "$TAR" "$(basename "$TAR")" --private --commit-message "$(basename "$OUT") checkpoints" \
    && echo "uploaded $TAR to https://huggingface.co/$HF_UPLOAD" \
    || echo "upload to $HF_UPLOAD failed; the checkpoints are still in $TAR"
fi
exit "$rc"
