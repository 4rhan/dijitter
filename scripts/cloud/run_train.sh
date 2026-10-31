#!/usr/bin/env bash
# Unattended run on a rented node: setup (skips finished steps), LoRA training (resumes from the last
# checkpoint), then the node stops itself so the meter stops - also when anything above fails.
#
#   HF_TOKEN=hf_... bash scripts/cloud/run_train.sh
#   STEPS=50 NO_STOP=1 bash scripts/cloud/run_train.sh     # dry run, keep the node up afterwards
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
# An H100 has 80 GB: no gradient checkpointing needed, and checkpoints often in case the host drops.
GRAD_CKPT=${GRAD_CKPT:-false} SAVE_FREQ=${SAVE_FREQ:-1000} NUM_WORKERS=${NUM_WORKERS:-$(( $(nproc) > 16 ? 12 : $(nproc) / 2 + 1 ))} \
  bash scripts/train_lora.sh
