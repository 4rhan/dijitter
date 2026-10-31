#!/usr/bin/env bash
# Unattended run on a rented node: setup (skips finished steps), LoRA training (resumes from the last
# checkpoint), then the node stops itself so the meter stops - also when anything above fails.
#
#   HF_TOKEN=hf_... bash scripts/cloud/run_train.sh
#   STEPS=50 NO_STOP=1 bash scripts/cloud/run_train.sh     # dry run, keep the node up afterwards
#
# Every run is logged to $HOME/logs/run_<time>.log. Training env knobs (STEPS, BATCH, GRAD_CKPT,
# SAVE_FREQ, OUT) are passed through to scripts/train_lora.sh.
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
    curl -s -m 20 -X POST -H 'Metadata-Flavor: givemeanode' http://169.254.42.1/v1/stop && return
    sleep 5
  done
}
trap 'stop_node $?' EXIT

git pull --ff-only || echo "git pull failed, continuing with the local checkout"
bash scripts/cloud/setup_node.sh || exit 1
source scripts/cloud/env.sh
# An H100 has 80 GB: no gradient checkpointing needed, and checkpoints often in case the host drops.
GRAD_CKPT=${GRAD_CKPT:-false} SAVE_FREQ=${SAVE_FREQ:-1000} bash scripts/train_lora.sh
