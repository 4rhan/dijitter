#!/usr/bin/env bash
# Pack a run's checkpoints for download: $HOME/export/<run>.tar, without the optimizer state, which
# only resuming needs and which stays on the node's disk. Written to a temporary name first, so an
# export can never pick up a half-written file.
#
#   bash scripts/cloud/bundle_ckpts.sh outputs/train/pi05_easy_h100
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?usage: bundle_ckpts.sh <run dir>}
[ -d "$OUT/checkpoints" ] || { echo "no checkpoints in $OUT, nothing to bundle"; exit 0; }
mkdir -p "$HOME/export"
TAR=$HOME/export/$(basename "$OUT").tar
tar -C "$OUT" --exclude=training_state --exclude=checkpoints/last -cf "$TAR.tmp" checkpoints
mv "$TAR.tmp" "$TAR"
echo "bundled $(ls "$OUT/checkpoints" | grep -cv '^last$') checkpoints: $TAR ($(du -h "$TAR" | cut -f1))"
