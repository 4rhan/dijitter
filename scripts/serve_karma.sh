#!/usr/bin/env bash
# Start the karma policy server on the GPU box (scripts/serve_karma.py), in the foreground; run it in tmux.
#
#   bash scripts/serve_karma.sh                                   # last checkpoint of the two-cube run
#   CKPT=outputs/train/pi05_two_cube_lora/checkpoints/015000/pretrained_model bash scripts/serve_karma.sh
#   MOCK=1 bash scripts/serve_karma.sh                            # no model: arms hold still (link test)
#   bash scripts/serve_karma.sh --save-frames outputs/karma_frames   # extra flags go to serve_karma.py
#
# The model needs about 8 GB of GPU memory and stops if less is free. The box is shared: while this
# runs, jobs that wait for a free GPU (Sahil's queue) stay paused, so stop it (Ctrl-C) when done.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f /usr/local/models/sra-arhan-dijitter/env.sh ] && source /usr/local/models/sra-arhan-dijitter/env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

if [ "${MOCK:-0}" = 1 ]; then
  exec python scripts/serve_karma.py --mock --port "${PORT:-8202}" "$@"
fi

CKPT=${CKPT:-outputs/train/pi05_two_cube_lora/checkpoints/last/pretrained_model}
[ -f "$CKPT/train_config.json" ] || { echo "no checkpoint at $CKPT"; exit 1; }
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt "${NEED_FREE_MB:-9000}" ]; then
  echo "only $free MB of GPU memory free, the server needs about 8 GB; what is running:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  exit 1
fi
echo "checkpoint: $(readlink -f "$CKPT")"
echo "this host: $(hostname -I | tr ' ' '\n' | grep -v '^$' | grep -vE '^(172\.17|192\.168\.122)\.' | tr '\n' ' ')"
exec python scripts/serve_karma.py --checkpoint "$CKPT" --port "${PORT:-8202}" "$@"
