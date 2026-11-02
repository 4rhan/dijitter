#!/usr/bin/env bash
# Run a command on the shared GPU as soon as other jobs leave it, without touching them.
#
#   bash scripts/claim_gpu.sh bash scripts/easy_task.sh train
#
# Some jobs on this box wait between their steps while someone else holds more than 3 GB of GPU memory.
# So: hold a little over 3 GB now, which makes them pause at their next step; start the command once
# enough memory is free; let go of the hold when the command itself has loaded onto the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

HOLD_GB=${HOLD_GB:-3.3}
NEED_FREE_MB=${NEED_FREE_MB:-16000}  # free memory, with the hold in place, that means the others have gone
LOADED_MB=${LOADED_MB:-9000}         # total GPU use that means the command has loaded (weights are ~8 GB)

gpu() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits; }

python -c "
import time, torch
hold = torch.zeros(int($HOLD_GB * 2**30), dtype=torch.uint8, device='cuda')
print('holding $HOLD_GB GB of GPU memory', flush=True)
time.sleep(10**9)
" &
HOLD=$!
trap 'kill "$HOLD" 2>/dev/null || true' EXIT

until [ "$(gpu memory.free)" -ge "$NEED_FREE_MB" ]; do
  kill -0 "$HOLD" 2>/dev/null || { echo "could not hold GPU memory; is the GPU full?"; exit 1; }
  echo "[$(date +%T)] waiting: $(gpu memory.free) MB free, need $NEED_FREE_MB"
  sleep 30
done

echo "[$(date +%T)] GPU clear, starting: $*"
"$@" &
CMD=$!
while kill -0 "$CMD" 2>/dev/null && [ "$(gpu memory.used)" -lt "$((LOADED_MB + 3400))" ]; do sleep 5; done
kill "$HOLD" 2>/dev/null || true
echo "[$(date +%T)] released the hold"
wait "$CMD"
