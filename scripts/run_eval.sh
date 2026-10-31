#!/usr/bin/env bash
# Closed-loop sim eval of one checkpoint: plain pi0.5 first, then each ACG scale, same seeds (same worlds).
#
#   bash scripts/run_eval.sh                                   # baseline only, 50 episodes
#   SCALES="none 1.5 2 3" bash scripts/run_eval.sh             # baseline + ACG sweep
#   CKPT=outputs/train/pi05_inhand_lora/checkpoints/006000/pretrained_model N=10 bash scripts/run_eval.sh
#
# For each setting: start scripts/serve_pi05.py, wait for it to load, run scripts/eval_sim.py, stop the
# server. Results land in outputs/eval/<NAME>_<setting>/summary.json; the last lines compare them.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${CKPT:-outputs/train/pi05_inhand_lora/checkpoints/last/pretrained_model}
NAME=${NAME:-$(basename "$(dirname "$(readlink -f "$CKPT")")")}  # step number of the checkpoint
SCALES=${SCALES:-none}
N=${N:-50}
PORT=${PORT:-8000}
export HF_HUB_OFFLINE=1
mkdir -p outputs/eval

SERVER=""
cleanup() { [ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT

for s in $SCALES; do
  if [ "$s" = none ]; then tag=${NAME}_baseline; acg=(); else tag=${NAME}_acg$s; acg=(--acg-scale "$s"); fi
  if [ -f "outputs/eval/$tag/summary.json" ]; then echo "== $tag: already done"; continue; fi
  echo "== $tag: serving $CKPT ${acg[*]}"
  python scripts/serve_pi05.py --checkpoint "$CKPT" --port "$PORT" "${acg[@]}" > "outputs/eval/$tag.server.log" 2>&1 &
  SERVER=$!
  until grep -q "pi05 server" "outputs/eval/$tag.server.log"; do
    kill -0 "$SERVER" 2>/dev/null || { echo "server died:"; tail -20 "outputs/eval/$tag.server.log"; exit 1; }
    sleep 5
  done
  grep -E "warm-up|GPU memory" "outputs/eval/$tag.server.log"
  python scripts/eval_sim.py --port "$PORT" --tag "$tag" --num-episodes "$N"
  kill "$SERVER"; wait "$SERVER" 2>/dev/null || true; SERVER=""
done

python - "$NAME" <<'EOF'
import json, sys, pathlib
rows = sorted(pathlib.Path("outputs/eval").glob(f"{sys.argv[1]}_*/summary.json"))
print(f"\n{'run':<24}{'success':>9}{'latency ms':>12}{'JerkRMS':>10}{'ATV':>10}{'switch':>9}")
for p in rows:
    s = json.loads(p.read_text())
    print(f"{s['tag']:<24}{s['success_rate']:>9.0%}{s['latency_ms_mean']:>12.0f}{s['jerk_rms']:>10.1f}{s['atv']:>10.5f}{s['switch_jump']:>9.4f}")
EOF
