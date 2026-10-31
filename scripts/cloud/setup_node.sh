#!/usr/bin/env bash
# One-time, idempotent setup of a rented GPU node (givemeanode / SF Compute H100) for training.
# Everything lands under $HOME, which is the node's persistent disk: a stopped node keeps it all,
# and re-running this script skips every step that already finished (markers in $HOME/.dj_done).
#
#   HF_TOKEN=hf_... bash scripts/cloud/setup_node.sh
#
# HF_TOKEN must have access to google/paligemma-3b-pt-224 (gated); pi05 uses its tokenizer.
# EPISODES (default 100, same as the lab-box dataset) sets how many demos are converted.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/env.sh

EPISODES=${EPISODES:-100}
DONE=$HOME/.dj_done
mkdir -p "$DONE"

step() {  # step <name> <command...>: run once, then remember it
  local name=$1; shift
  if [ -f "$DONE/$name" ]; then echo "== $name: already done"; return; fi
  echo "== $name"
  local t0=$SECONDS
  "$@"
  touch "$DONE/$name"
  echo "== $name: done in $((SECONDS - t0)) s"
}

install_uv() { command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; }
submodule() { git submodule update --init third_party/abc; }
venv() { uv sync --extra gpu --locked; }

models() {
  : "${HF_TOKEN:?set HF_TOKEN (needs access to google/paligemma-3b-pt-224)}"
  hf download lerobot/pi05_base
  hf download google/paligemma-3b-pt-224 --include "tokenizer*" "special_tokens_map.json" "config.json"
}

abc_data() { (cd third_party/abc && python prepare.py --sim-data sim_inhand_transfer_the_item_to_other_side); }

dataset() {
  python scripts/abc_to_lerobot.py --repo-id abc_inhand_transfer --max-episodes "$EPISODES" --overwrite
  python scripts/check_lerobot.py --repo-id abc_inhand_transfer
}

# pi05_base loads, predicts, and ACG at scale 1.0 reproduces the baseline exactly.
smoke() { HF_HUB_OFFLINE=1 python scripts/test_pi05.py --acg-scales 1.0 2.0 3.0; }

step uv install_uv
export PATH=$HOME/.local/bin:$PATH
step submodule submodule
step venv venv
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
step models models
step abc_data abc_data
step "dataset_$EPISODES" dataset
step smoke smoke
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "$HOME" | tail -1
echo "setup complete"
