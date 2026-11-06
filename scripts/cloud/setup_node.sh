#!/usr/bin/env bash
# One-time, idempotent setup of a rented GPU node (givemeanode / SF Compute H100) for training.
# Everything lands under $HOME, which is the node's persistent disk: a stopped node keeps it all,
# and re-running this script skips every step that already finished (markers in $HOME/.dj_done).
#
#   HF_TOKEN=hf_... bash scripts/cloud/setup_node.sh
#
# HF_TOKEN must have access to google/paligemma-3b-pt-224 (gated); pi05 uses its tokenizer.
# EPISODES (default 100, same as the lab-box dataset) sets how many demos are converted.
# EASY=1 builds abc_easy instead (same filter as scripts/easy_task.sh, 200 demos by default).
# MID=1 builds abc_mid: the same four objects from both sides (450 demos, 234k frames).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/env.sh

if [ "${MID:-0}" = 1 ]; then
  EPISODES=${EPISODES:-450}
  DATASET=abc_mid
  FILTER=(--categories can rolling_pin pizza_cutter whisk --sides left right)
elif [ "${EASY:-0}" = 1 ]; then
  EPISODES=${EPISODES:-200}
  DATASET=abc_easy
  FILTER=(--categories can rolling_pin pizza_cutter whisk --sides left)
else
  EPISODES=${EPISODES:-100}
  DATASET=abc_inhand_transfer
  FILTER=()
fi
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
# The node image points uv at its own package mirror (UV_INDEX_URL / UV_DEFAULT_INDEX), which makes
# --locked fail: the lockfile names pypi.org. Same versions and hashes either way, so use the lockfile's.
venv() { env -u UV_INDEX_URL -u UV_DEFAULT_INDEX uv sync --extra gpu --locked; }

models() {
  : "${HF_TOKEN:?set HF_TOKEN (needs access to google/paligemma-3b-pt-224)}"
  hf download lerobot/pi05_base
  hf download google/paligemma-3b-pt-224 --include "*.json" --include "tokenizer.model"
  # Fail here, not mid-training, if the tokenizer files are incomplete.
  HF_HUB_OFFLINE=1 python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')"
}

# Episode data only. prepare.py --sim-data also installs the scene assets (7.5 GB, partly from a
# third-party host); training does not need them, the sim eval does.
abc_data() {
  (cd third_party/abc && python -c "
import sys
from prepare import PrepareConfig, prepare_sim_data
cfg = PrepareConfig(sim_data=('sim_inhand_transfer_the_item_to_other_side',))
sys.exit(prepare_sim_data(cfg, cfg.cache.expanduser().resolve()))
")
}

dataset() {
  rm -f "$DONE/dataset_rel_${DATASET}_"*  # a rebuilt dataset invalidates its derived one
  # LeRobot's writer is single-threaded Python (~30 s/episode), so convert shards in parallel.
  local workers=$(( $(nproc) / 3 ))
  workers=$(( workers < 1 ? 1 : workers > 16 ? 16 : workers ))
  python scripts/abc_to_lerobot.py --repo-id "$DATASET" "${FILTER[@]}" --max-episodes "$EPISODES" --workers "$workers" --overwrite
  python scripts/check_lerobot.py --repo-id "$DATASET" "${FILTER[@]}"
}

# Same dataset with action stats recomputed on 50-step chunks relative to the state (arm joints only,
# grippers stay absolute), for RELATIVE=1 training. Videos are copied, not re-encoded.
dataset_rel() {
  rm -rf "data/lerobot/${DATASET}_rel"
  lerobot-edit-dataset --repo_id "$DATASET" --root "data/lerobot/$DATASET" \
    --new_repo_id "${DATASET}_rel" --new_root "data/lerobot/${DATASET}_rel" \
    --operation.type recompute_stats --operation.relative_action true --operation.chunk_size 50 \
    "--operation.relative_exclude_joints=['gripper']"
}

# pi05_base loads, predicts, and ACG at scale 0 reproduces the baseline exactly (exits non-zero if not).
smoke() { HF_HUB_OFFLINE=1 python scripts/test_pi05.py --repo-id "$DATASET" --acg-scales 0.0 1.0 3.0; }

step uv install_uv
export PATH=$HOME/.local/bin:$PATH
step submodule submodule
step venv venv
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
step models models
step abc_data abc_data
step "dataset_${DATASET}_$EPISODES" dataset
step "dataset_rel_${DATASET}_$EPISODES" dataset_rel
step "smoke_$DATASET" smoke
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "video decoder for training: $(python -c 'from lerobot.utils.import_utils import get_safe_default_video_backend as f; print(f())' 2>/dev/null)"
echo "cpus: $(nproc)  ram: $(free -g | awk '/Mem/ {print $2}') GB  disk: $(df -h "$HOME" | awk 'END {print $4}') free"
echo "setup complete"
