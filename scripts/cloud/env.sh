# Environment for a rented GPU node: caches, venv and data all on the persistent $HOME disk.
#   source scripts/cloud/env.sh
export UV_CACHE_DIR=$HOME/.cache/uv
export UV_PROJECT_ENVIRONMENT=$HOME/venv
export HF_HOME=$HOME/.cache/huggingface
export HF_LEROBOT_HOME=$HOME/dijitter/data/lerobot
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PATH=$HOME/.local/bin:$PATH
if [ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then source "$UV_PROJECT_ENVIRONMENT/bin/activate"; fi
