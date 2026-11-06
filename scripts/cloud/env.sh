# Environment for a rented GPU node: caches, venv and data all on the persistent $HOME disk.
#   source scripts/cloud/env.sh
export UV_CACHE_DIR=$HOME/.cache/uv
export UV_PROJECT_ENVIRONMENT=$HOME/venv
export HF_HOME=$HOME/.cache/huggingface
export HF_LEROBOT_HOME=$HOME/dijitter/data/lerobot
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# The node image puts its own CUDA 12.9 libraries on LD_LIBRARY_PATH. torch (cu128 wheels) then loads the
# image's libcublasLt next to its own libcublas, and every bf16 matmul fails with CUBLAS_STATUS_INVALID_VALUE.
unset LD_LIBRARY_PATH
export PATH=$HOME/.local/bin:$PATH
if [ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]; then source "$UV_PROJECT_ENVIRONMENT/bin/activate"; fi
