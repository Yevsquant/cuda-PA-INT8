# Source this before working:  source env.sh
# (assumes the conda env from setup.sh already exists)

export ENV_NAME="${ENV_NAME:-specdec}"

# A100-SXM4-80GB is compute capability 8.0. Building only sm_80 makes kernel
# compiles dramatically faster than the default all-architectures build.
export TORCH_CUDA_ARCH_LIST="8.0"

# HF model cache. Home has ~1.7 PB free here, so keeping it under $HOME is fine.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Activate the env (works in non-interactive shells too).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "env ready: ${ENV_NAME} | arch sm_80 | HF_HOME=${HF_HOME}"
