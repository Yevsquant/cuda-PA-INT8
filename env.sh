# Source this before working:  source env.sh
# (assumes the conda env from setup.sh already exists)

export ENV_NAME="${ENV_NAME:-specdec}"

# A100-SXM4-80GB is compute capability 8.0. Building only sm_80 makes kernel
# compiles dramatically faster than the default all-architectures build.
export TORCH_CUDA_ARCH_LIST="8.0"

# Cap parallel nvcc on the 8 GiB pod: default ninja parallelism (= 24 CPUs)
# OOM-kills the pod when JIT-building our heavy cuda_ext kernels (torch's
# cpp_extension reads MAX_JOBS). Override to 1 for the very heaviest builds.
export MAX_JOBS="${MAX_JOBS:-2}"

# HF model cache. Home has ~1.7 PB free here, so keeping it under $HOME is fine.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Activate the env (works in non-interactive shells too).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# vLLM JIT-compiles flashinfer fp8 kernels when serving with --kv-cache-dtype fp8.
# The conda linker searches the conda sysroot (not the system driver dir), so the
# final link fails with "cannot find -lcuda". Point it at the CUDA driver stub
# shipped in the env; the real libcuda.so.1 is loaded at runtime.
export LIBRARY_PATH="${CONDA_PREFIX}/lib/stubs:${LIBRARY_PATH}"

echo "env ready: ${ENV_NAME} | arch sm_80 | HF_HOME=${HF_HOME}"
