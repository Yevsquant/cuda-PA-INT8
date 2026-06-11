#!/usr/bin/env bash
# Reproducible environment build for cuda-PA-INT8 on UIUC iCRN (A100-SXM4-80GB).
#
# IMPORTANT — driver constraint:
#   Node driver is 570.211.01 => supports CUDA runtime up to 12.8 (CUDA 12.x
#   only; CUDA 13 needs a >=580 driver). torch cu129 happens to run via 12.x
#   minor-version compat, but anything linking libcudart.so.13 will NOT load.
#   So we pin the ENTIRE stack to cu128.
#
# WHY vllm==0.19.1 (not latest):
#   vllm's PyPI wheel is CUDA-specific, NOT CUDA-agnostic. vllm >=0.20.0 ships a
#   CUDA-13 build (vllm/_C.abi3.so links libcudart.so.13) and cannot import on
#   this driver. vllm 0.19.1 is the last release whose _C links libcudart.so.12.
#   Likewise --torch-backend=auto picks cu129 here, so we force cu128 explicitly.
#
# Strategy:
#   - conda owns Python + CUDA toolkit (nvcc 12.8.2) needed to compile kernels.
#   - uv installs the pinned cu128 vllm + torch stack (verified before anything).
#
# Usage:   bash setup.sh
# Override: ENV_NAME=foo PY_VER=3.12 CUDA_VER=12.8.2 bash setup.sh
set -eo pipefail   # NOTE: no `-u` — conda's cuda-nvcc activate hook references
                   # NVCC_PREPEND_FLAGS; we pre-export it below to be safe.

ENV_NAME="${ENV_NAME:-specdec}"
PY_VER="${PY_VER:-3.12}"
CUDA_VER="${CUDA_VER:-12.8.2}"   # exact; must be <= driver max (12.8) and match cu128 torch
VLLM_VER="${VLLM_VER:-0.19.1}"   # last vllm whose _C links libcudart.so.12 (CUDA 12)
TORCH_VER="${TORCH_VER:-2.10.0}" # torch pinned by vllm 0.19.1

# Make conda's cuda-nvcc activation hook safe (it references these unbound).
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

echo ">>> conda base: $(conda info --base)"
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 1. base env -----------------------------------------------------------
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo ">>> env '${ENV_NAME}' already exists; reusing"
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi
conda activate "${ENV_NAME}"

# --- 2. vllm + cu128 torch (the critical path; verify before anything else) --
python -m pip install --upgrade pip uv
# --torch-backend=cu128 forces the cu128 PyTorch index; pinning torch alongside
# vllm keeps the whole stack on CUDA 12.8 (see header for why these versions).
uv pip install --torch-backend=cu128 \
  "vllm==${VLLM_VER}" \
  "torch==${TORCH_VER}" "torchvision==0.25.0" "torchaudio==${TORCH_VER}"
echo ">>> verifying GPU + vllm._C (catches the cu13 libcudart.so.13 trap)..."
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch        :", torch.__version__)
print("torch.cuda   :", torch.version.cuda)
print("cuda avail   :", ok)
if not ok:
    raise SystemExit("FATAL: torch cannot see the GPU — CUDA build vs driver mismatch")
if not torch.version.cuda.startswith("12.8"):
    raise SystemExit(f"FATAL: expected cu128 torch, got {torch.version.cuda}")
print("device       :", torch.cuda.get_device_name(0))
print("capability   :", torch.cuda.get_device_capability(0))
import vllm, vllm._C  # if this links libcudart.so.13 the driver can't load it
print("vllm._C      : OK (links libcudart.so.12)")
PY

# --- 3. CUDA toolkit (nvcc) pinned to 12.8.2 to match cu128 torch ------------
# strict-channel-priority + exact pin: prevents the solver from dragging in
# cuda-nvcc 13.0 from conda-forge.
conda install -y --strict-channel-priority -c nvidia -c conda-forge \
  "cuda-toolkit=${CUDA_VER}"

# --- 4. dev / benchmark deps -----------------------------------------------
uv pip install pytest ninja pandas matplotlib tabulate

# --- 5. final sanity -------------------------------------------------------
echo ">>> verification:"
nvcc --version | tail -2
python -c "import vllm; print('vllm         :', vllm.__version__)"

echo
echo ">>> done. Pin the env with:  bash freeze.sh"
echo ">>> activate with:           conda activate ${ENV_NAME}"
