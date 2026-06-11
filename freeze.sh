#!/usr/bin/env bash
# Snapshot the environment for reproducibility. Run after setup.sh succeeds.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-specdec}"

mkdir -p report
pip freeze | grep -iE 'vllm|^torch|xformers|flashinfer|transformers|numpy' > report/versions.txt
{
  echo "# environment snapshot"
  echo "host        : $(hostname)"
  echo "gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "driver      : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "nvcc        : $(nvcc --version | tail -1)"
  echo "python      : $(python --version)"
} >> report/versions.txt

conda env export --no-builds > environment.yml
echo ">>> wrote report/versions.txt and environment.yml"
cat report/versions.txt
