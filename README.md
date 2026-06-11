# cuda-PA-INT8

CUDA PagedAttention decode kernel with INT8 KV-cache quantization, built from
scratch, plus vLLM speculative-decoding benchmarks. See [CLAUDE.md](CLAUDE.md)
for the full technical spec.

## Environment

Built and run on **UIUC iCRN**: 1× NVIDIA A100-SXM4-80GB (sm_80), driver
570.211.01 (CUDA 12.8 ceiling), conda 25.x. No Slurm/module system on this node —
the GPU is directly visible.

> The original spec (CLAUDE.md) targeted NCSA Delta with `uv venv`. The actual
> build uses **conda** because the custom kernels need `nvcc`, and conda's
> `cuda-toolkit` is the cleanest no-root way to get a toolchain matching torch.

### Setup

```bash
bash setup.sh        # creates conda env 'specdec', installs vllm + matching nvcc
bash freeze.sh       # pins versions -> report/versions.txt + environment.yml
```

`setup.sh` installs vLLM first (so it pins its own torch), reads
`torch.version.cuda`, then installs the matching `cuda-toolkit` for `nvcc`.

### Daily use

```bash
source env.sh                        # activates env, sets sm_80 arch + HF_HOME
python benchmarks/smoke_vllm.py      # Day-0 smoke test (Qwen2.5-1.5B-Instruct)
pytest tests/                        # correctness tests vs PyTorch reference
```

## Layout

```
kernels/      # CUDA kernel sources (.cu/.cuh)
tests/        # pytest — correctness vs PyTorch reference
benchmarks/   # kernel microbenchmarks + vLLM serving benchmarks
report/       # versions.txt, benchmark report
```
