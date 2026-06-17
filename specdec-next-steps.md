# specdec — next steps

Context: the CUDA extension now builds and is cached. The build OOM-kills (the
"disconnect") are fixed — `tests/cuda_ext.py` caps `MAX_JOBS` so parallel `nvcc`
no longer blows past this JupyterHub pod's 8 GiB cap. Run the steps below in your
terminal with the env activated.

## 0. Activate the env (every new shell)

```bash
conda activate specdec   # puts ninja + nvcc + torch on PATH
cd ~/cuda-PA-INT8
```

## 1. Run the tests (the .so is already built — no compile, <1 s to load)

```bash
pytest tests/test_int8_bridge.py -v -m "not slow"
```

Expected: all 31 selected tests run quickly. The first test loads the cached
`paged_attn_opt.so`; it does NOT recompile, so a disconnect here is harmless.

## 2. If you ever change a kernel (.cu) and it must rebuild

Only the changed kernel recompiles (ninja is incremental). It's already safe
under the default `MAX_JOBS=2`, but for a guaranteed no-OOM build force serial:

```bash
MAX_JOBS=1 python -c "import sys; sys.path.insert(0,'tests'); import cuda_ext; cuda_ext._opt_ext(); print('BUILD_OK')"
```

Do this BEFORE running pytest, so the slow/risky compile happens once on its own
rather than lazily inside the first test.

## 3. If a build ever dies mid-way and the next run seems to hang

The stale-lock guard in `cuda_ext.py` clears it automatically. To check/clear by
hand:

```bash
ls ~/.cache/torch_extensions/py312_cu128/paged_attn_opt/lock   # exists + no nvcc running = stale
rm -f ~/.cache/torch_extensions/py312_cu128/paged_attn_opt/lock
```

## Why this works (one line)

The pod is capped at 8 GiB; `nvcc` sees 24 CPUs and ninja used to launch ~24
parallel compiles → OOM-kill. `MAX_JOBS` keeps peak memory ~3–5 GiB. Once the
`.so` is cached, normal test runs just load it and disconnects stop mattering.
