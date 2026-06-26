# AP26/AP27 — CUDA backend

A CUDA port of the PrimeGrid AP26 GPU app (originally OpenCL, by Bryan Little with
contributions by Yves Gallot). It is **bit-exact** with the OpenCL/CPU reference —
same solutions and same full-search BOINC checksum.

## Design

The OpenCL app drives the GPU through a thin helper library, `simpleCL`. The port
keeps the host (`AP26.cpp`, `AP26.h`) almost unchanged and swaps that helper for a
CUDA one:

- **`simpleCU.h`** — header-only reimplementation of the `simpleCL` API on the
  **CUDA Driver API**: `sclMalloc`→`cuMemAlloc`, `sclWrite/Read`→`cuMemcpy*`,
  `sclEnqueueKernel`→`cuLaunchKernel`, `sclGetCLSoftware`→ load a kernel (see
  below), `ProfilesclEnqueueKernel`→ CUDA-event timing, etc.
  - BOINC's headers already define `cl_mem`/`cl_int`/`cl_event`, so the wrapper
    does **not** redefine them: a `cl_mem` is used as an opaque handle holding a
    `CUdeviceptr`, and GPU events use `CUevent`.
  - CUDA has no persistent kernel-arg state, so `sclSetKernelArg` caches arg blobs
    that are assembled for `cuLaunchKernel`.
- **`kernels/*.cu`** — the 10 OpenCL kernels translated to CUDA (mechanical, bit-
  exact: `__kernel`→`__global__`, `__local`→`__shared__`, `get_global_id`→
  `blockIdx*blockDim+threadIdx`, `barrier`→`__syncthreads`, `popcount`→`__popcll`,
  `clz`→`__clzll`, `atomic_inc`→`atomicAdd(p,1)`, etc.). The `.cl` originals are
  kept beside them for reference.
- Host device-init (`AP26.cpp`) uses CUDA: device chosen from BOINC's
  `aid.gpu_device_num`, info via `cuDeviceGet*`. The `cc>=7` / `cc<7` sieve
  selection (`sieve` vs `sieve_nv`) is preserved.

## Two kernel-loading paths

`sclGetCLSoftware()` selects at build time:

- **AOT / embedded fatbins** (`-DAP26_EMBED_FATBINS`, the shipped binary): each
  kernel is compiled offline by `nvcc` to a multi-arch fatbin and loaded with
  `cuModuleLoadData` — **no NVRTC at runtime**. The binary needs only the GPU
  driver, not the CUDA toolkit. `build_fatbins.sh` builds `sm_50…sm_90` directly
  and Blackwell `sm_100f/sm_120f` via `compute_89` PTX → `ptxas` (the genefer22
  scheme), `.incbin`-embeds them, and emits a source-pointer→fatbin registry.
- **NVRTC** (default dev build): compiles the embedded kernel source strings at
  runtime. Needs `libnvrtc` present.

## Build

```sh
./build.sh --fatbin      # AOT: ap27_linux64_cuda        (driver-only, shipped)
./build.sh               # NVRTC: ap27_linux64_cuda_nvrtc (dev)
```

Environment (defaults shown):

```sh
CUDA=/usr/local/cuda-12.9        # 13.x changes cuCtxCreate's signature; use 12.9
BOINC_DIR=/home/ian/builds/boinc
```

> The AOT binary links the **stub** `libcuda` and uses the client's real driver at
> runtime; `ldd` shows `libcuda.so.1` and **no** `libnvrtc`.

## Validate

```sh
./ap27_linux64_cuda 366382 366387 0
diff SOL-AP26.txt test_366382_366387_0.txt    # must match exactly
```

The bundled `test_KMIN_KMAX_SHIFT.txt` files are the upstream reference outputs;
a byte-for-byte match certifies the whole search (solutions + checksum) is
identical to the original app. All four cases pass on a Tesla V100 for both
builds.

## Performance (Tesla V100)

Measured with a difference method (large-range minus small-range wall-clock, so
one-time startup cancels) over identical K-work:

- **CUDA vs OpenCL, 1×:** ~parity. CUDA `0.98×` (−1.8%, within run-to-run noise);
  output byte-identical. The sieve is compute/ALU-bound (V100: 100% compute, ~1%
  memory) so both backends saturate the GPU.
- **Multiple tasks/GPU under CUDA MPS:** no benefit — aggregate throughput
  `0.92×` (2 tasks) / `0.83×` (3 tasks). A saturated, compute-bound workload gains
  nothing from concurrency; it only adds contention. (Small-GFN genefer workloads
  under-utilize the GPU and *do* gain from MPS; AP26 does not.)

## Notes

- The `.cl` kernel sources and the OpenCL `simpleCL.*` / `Makefile*` were copied in
  from `opencl/` as the porting base; the `.cl` files remain a useful reference for
  the translation.
- Blackwell SASS (`sm_100f/sm_120f`) is built but was not runtime-verified here (no
  Blackwell GPU available); it is correct-by-construction (clean `compute_89` PTX
  retargeted via `ptxas`, the genefer22 approach).
