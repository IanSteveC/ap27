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

## Sieve optimization (the `sieve` kernel = 96% of GPU time)

Profiling (`ncu`) shows the sieve is **L1-cache gather-throughput-bound** (L1/TEX
90%, DRAM 0.5%, ~56% excessive sectors from the scattered `OKOK[(n59a+k·n59b)%p]`
lookups) — *not* DRAM- or compute-bound. The optimization:

- **Cache the hottest prime rows in shared memory.** The first ~26 primes (≤191,
  3198 entries ≈ 25 KB) are evaluated by every thread (the rest are early-out
  gated to ~13%/2%), so they dominate L1 traffic. They are staged into `__shared__`
  and read from there, offloading the L1 gather pressure.
- **Keep the early-out short-circuit.** This is what the upstream `sieve_nv` got
  wrong — it ANDs all cached primes in one expression (no early-out → ~37 gathers
  every time) and is *slower*. Keeping the per-group `if(sito &= …)` chain means
  most threads do only ~5–10 gathers.
- **Fill the SM's thread ceiling** (measured on V100 + 7 GeForce cards,
  Turing→Blackwell, all bit-exact): the winning launch geometry on *every* arch
  is the block size that exactly fills threads/SM, full cache, leftover memory
  to L1. The app picks it from the device's attributes at runtime:
  - 2048 thr/SM (Volta/A100/Hopper/dc-Blackwell) → **2×1024** — V100 **+10%**
  - 1536 thr/SM (consumer Ampere/Ada/Blackwell) → **2×768** — measured
    **+8…+12%** vs 1×1024 (3070 Ti +11.1%, 3090 +8.0%, 4070 Ti S +11.7%,
    5070 +11.0%, 5090 +6.6%)
  - 1024 thr/SM (Turing) → **1×1024** (already 100%; 768 fits only one block =
    75% and measures slower: 2080 Ti −2.7%, T600 −4.6%)
  The carveout then reserves exactly `blocks × 25.5 KB` of shared (driver
  rounds up to a tier; L1 gets the rest). Without forcing a value the driver
  picks inconsistently → random ~2× slowdowns. The choice is logged to stderr.

Result on a **real 124-K work unit** (`457248768 457248891 1280`, full WU,
bit-identical output): OpenCL **241 s**, first-cut CUDA **245 s**, **optimized
CUDA 220 s** — **+11% over the first cut, +10% over OpenCL**. Dead-ends (measured):
caching *all* 38 primes (`sieve_nv` style) is +13% *slower*; `__launch_bounds__`,
bigger caches past the cliff, and forced 96 KB carveout all regress.

### Experiment tunables (env, all bit-exact)

| var | values | effect |
|---|---|---|
| `AP26_CARVEOUT` | 0–100 | override the shared/L1 split (driver rounds up to a legal tier) |
| `AP26_BLOCK` | multiple of 32, ≤1024 | sieve block size; `768` → 2 blocks = 100% occupancy on 1536-thread/SM GPUs (consumer Ampere/Ada/Blackwell, where 1024-thread blocks cap at 67%) |
| `AP26_SIEVE` | `std` / `mid` / `ilp2` / `nv` | full 25.5 KB cache (default cc≥7) / reduced 11.3 KB cache (2×768 blocks fit the 32 KB shared tier keeping ~96 KB L1) / 2-words-per-thread ILP variant (latency hiding under an occupancy cap; ~2× slower on V100 due to an sm_70 register squeeze — for 1536-thr archs) / legacy cache-all |

`bench_sm86.sh` sweeps the interesting combinations on 1536-thread/SM GPUs and
bit-exact-checks every run; `build_fatbins.sh` enforces the per-arch register
budgets (sm_70 ≤32, sm_86/89 ≤42) so compiler drift can't silently fall off an
occupancy cliff.

## Performance vs OpenCL / MPS (Tesla V100)

- **CUDA vs OpenCL, 1×:** first-cut translation ~parity (`0.98×`); the **optimized
  sieve makes it +10% faster than OpenCL** (see above). Output byte-identical.
- **Multiple tasks/GPU under CUDA MPS:** a modest gain. Full grid (concurrency
  2–4 × MPS active-thread-% 30–100), aggregate-throughput speedup vs 1 task,
  identical per-task workload, startup-corrected:

  | C \ MPS% | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
  |---|---|---|---|---|---|---|---|---|
  | 2 | 0.62 | 0.83 | 1.01 | 1.01 | 1.01 | 1.01 | 1.01 | 1.00 |
  | 3 | 0.93 | 1.01 | 1.02 | 1.04 | 1.06 | 1.06 | 1.07 | 1.01 |
  | 4 | 1.00 | 1.03 | 1.02 | 1.06 | **1.12** | 1.07 | 1.07 | 1.07 |

  Best **+12% at 4 tasks @ 70%**; ~+7% typical at 3–4 tasks / 60–90%; 2 tasks
  flat; low MPS% caps *hurt* (each task starved of SMs). nvidia-smi's solo
  "100% utilization" hides latency gaps that 3–4 concurrent tasks fill — the
  effect is real but small (AP26 is near-saturated; small-GFN genefer reaches
  ~2.8× because it under-utilizes the GPU). Reproduce with `ap26-mps-sweep.sh`.

## Notes

- The `.cl` kernel sources and the OpenCL `simpleCL.*` / `Makefile*` were copied in
  from `opencl/` as the porting base; the `.cl` files remain a useful reference for
  the translation.
- Blackwell SASS (`sm_100f/sm_120f`) is built but was not runtime-verified here (no
  Blackwell GPU available); it is correct-by-construction (clean `compute_89` PTX
  retargeted via `ptxas`, the genefer22 approach).
