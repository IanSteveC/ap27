# AP27 — CUDA port

A **CUDA** backend port of the PrimeGrid **AP26/AP27** GPU application
([mfl0p/ap27](https://github.com/mfl0p/ap27)), which searches for arithmetic
progressions of primes (the AP26 "10-shift" search). **Bit-exact** with the
original OpenCL app.

> Fork of **mfl0p/ap27**. The CUDA port lives in **[`cuda/`](cuda/)**; the
> upstream CPU and OpenCL apps (`cpu/`, `opencl/`) are unchanged.

## What's in this fork

- **CUDA backend** (`cuda/`): the 10 OpenCL kernels translated to CUDA `.cu`, and
  a header-only `simpleCU.h` that reimplements the app's `simpleCL` helper on the
  **CUDA Driver API** (so the host code barely changes). Two build modes:
  - **AOT (shipped):** each kernel compiled to a **multi-arch fatbin**
    (`sm_50…sm_90` + Blackwell `sm_100f/sm_120f`), embedded in the binary and
    loaded via the driver — **no NVRTC at runtime**, so clients need only the GPU
    driver (not the CUDA toolkit).
  - **NVRTC (dev):** compiles the embedded kernel sources at runtime (fast
    iteration / validation).

## Build

```sh
cd cuda
./build.sh --fatbin      # -> ap27_linux64_cuda        (driver-only, shipped)
./build.sh               # -> ap27_linux64_cuda_nvrtc  (dev, runtime NVRTC)
```

Requires CUDA 12.9 and BOINC. See **[`cuda/README.md`](cuda/README.md)** for
prerequisites and the design.

## Validation

Bit-exact against the bundled reference outputs (`cuda/test_*.txt`) — the same
solutions **and** the same full-search BOINC checksum, on a Tesla V100. Verified
for both the NVRTC and the AOT (driver-only) binaries.

## Performance (Tesla V100, vs the upstream OpenCL app)

- **1× throughput: parity** — CUDA is within ~2% of OpenCL, and the output is
  byte-identical. The AP26 sieve is compute/ALU-bound and both backends saturate
  the V100, so there is no headroom for CUDA to pull ahead.
- **Multiple tasks per GPU (CUDA MPS): no benefit** (~0.83–0.92× aggregate at
  2–3 concurrent tasks). Because AP26 already saturates the GPU solo, extra
  concurrency only adds contention — the opposite of small-GFN workloads.

See **[`cuda/README.md`](cuda/README.md)** for the port design, the fatbin
scheme, and full results.
