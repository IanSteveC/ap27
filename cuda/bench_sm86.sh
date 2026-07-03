#!/bin/bash
#
# Occupancy/L1 sweep for the AP26 sieve on 1536-thread/SM GPUs
# (consumer Ampere sm_86, Ada sm_89, consumer Blackwell sm_120).
#
# Background: the V100's +10% came from running 2 resident blocks/SM (100%
# occupancy). With 1024-thread blocks, a 1536-thread SM is stuck at 67%
# occupancy (so is the OpenCL app - it also uses 1024). Two levers, both
# bit-exact, exposed via env:
#   AP26_BLOCK=768|512   - smaller blocks so 2-3 fit the thread budget
#   AP26_SIEVE=mid       - reduced 11.3KB shared cache (primes <=131) so two
#                          blocks fit the 32KB shared tier, keeping ~96KB L1
#   AP26_CARVEOUT=N      - shared/L1 split (driver rounds up to a legal tier)
#
# Sweep matrix on a 128KB-unified / 100KB-max-shared / 1536-thr SM:
#   A std  1024 cv0  -> 1 blk, 32KB sh, 96KB L1, 67% occ   (current default = OpenCL parity)
#   B std   768 cv50 -> 2 blk, 64KB sh, 64KB L1, 100% occ  (occupancy, pay L1)
#   C mid   768 cv23 -> 2 blk, 32KB sh, 96KB L1, 100% occ  (occupancy AND L1)
#   D mid  1024 cv0  -> 1 blk, 16KB sh, 112KB L1, 67% occ  (max L1, small cache)
#   E mid   512 cv34 -> 3 blk, 64KB sh, 64KB L1, 100% occ  (3-block variant)
#   F std   768 cv0  -> 1 blk, 32KB sh, 96KB L1, 50% occ   (control: should lose to A)
#   G ilp2 1024 cv0  -> 1 blk, 67% occ + 2 words/thread    (ILP instead of occupancy;
#                       40 regs on sm_86, SASS-verified 10 batched shared loads)
#   H ilp2  768 cv50 -> 2 blk, 100% occ + 2 words/thread   (both levers stacked)
# NOTE: ilp2 on the V100 is ~2x SLOWER - an sm_70 codegen artifact (ptxas squeezes
# it to 32 regs, serializing the dual chains; ncu shows 2 blocks resident but issue
# 71%->46%). sm_86 compiles to 40 regs with healthy SASS - measure, don't assume.
#
# Usage:  ./bench_sm86.sh [path-to-binary]     (default ./ap27_linux64_cuda)
# Runs each config interleaved x ROUNDS on a fixed workload, bit-exact-checks
# every run, prints a ranked table. Strictly sequential; waits for GPU idle.
#
set -u
cd "$(dirname "$0")"

BIN="${1:-./ap27_linux64_cuda}"
REF="test_366382_366387_0.txt"
ARGS="366382 366387 0"
ROUNDS="${ROUNDS:-3}"

[ -x "$BIN" ] || { echo "binary not found: $BIN (build with ./build.sh --fatbin or pass a path)"; exit 1; }
[ -f "$REF" ] || { echo "reference file $REF missing (run from the cuda/ dir)"; exit 1; }

CONFIGS=(
  "A|baseline_1024_maxL1|AP26_CARVEOUT=0"
  "B|std_768_2blk_64L1|AP26_BLOCK=768 AP26_CARVEOUT=50"
  "C|mid_768_2blk_96L1|AP26_SIEVE=mid AP26_BLOCK=768 AP26_CARVEOUT=23"
  "D|mid_1024_112L1|AP26_SIEVE=mid AP26_CARVEOUT=0"
  "E|mid_512_3blk_64L1|AP26_SIEVE=mid AP26_BLOCK=512 AP26_CARVEOUT=34"
  "F|std_768_1blk_ctrl|AP26_BLOCK=768 AP26_CARVEOUT=0"
  "G|ilp2_1024_67occ|AP26_SIEVE=ilp2 AP26_CARVEOUT=0"
  "H|ilp2_768_2blk_100occ|AP26_SIEVE=ilp2 AP26_BLOCK=768 AP26_CARVEOUT=50"
)

idle(){ while [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' %' | head -1)" -gt 2 ]; do sleep 1; done; }

# --- device probe: settle threads/SM, shared/SM, regs/SM ------------------
CUDA="${CUDA:-/usr/local/cuda-12.9}"
if [ -x "$CUDA/bin/nvcc" ] || [ -d "$CUDA/include" ]; then
  cat > /tmp/ap26_devq.c <<'EOF'
#include <cuda.h>
#include <stdio.h>
int main(){ CUdevice d; int v; char n[128];
  if(cuInit(0)) return 1; cuDeviceGet(&d,0); cuDeviceGetName(n,127,d);
  printf("device: %s\n", n);
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d); printf("  cc major: %d\n", v);
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d); printf("  cc minor: %d\n", v);
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, d);   printf("  threads/SM: %d\n", v);
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, d); printf("  shared/SM: %d B\n", v);
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, d); printf("  regs/SM: %d\n", v);
  cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d); printf("  SMs: %d\n", v);
  return 0; }
EOF
  if gcc /tmp/ap26_devq.c -I"$CUDA/include" -lcuda -o /tmp/ap26_devq 2>/dev/null; then
    echo "=== device attributes (settles the 1536-vs-2048 thr/SM question) ==="
    /tmp/ap26_devq
    echo ""
  fi
fi

# --- sweep -----------------------------------------------------------------
declare -A SUM CNT OK
echo "=== sweep: ${ROUNDS} interleaved rounds x ${#CONFIGS[@]} configs, workload: $ARGS ==="
for round in $(seq 1 "$ROUNDS"); do
  for c in "${CONFIGS[@]}"; do
    id=${c%%|*}; rest=${c#*|}; name=${rest%%|*}; envs=${rest#*|}
    d="bench_$id"; rm -rf "$d"; mkdir -p "$d"; cp "$BIN" "$d/app"
    idle
    ( cd "$d"
      rm -f AP26-state*.txt SOL-AP26.txt stderr.txt
      s=$(date +%s.%N); env $envs ./app $ARGS >/dev/null 2>&1; e=$(date +%s.%N)
      echo "$e - $s" | bc > wall.txt
      diff -q SOL-AP26.txt "../$REF" >/dev/null 2>&1 && echo MATCH > ok.txt || echo DIFFER > ok.txt
    )
    t=$(cat "$d/wall.txt"); k=$(cat "$d/ok.txt")
    SUM[$id]=$(echo "${SUM[$id]:-0} + $t" | bc); CNT[$id]=$(( ${CNT[$id]:-0} + 1 )); OK[$id]="${OK[$id]:-MATCH}"
    [ "$k" = MATCH ] || OK[$id]=DIFFER
    printf "  r%d %s(%s) %.2fs %s\n" "$round" "$id" "$name" "$t" "$k"
  done
done

echo ""
echo "=== results (mean of $ROUNDS, vs baseline A) ==="
BASE=$(echo "${SUM[A]} / ${CNT[A]}" | bc -l)
for c in "${CONFIGS[@]}"; do
  id=${c%%|*}; rest=${c#*|}; name=${rest%%|*}
  m=$(echo "${SUM[$id]} / ${CNT[$id]}" | bc -l)
  printf "  %s %-22s %7.2fs  %+6.1f%%  bit-exact:%s\n" "$id" "$name" "$m" \
    "$(echo "($BASE / $m - 1) * 100" | bc -l)" "${OK[$id]}"
done
echo ""
echo "(positive % = faster than the current default A; DIFFER on any config = report it, do not use)"
