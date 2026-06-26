#!/bin/bash
#
# AP26 (CUDA) MPS concurrency sweep.
#
# For a fixed per-task workload, measures aggregate throughput when running C
# tasks per GPU under CUDA MPS at a capped SM percentage, across the full grid of
# (concurrency x MPS active-thread-%). Reports EVERY cell, not just the best.
#
# Metric (startup-corrected, like the genefer sweep):
#   * one-time solo run pair gives  w = per-K work time  and  S = startup, via a
#     difference of two solo wall-clocks (cancels GPU init / module load).
#   * solo work time for the task workload:  Wsolo = Tsolo - S
#   * a config's per-task startup-free time:  t_i = wall_i - S
#   * effective time per work-unit:           eff  = mean(t_i) / C
#   * aggregate speedup vs 1 task:            spd  = Wsolo / eff = C*Wsolo/mean(t_i)
#   spd = 1.0 -> no gain ; spd = C -> perfect scaling ; spd < 1 -> MPS hurts.
#
# Strictly sequential: ONE (C,P) config at a time, GPU idle-gated between configs,
# all tasks waited to completion (no pileup), MPS torn down on exit.
#
set -u

# ===================== configurable =====================
BINARY="${BINARY:-/home/ian/builds/primegrid/ap27/cuda/ap27_linux64_cuda}"
G_FULL="${G_FULL:-366382 366387 0}"     # per-task workload (KMIN KMAX SHIFT)
G_HALF="${G_HALF:-366382 366384 0}"     # ~half of G_FULL, same start (for w,S)
CONCURRENCY=(2 3 4)
MPS_PCT=(30 40 50 60 70 80 90 100)
WORKDIR="${WORKDIR:-/tmp/ap26_mps_sweep}"
OUTFILE="${OUTFILE:-$WORKDIR/results.csv}"
CFG_TIMEOUT="${CFG_TIMEOUT:-300}"       # per-task timeout (s)
IDLE_MAX=2                              # %GPU considered idle
# ========================================================

MPSDIR="$WORKDIR/mps"
export CUDA_MPS_PIPE_DIRECTORY="$MPSDIR/pipe"
export CUDA_MPS_LOG_DIRECTORY="$MPSDIR/log"

mps_up=0
log(){ echo "$@"; }
# count live worker processes by exact comm name (NOT pgrep -f, which matches wrappers)
nkount2(){ ps -eo comm 2>/dev/null | grep -cx "$(basename "$BINARY")"; }
# block until the GPU is idle AND no worker is still running
wait_for_idle(){ while [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader|tr -d ' %'|head -1)" -gt "$IDLE_MAX" ] || [ "$(nkount2)" -gt 0 ]; do sleep 1; done; }

stop_mps(){
  if [ "$mps_up" = 1 ]; then echo quit | nvidia-cuda-mps-control 2>/dev/null; mps_up=0; fi
  pkill -f nvidia-cuda-mps 2>/dev/null
  sleep 1
}
cleanup(){ pkill -x "$(basename "$BINARY")" 2>/dev/null; stop_mps; }
trap cleanup EXIT INT TERM

# ---- run one solo invocation in a clean dir; echo wall-clock seconds ----
solo_run(){  # $1=label $2="KMIN KMAX SHIFT"
  local d="$WORKDIR/$1"; rm -rf "$d"; mkdir -p "$d"; cp "$BINARY" "$d/app"
  ( cd "$d"; s=$(date +%s.%N); ./app $2 >run.log 2>&1; e=$(date +%s.%N); echo "$e - $s" | bc >elapsed.txt )
  cat "$d/elapsed.txt"
}
kdone(){ grep -c 'done in' "$WORKDIR/$1/run.log" 2>/dev/null || echo 0; }   # # of K searched

rm -rf "$WORKDIR"; mkdir -p "$MPSDIR/pipe" "$MPSDIR/log"
[ -x "$BINARY" ] || { echo "binary not found: $BINARY"; exit 1; }

# clean slate
pkill -x "$(basename "$BINARY")" 2>/dev/null; stop_mps
echo "pre: GPU $(nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader | head -1)"

# ---------------- solo baseline (no MPS): get w and S ----------------
log "== solo baseline (no MPS) =="
wait_for_idle; TF=$(solo_run solo_full "$G_FULL"); NF=$(kdone solo_full)
wait_for_idle; TH=$(solo_run solo_half "$G_HALF"); NH=$(kdone solo_half)
read W S WSOLO < <(awk -v tf="$TF" -v th="$TH" -v nf="$NF" -v nh="$NH" 'BEGIN{
  dk=nf-nh; if(dk<1)dk=1; w=(tf-th)/dk; s=tf-nf*w; if(s<0)s=0; printf "%.6f %.6f %.6f", w, s, tf-s }')
log "  G_FULL='$G_FULL' (K=$NF) Tfull=${TF}s ; G_HALF (K=$NH) Thalf=${TH}s"
log "  => per-K work w=${W}s ; startup S=${S}s ; solo work-time Wsolo=${WSOLO}s"

# ---------------- start MPS ----------------
log "== start MPS daemon =="
nvidia-cuda-mps-control -d && mps_up=1; sleep 2
log "  mps server: $(echo get_server_list | nvidia-cuda-mps-control 2>/dev/null | tr '\n' ' ')(starts on first client)"

echo "C,P,n_ok,mean_pertask_wall_s,eff_worktime_s,speedup" > "$OUTFILE"
declare -A SPD MEANT
diag_done=0

for C in "${CONCURRENCY[@]}"; do
  for P in "${MPS_PCT[@]}"; do
    wait_for_idle
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$P
    cdir="$WORKDIR/c${C}_p${P}"; mkdir -p "$cdir"
    pids=()
    for ((i=0;i<C;i++)); do
      wd="$cdir/task$i"; rm -rf "$wd"; mkdir -p "$wd"; cp "$BINARY" "$wd/app"
      ( cd "$wd"; s=$(date +%s.%N); timeout "$CFG_TIMEOUT" ./app $G_FULL >run.log 2>&1; e=$(date +%s.%N); echo "$e - $s"|bc >elapsed.txt ) &
      pids+=($!)
    done
    # one-time diagnostic: confirm C clients are concurrent under the MPS server
    if [ "$diag_done" = 0 ]; then sleep 3; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$WORKDIR/mps_concurrency_proof.txt" 2>&1; diag_done=1; fi
    for pp in "${pids[@]}"; do wait "$pp"; done

    # collect per-task times + ok count
    times=(); nok=0
    for ((i=0;i<C;i++)); do
      t=$(cat "$cdir/task$i/elapsed.txt" 2>/dev/null)
      g=$(grep -c 'search finished' "$cdir/task$i/run.log" 2>/dev/null)
      if [ -n "$t" ] && [ "${g:-0}" -ge 1 ]; then times+=("$t"); nok=$((nok+1)); fi
    done
    read MEAN EFF SPDV < <(awk -v S="$S" -v C="$C" -v WS="$WSOLO" 'BEGIN{n=0;sum=0}
      {for(i=1;i<=NF;i++){sum+=$i;n++}} END{ if(n==0){print "0 0 0"; exit}
        mean=sum/n; eff=(mean-S)/C; if(eff<=0)eff=0.000001; printf "%.3f %.4f %.3f", mean, eff, WS/eff }' <<<"${times[*]}")
    SPD["$C,$P"]=$SPDV; MEANT["$C,$P"]=$MEAN
    echo "$C,$P,$nok,$MEAN,$EFF,$SPDV" >> "$OUTFILE"
    printf "  C=%d P=%3d%%  mean/task=%6ss  speedup=%5sx  (n_ok=%d)\n" "$C" "$P" "$MEAN" "$SPDV" "$nok"
  done
done

stop_mps

# ---------------- grid tables ----------------
{
echo ""
echo "================== AP26 CUDA MPS sweep =================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)   binary: $(basename "$BINARY")"
echo "workload/task: '$G_FULL'  (K=$NF, solo ${TF}s)   startup S=${S}s   solo work=${WSOLO}s"
echo "metric: aggregate throughput speedup vs 1 task (1.00=no gain, =C perfect, <1 MPS hurts)"
echo ""
printf "  speedup     "; for P in "${MPS_PCT[@]}"; do printf "  %4d%%" "$P"; done; echo
printf "  ---------"; for P in "${MPS_PCT[@]}"; do printf " ------"; done; echo
for C in "${CONCURRENCY[@]}"; do
  printf "  C=%d        " "$C"
  for P in "${MPS_PCT[@]}"; do printf " %5sx" "${SPD[$C,$P]:- -}"; done; echo
done
echo ""
printf "  mean/task s "; for P in "${MPS_PCT[@]}"; do printf "  %4d%%" "$P"; done; echo
printf "  ---------"; for P in "${MPS_PCT[@]}"; do printf " ------"; done; echo
for C in "${CONCURRENCY[@]}"; do
  printf "  C=%d        " "$C"
  for P in "${MPS_PCT[@]}"; do printf " %6s" "${MEANT[$C,$P]:- -}"; done; echo
done
echo ""
echo "concurrency proof (a config's simultaneous MPS clients):"
sed 's/^/  /' "$WORKDIR/mps_concurrency_proof.txt"
echo "full CSV: $OUTFILE"
} | tee "$WORKDIR/summary.txt"
