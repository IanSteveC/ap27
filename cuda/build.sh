#!/bin/bash
#
# Build the AP26 CUDA application.
#
#   ./build.sh                 NVRTC dev build (compiles kernels at runtime;
#                              needs libnvrtc present). Used for validation.
#   ./build.sh --fatbin        AOT build: nvcc multi-arch fatbins embedded,
#                              loaded via the driver. NO NVRTC at runtime -
#                              the self-contained shipped BOINC binary.
#   ./build.sh --no-boinc      standalone (do not link libboinc)
#
# Env overrides: CUDA=/usr/local/cuda  BOINC_DIR=/home/ian/builds/boinc
#
set -e
cd "$(dirname "$0")"

CUDA="${CUDA:-/usr/local/cuda-12.9}"
BOINC_DIR="${BOINC_DIR:-/home/ian/builds/boinc}"
VER="23_5_6"		# upstream app version (printed via -DVERS; not in the binary name)

EMBED=""; NOBOINC="${NOBOINC:-}"
for a in "$@"; do
  case "$a" in
    --fatbin)   EMBED=1 ;;
    --no-boinc) NOBOINC=1 ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

CXX=g++
KERNELS="checkn clearn clearok clearokok offset setupn setupok setupokok sieve sieve_nv"

# 1. embed kernel sources as C strings (var <name>_cl, same names the host uses)
echo "[cltoh] generating kernel source headers"
for k in $KERNELS; do perl cltoh.pl "kernels/$k.cu" > "kernels/$k.h"; done

CXXFLAGS="-I . -I kernels -I $CUDA/include -O3 -m64 -DVERS=\\\"$VER\\\""
LDFLAGS="-static-libgcc -static-libstdc++"

OBJEXTRA=""
if [ -n "$EMBED" ]; then
  # ---- AOT fatbin path (driver-only, NO NVRTC) ----
  echo "[fatbin] building per-kernel multi-arch fatbins with nvcc"
  bash ./build_fatbins.sh                 # produces fatbins.s + fatbins.h
  $CXX -c fatbins.s -o fatbins.o          # .incbin-embedded fatbin blobs
  OBJEXTRA="fatbins.o"
  CPPEXTRA="-DAP26_EMBED_FATBINS"
  CULIBS="-L $CUDA/lib64/stubs -lcuda"     # link stub; client's real driver at runtime
  APP="ap27_linux64_cuda"
else
  # ---- NVRTC dev path ----
  CPPEXTRA=""
  CULIBS="-L $CUDA/lib64 -lnvrtc -lcuda"
  APP="ap27_linux64_cuda_nvrtc"
fi

if [ -z "$NOBOINC" ]; then
  [ -f "$BOINC_DIR/api/libboinc_api.a" ] || { echo "no BOINC at $BOINC_DIR (use --no-boinc)"; exit 1; }
  BINC="-I$BOINC_DIR -I$BOINC_DIR/api -I$BOINC_DIR/lib"
  BLIBS="$BOINC_DIR/api/libboinc_api.a $BOINC_DIR/lib/libboinc.a"
else
  BINC="-DSTANDALONE"; BLIBS=""
fi

echo "[cc] AP26.cpp ${EMBED:+(fatbin)}${EMBED:-(nvrtc)}"
eval $CXX $CXXFLAGS $CPPEXTRA $BINC -c -o AP26.o AP26.cpp

echo "[ld] $APP"
$CXX $LDFLAGS AP26.o $OBJEXTRA $CULIBS $BLIBS -lpthread -ldl -lrt -o "$APP"
echo "built: $APP"
