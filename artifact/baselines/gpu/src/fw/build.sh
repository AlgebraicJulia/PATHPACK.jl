#!/bin/bash
# Build fwbench (tuned + cuASR + Lund-Smith) for sm_100 and sm_120 on a login node.
command -v module >/dev/null 2>&1 && module load cuda/12.9.1 >/dev/null 2>&1 || true   # (or CUDA 12.9 nvcc on PATH)
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # the build root: src/, jobs/, ext/ (fetch_ext.sh), envs/, build/
CUASR=$B/ext/cuASR
mkdir -p $B/build/fw
cd $B/src/fw
nvcc -O3 -std=c++17 -lineinfo -DWITH_CUASR \
  -I$CUASR/include -I$CUASR/cutlass/include \
  -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120 \
  -Xcompiler -fopenmp,-O3 -o $B/build/fw/fwbench.new fwbench.cu "$@" > $B/build/fw/build.log 2>&1
rc=$?
grep -v "Warning: The 'compute_" $B/build/fw/build.log | head -40
if [ $rc -ne 0 ]; then echo "BUILD FAILED rc=$rc"; exit 1; fi
mv $B/build/fw/fwbench.new $B/build/fw/fwbench
ls -la $B/build/fw/fwbench
cuobjdump -res-usage $B/build/fw/fwbench 2>&1 | grep -A1 "minplus_gemm_tuned" | grep -E "Function|REG" | sed "s/Function \(.\{60\}\).*/\1/; s/ TEXTURE.*//"
