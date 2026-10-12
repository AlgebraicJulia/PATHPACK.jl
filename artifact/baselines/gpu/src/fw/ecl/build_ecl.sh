#!/bin/bash
# Build ECL-APSP (patched 64-bit) in our harness for sm_100 + sm_120 (login node).
command -v module >/dev/null 2>&1 && module load cuda/12.9.1 >/dev/null 2>&1 || true   # (or CUDA 12.9 nvcc on PATH)
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # the build root: src/, jobs/, ext/ (fetch_ext.sh), envs/, build/
cd $B/src/fw/ecl
python3 -I make_ecl_kernels.py $B/ext/ECL-APSP/ECL-APSP_10.cu > ecl_kernels.cuh || exit 1
mkdir -p $B/build/fw
# ECL's README/Makefile: nvcc -O3 -arch=sm_XX
nvcc -O3 -std=c++17 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120 \
  -o $B/build/fw/ecl_apsp ecl_driver.cu > $B/build/fw/build_ecl.log 2>&1 || { cat $B/build/fw/build_ecl.log; exit 1; }
grep -v "Warning: The 'compute_" $B/build/fw/build_ecl.log | head
ls -la $B/build/fw/ecl_apsp
cuobjdump -res-usage $B/build/fw/ecl_apsp 2>&1 | grep -A1 "FWrem\|FWrowcol\|FW0" | grep -E "REG" | head
