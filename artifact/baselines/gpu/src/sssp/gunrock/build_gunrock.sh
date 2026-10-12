#!/bin/bash
# Build the Gunrock (essentials API, github.com/gunrock/gunrock main) APSP driver, sm_100 + sm_120, on the login node.
# Uses Gunrock's own CUDA release flags (--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math, C++17,
# separable compilation). SM_TARGET: Gunrock's sm_flag_t enum stops at sm_90, and all operators use the
# "fallback" launch params, so SM_TARGET=90 selects exactly the same launch configuration on both GPUs
# (CMake would pass the invalid SM_TARGET=100;120 for a fat binary). CCCL: the CUDA 12.9 toolkit's.
set -e
command -v module >/dev/null 2>&1 && module load cuda/12.9.1 >/dev/null 2>&1 || true   # (or CUDA 12.9 nvcc on PATH)
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # the build root: src/, jobs/, ext/ (fetch_ext.sh), envs/, build/
E=$B/ext/gunrock
O=$B/build/gunrock; mkdir -p $O
[ -d $B/ext/cxxopts ] || git clone -q --depth 1 -b v3.0.0 https://github.com/jarro2783/cxxopts.git $B/ext/cxxopts
[ -d $B/ext/nlohmann_json ] || git clone -q --depth 1 -b v3.11.3 https://github.com/nlohmann/json.git $B/ext/nlohmann_json
ARCH="-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120"
cd $O
nvcc -std=c++17 -O3 $ARCH --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math \
  --diag-suppress=177 --diag-suppress=550 --diag-suppress=20014 -rdc=true \
  -DSM_TARGET=90 -DESSENTIALS_BACKEND=\"NVIDIA\" -D__HIP_PLATFORM_NVIDIA__ -DHIP_BACKEND=1 \
  -DESSENTIALS_VERSION=\"main\" -DGIT_SHA1=\"$(cd $E && git rev-parse HEAD)\" \
  -I$B/src/common -I$E/include -I$E/include/gunrock/compat -I$B/ext/cxxopts/include -I$B/ext/nlohmann_json/include \
  -Xcompiler -fopenmp -Xcompiler -O3 \
  $B/src/sssp/gunrock/gunrock_apsp.cu -o gunrock_apsp.new -lcuda -lgomp -lstdc++fs 2>&1 | grep -v "^$" | head -40
mv -f $O/gunrock_apsp.new $O/gunrock_apsp  # atomic swap: running jobs keep their inode
ls -la $O/gunrock_apsp
cuobjdump --list-elf $O/gunrock_apsp | head
