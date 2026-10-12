#!/bin/bash
# Build the cuGraph APSP driver against libcugraph 26.08 (conda env envs/rapids: rapidsai + conda-forge,
# cuda-version=12.9, cuda-nvcc 12.9, gxx 14). libcugraph ships SASS for sm_100 and sm_120 itself; our driver
# is compiled for sm_100 + sm_120. rpath to the env lib dir, so no LD_LIBRARY_PATH is needed at run time.
# Env creation (once, login node): envs/make_rapids_env.sh $B/envs/rapids (the exact package list we used), made with
#   micromamba create -p $B/envs/rapids -c rapidsai -c conda-forge libcugraph=26.08 cuda-version=12.9 \
#       cuda-nvcc=12.9 cuda-cudart-dev=12.9 gxx_linux-64=14 "cmake>=3.30" make
set -e
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # the build root: src/, jobs/, ext/ (fetch_ext.sh), envs/, build/
R=$B/envs/rapids
O=$B/build/cugraph; mkdir -p $O
export PATH=$R/bin:$PATH
ARCH="-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120"
cd $O
$R/bin/nvcc -std=c++20 -O3 $ARCH -ccbin $R/bin/x86_64-conda-linux-gnu-g++ --expt-extended-lambda --expt-relaxed-constexpr \
  -I$B/src/common -I$R/include -I$R/include/rapids -I$R/targets/x86_64-linux/include \
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DRAFT_SYSTEM_LITTLE_ENDIAN=1 \
  -Xcompiler -fopenmp -Xcompiler -O3 \
  $B/src/sssp/cugraph/cugraph_apsp.cu -o cugraph_apsp.new \
  -L$R/lib -Xlinker -rpath -Xlinker $R/lib -lcugraph -lraft -lrmm -lrapids_logger -lcudart -lgomp 2>&1 | grep -v "^$" | head -40
mv -f $O/cugraph_apsp.new $O/cugraph_apsp  # atomic swap
ls -la $O/cugraph_apsp
ldd $O/cugraph_apsp | grep -E "not found|cugraph|cudart|rmm"
