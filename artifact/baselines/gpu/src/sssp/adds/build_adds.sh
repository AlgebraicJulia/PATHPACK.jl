#!/bin/bash
# Build ADDS (UCLA-VAST/adds @ 9ef8166, PPoPP'21) APSP drivers for sm_100 + sm_120 on a login node.
# Copies ads_int / ads_float from ext/adds into build/adds/{int,float} and applies three porting patches:
#   1. MAX_TB (int: 128, float: 256 "increased to use more SMs") -> 512 in wl.h. The persistent kernel uses
#      num_tb = tb_per_sm*(SMs-1) blocks; B200 has 148 SMs and RTX PRO 6000 188, so num_tb (up to ~374) exceeds
#      the Turing-era cap and wl.alloc() asserts. MAX_TB only sizes shared/host arrays indexed by tb.
#   2. drop the unused '#include <boost/mpl/if.hpp>' in csr_graph.cu (not needed: nothing from Boost is used).
#   3. kernel.cu: '#include "../cub-1.8.0/cub/cub.cuh"' -> '#include <cub/cub.cuh>' (the toolkit's CCCL CUB):
#      the bundled CUB 1.8.0 (2018) does not compile with CUDA 12.9 (CUB_NS_QUALIFIER #error). Only
#      cub::ThreadLoad/WarpScan/BlockReduce/LaneId are used, all unchanged in modern CUB.
#   4. nothing else: kernels, worklist, delta heuristics and launch config are untouched. Our driver
#      (src/sssp/adds/adds_apsp.cu) #includes their kernel.cu with main renamed, and replaces only main()/gg_main()
#      with a loop over sources that reuses their exact per-source sequence (init kernel, wl.reinit, driver kernel).
set -e
command -v module >/dev/null 2>&1 && module load cuda/12.9.1 >/dev/null 2>&1 || true   # (or CUDA 12.9 nvcc on PATH)
B=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # the build root: src/, jobs/, ext/ (fetch_ext.sh), envs/, build/
E=$B/ext/adds
O=$B/build/adds
mkdir -p $O
ln -sfn $E/cub-1.8.0 $O/cub-1.8.0
ARCH="-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120"
for v in int float; do
  # build in a staging dir and swap at the end, so running jobs never see a missing binary
  rm -rf $O/$v.new; cp -r $E/ads_$v $O/$v.new
  sed -i 's|#include <boost/mpl/if.hpp>||' $O/$v.new/csr_graph.cu
  sed -i 's|#include "../cub-1.8.0/cub/cub.cuh"|#include <cub/cub.cuh> /* port: was ../cub-1.8.0 */|' $O/$v.new/kernel.cu
  sed -i -E 's|^#define MAX_TB (128\|256)(.*)$|#define MAX_TB 512 /* port: was \1 */|' $O/$v.new/wl.h
  grep -n "define MAX_TB" $O/$v.new/wl.h
  cp $B/src/sssp/adds/adds_apsp.cu $O/$v.new/
  DEF=""; [ $v = float ] && DEF="-DADDS_FLOAT"
  ( cd $O/$v.new && nvcc -std=c++17 $DEF -I$B/src/common adds_apsp.cu support.cu csr_graph.cu -O3 $ARCH -rdc=true -lcudadevrt \
      -Xptxas -O3 -Xcompiler -fopenmp -Xcompiler -O3 -lcuda -lgomp -o adds_apsp_$v 2>&1 | grep -v -E "Warning: .*deprecated|^$" | grep -E -i "error|registers|warning" | head -30 )
  [ -x $O/$v.new/adds_apsp_$v ] || { echo "build of $v failed"; exit 1; }
  # swap: the binary path is replaced by rename (running processes keep their inode)
  mkdir -p $O/$v
  for f in $O/$v.new/*; do [ "$(basename $f)" = adds_apsp_$v ] || cp $f $O/$v/; done   # never cp onto a running binary
  mv -f $O/$v.new/adds_apsp_$v $O/$v/adds_apsp_$v.tmp && mv -f $O/$v/adds_apsp_$v.tmp $O/$v/adds_apsp_$v
  rm -rf $O/$v.new
  ls -la $O/$v/adds_apsp_$v
  cuobjdump --list-elf $O/$v/adds_apsp_$v | head
done
