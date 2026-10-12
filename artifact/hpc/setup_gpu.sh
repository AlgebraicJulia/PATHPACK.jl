#!/bin/bash
# One-time setup for the GPU experiments (safe to rerun; run hpc/setup.sh first for the graph data).
#   ./hpc/setup_gpu.sh [julia | tests | rome | others | check]     (default: all, in this order)
#   1. julia    the GPU benchmarks' environment (bench/gpu): this repository's library with CUDA.jl, precompiled here
#               (set JULIA_CPU_TARGET in config.local.sh if the login and GPU nodes have different CPUs)
#   2. tests    the library's own GPU test environment (test/gpu), for target gpu_tests
#   3. rome     ROME's binaries (baselines/rome/: prepare_rome.sh makes the sources, rome_patch.py patches them: device-only mode and a
#               post-timer dump of sampled rows; ROME's timers and kernels unchanged), default and tuned leaf sizes,
#               and the instrumented build (allocation split); login node, compile only
#   4. others   cuGraph, Gunrock, ADDS, dense Floyd-Warshall, ECL-APSP, Fast-APSP: the earlier comparison's drivers with
#               per-row fingerprints for real weights (baselines/gpu_fingerprint.py), built into $BASELINES/gpu
#   5. check    Fast-APSP's driver build ($FASTAPSP_BUILD) and the baselines' binaries are present
set -eo pipefail
source "$(dirname "$0")/env.sh"
step() { echo; echo "==== $* ===="; }
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
ONLY=$1

if want julia; then
  step "Julia environment for bench/gpu ($LIB_GPU with CUDA.jl; CliqueTrees as in the reference runs)"
  CTV=${CLIQUETREES_VERSION:-1.19.7}
  # the CUDA runtime: the reference runs', 13.0 (needs a driver for CUDA 13; pinned so that a login node without a driver can
  # install it), or CUDA_RUNTIME=auto: the newest one the driver runs (quick.sh sets this for older drivers)
  if [ "${CUDA_RUNTIME:-13.0}" = auto ]; then rm -f $BENCH_GPU/LocalPreferences.toml
  else printf '# The CUDA runtime CUDA.jl uses (hpc/setup_gpu.sh writes this: CUDA_RUNTIME, default 13.0 as in the reference runs).\n[CUDA_Runtime_jll]\nversion = "%s"\n' ${CUDA_RUNTIME:-13.0} > $BENCH_GPU/LocalPreferences.toml; fi
  if [ -f $BENCH_GPU/Manifest.toml ]; then   # the exact package versions of the reference GPU runs (Julia 1.12.6, CUDA.jl 6.1.0)
    $JULIA --project=$BENCH_GPU -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
  else
    $JULIA --project=$BENCH_GPU -e "using Pkg; Pkg.develop(path=ENV[\"LIB_GPU\"]); Pkg.add(name=\"CliqueTrees\", version=\"$CTV\"); Pkg.instantiate(); Pkg.precompile()"
  fi
  $JULIA --project=$BENCH_GPU -e 'using PATHPACK, CUDA, CliqueTrees; println("PATHPACK at ", pkgdir(PATHPACK), ", CUDA.jl ", pkgversion(CUDA), ", CliqueTrees ", pkgversion(CliqueTrees), ", extension ", Base.get_extension(PATHPACK, :PATHPACKCUDAExt) !== nothing)'
fi

if want tests; then
  step "the library's GPU test environment -> $WORK/gpu-testenv (test/gpu/Project.toml, the library above)"
  T=$WORK/gpu-testenv; mkdir -p $T
  sed -e "s#^PATHPACK = {path = .*#PATHPACK = {path = \"$LIB_GPU\"}#" $LIB_GPU/test/gpu/Project.toml > $T/Project.toml
  grep -q '^\[extras\]' $T/Project.toml || printf '\n[extras]\nCUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"\n' >> $T/Project.toml
  if [ -f $BENCH_GPU/LocalPreferences.toml ]; then cp $BENCH_GPU/LocalPreferences.toml $T/; else rm -f $T/LocalPreferences.toml; fi
  [ -f $BENCH_GPU/tests.Manifest.toml ] && sed "s#@LIB@#$LIB_GPU#" $BENCH_GPU/tests.Manifest.toml > $T/Manifest.toml   # the reference runs' versions
  $JULIA --project=$T -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
fi

if want rome; then
  step "ROME -> $BASELINES/rome"
  if [ -z "$ROME_SRC" ]; then          # ROME at 25beef3 with our patches, METIS and Scotch: baselines/rome/prepare_rome.sh
    bash $REPO/baselines/rome/prepare_rome.sh $WORK/rome-src
    ROME_SRC=$WORK/rome-src/src-fair; ROME_SRC_INSTR=$WORK/rome-src/src-instr; ROME_MODULES=$WORK/rome-src/modules; ROME_SCOTCH=$WORK/rome-src/scotch-build-O3
  fi
  module load cuda/12.9.1 cmake/3.30.5 2>/dev/null || true    # (CUDA 12.9 and CMake >= 3.30 on the PATH otherwise)
  R=${ROME_DIR:-$BASELINES/rome}; mkdir -p $R/bin      # (ROME_DIR, ROME_ARCHS: e.g. an sm_89 set for an L4)
  ROME_ARCHS=${ROME_ARCHS:-100;120}                                 # (the reference runs': sm_100, B200, and sm_120)
  for v in default leaf400 leaf800 instr; do
    s=$R/src-$v; b=$R/b-$v; rm -rf $s $b; mkdir -p $s
    src=$ROME_SRC; [ $v = instr ] && src=$ROME_SRC_INSTR
    cp -a $src/CMakeLists.txt $src/src $src/include $s/
    python3 $REPO/baselines/rome/rome_patch.py $s
    case $v in leaf*) sed -i -E "s/sep=\(\/vert<150\?/sep=(\/vert<${v#leaf}?/" $s/src/ETree/ETreeOptimized.cpp
                      grep -q "vert<${v#leaf}?" $s/src/ETree/ETreeOptimized.cpp || { echo "leaf size edit failed"; exit 1; } ;; esac
    ( cd $R && diff -ru $src $s > $R/src-$v.diff ) || true
    mkdir -p $b/modules/scotch; cp $s/CMakeLists.txt $b/; ln -s $s/src $b/src; ln -s $s/include $b/include
    ln -s $ROME_MODULES/metis $b/modules/metis; ln -s $ROME_SCOTCH $b/modules/scotch/build
    mkdir -p $b/build && cd $b/build
    # (the reference builds: the system GCC; ROME_CC and ROME_CXX for a host compiler your nvcc supports)
    CC=${ROME_CC:-/usr/bin/gcc} CXX=${ROME_CXX:-/usr/bin/g++} cmake -Wno-dev -DCMAKE_BUILD_TYPE=Release -DUSE_THREAD=Y -DROME_ARCHS="$ROME_ARCHS" \
      -DCMAKE_CUDA_HOST_COMPILER=${ROME_CXX:-/usr/bin/g++} -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME .. > cmake.log 2>&1 || { tail -30 cmake.log; exit 1; }
    nice -n 10 cmake --build . -j${J:-8} > make.log 2>&1 || { grep -iE "error" make.log | head; tail -20 make.log; exit 1; }
    cp bin/rome $R/bin/rome_$v
    echo "rome_$v $(date -Is) sha256 $(sha256sum $R/bin/rome_$v | cut -d' ' -f1) from $src (+ rome_patch.py$( [ $v != default ] && [ $v != instr ] && echo ", Scotch leaf ${v#leaf}"))" | tee -a $R/bin/BUILDINFO.txt
  done
fi

if want others; then
  step "the other GPU baselines -> $BASELINES/gpu (cuGraph, Gunrock, ADDS, dense FW, ECL-APSP, Fast-APSP)"
  # the earlier comparison's drivers and build scripts ($GPU_BASELINES_SRC), copied, with per-row fingerprints for real
  # weights (baselines/gpu_fingerprint.py; outside every timer), built into a folder of their own
  [ -d "$GPU_BASELINES_SRC" ] || { echo "GPU_BASELINES_SRC ($GPU_BASELINES_SRC) not found"; exit 1; }
  N=$BASELINES/gpu; mkdir -p $N/build $N/results
  rm -rf $N/src $N/jobs; cp -a $GPU_BASELINES_SRC/src $N/src; mkdir -p $N/jobs; cp $GPU_BASELINES_SRC/jobs/build_fastapsp.sh $N/jobs/
  bash $GPU_BASELINES_SRC/fetch_ext.sh $N/ext                                       # the third-party sources, at pinned commits
  [ -d $N/envs/rapids ] || bash $GPU_BASELINES_SRC/envs/make_rapids_env.sh $N/envs/rapids   # cuGraph (RAPIDS, conda)
  python3 $GPU_BASELINES_SRC/gpu_fingerprint.py $N/src
  for b in $N/src/sssp/cugraph/build_cugraph.sh $N/src/sssp/gunrock/build_gunrock.sh $N/src/sssp/adds/build_adds.sh \
           $N/src/fw/build.sh $N/src/fw/ecl/build_ecl.sh $N/jobs/build_fastapsp.sh; do
    echo "-- $(basename $b)"; ( bash $b 2>&1 | tail -4 ) || { echo "$b failed"; exit 1; }
  done
fi

if want check; then
  step "baselines present"
  for f in $BASELINES/rome/bin/rome_default $BASELINES/rome/bin/rome_instr $FASTAPSP_BUILD/fap_driver $FASTAPSP_BUILD/fap_mpi_driver $FASTAPSP_BUILD/mtx2fap \
           $BASELINES/gpu/build/cugraph/cugraph_apsp $BASELINES/gpu/build/gunrock/gunrock_apsp $BASELINES/gpu/build/adds/float/adds_apsp_float \
           $BASELINES/gpu/build/fw/fwbench $BASELINES/gpu/build/fw/ecl_apsp; do
    [ -x $f ] && echo "  ok      $f" || echo "  MISSING $f"
  done
fi
echo; echo "GPU setup done. Try:  ./hpc/run_gpu.sh gpu_kernels --dry-run"
