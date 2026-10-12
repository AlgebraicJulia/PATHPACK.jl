#!/bin/bash
# One-time setup for the CPU experiments (safe to rerun: it skips what is already done).
#   1. check the library (this repository)   2. install the Julia packages   3. build the C++ baselines   4. download the graphs
set -eo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p $WORK
step() { echo; echo "==== $* ===="; }

step "1/4 library: $LIB"
[ -f $LIB/Project.toml ] || { echo "no Project.toml in $LIB: run this from the artifact/ folder of the repository"; exit 1; }
echo "version $(libcommit)"

step "2/4 Julia packages for bench/ (the library above; CliqueTrees as in the reference runs)"
[ -x "$JULIA" ] || { echo "Julia not found at $JULIA. Install it with:  curl -fsSL https://install.julialang.org | sh"; exit 1; }
CTV=${CLIQUETREES_VERSION:-1.19.7}
if [ -f $BENCH/Manifest.toml ]; then   # the exact package versions of the reference CPU runs (Julia 1.12.7)
  $JULIA --project=$BENCH -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
else
  $JULIA --project=$BENCH -e "using Pkg; Pkg.develop(path=ENV[\"LIB\"]); Pkg.add(name=\"CliqueTrees\", version=\"$CTV\"); Pkg.instantiate(); Pkg.precompile()"
fi

step "3/4 C++ baselines -> $BASELINES"
module load gcc >/dev/null 2>&1 || true
mkdir -p $BASELINES && cd $BASELINES
[ -d gapbs ] || git clone -q https://github.com/sbeamer/gapbs
git -C gapbs checkout -q --detach 2972aeb2703165bafd921222f4ed7196f542d3a8   # the commit of the reference runs
( cd gapbs && git checkout -q -- src/benchmark.h src/builder.h \
  && sed -i 's/^typedef int32_t WeightT;/typedef double WeightT;/' src/benchmark.h \
  && python3 - << 'PY'
p = "src/builder.h"; s = open(p).read()
a = "    if (needs_weights_)\n      Generator<NodeID_, DestID_, WeightT_>::InsertWeights(el);\n"
assert s.count(a) == 1, "GAP builder.h changed: update the double-weight patch"
open(p, "w").write(s.replace(a, "    if (needs_weights_) {  // double weights: random weight insertion is integer-only\n      std::cout << \"input has no weights\" << std::endl;\n      exit(-31);\n    }\n"))
PY
  g++ -std=c++11 -O3 -march=native -fopenmp src/sssp.cc -o sssp_double \
  && g++ -std=c++11 -O3 -march=native -fopenmp -I src $REPO/baselines/gap_apsp.cpp -o gap_apsp )
ldd gapbs/sssp_double | grep -q gomp || { echo "GAP's sssp_double was built without OpenMP: check the compiler module"; exit 1; }
[ -d RoutingKit ] || git clone -q https://github.com/RoutingKit/RoutingKit
git -C RoutingKit checkout -q --detach 54d49bb0cdea56dde182357522e4e86a03c57852
cp $REPO/baselines/phast_driver.cpp $REPO/baselines/cch_driver.cpp RoutingKit/
( cd RoutingKit && make -s && for d in phast_driver cch_driver; do
    g++ -std=c++17 -O3 -march=native -fopenmp -I include $d.cpp lib/libroutingkit.a -o $d; done )
[ -d boost-1.86.0 ] || { curl -sL https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-b2-nodocs.tar.gz | tar -xz --wildcards '*/boost/*'; }
g++ -std=c++17 -O3 -march=native -I$BASELINES/boost-1.86.0 $REPO/baselines/bottleneck_boost.cpp -o $BASELINES/bottleneck_boost
[ -d STREAM ] || git clone -q https://github.com/jeffhammond/STREAM
git -C STREAM checkout -q --detach 6703f7504a38a8da96b353cadafa64d3c2d7a2d3
( cd STREAM && gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream_c )
echo "built: $(ls $BASELINES/gapbs/sssp_double $BASELINES/RoutingKit/phast_driver $BASELINES/RoutingKit/cch_driver $BASELINES/STREAM/stream_c $BASELINES/bottleneck_boost | xargs -n1 basename | paste -sd' ')"

step "4/4 graph data -> $BENCH_DATA"
bash $HPC/fetch_data.sh

echo; echo "Setup done. Try:  ./hpc/run.sh queries --dry-run"
