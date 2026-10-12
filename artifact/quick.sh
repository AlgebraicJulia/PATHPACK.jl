#!/bin/bash
# The quick benchmark: one machine, no Slurm. Our solver against state-of-the-art codes on a few all-pairs graphs,
# timed and checked as in the reference runs, and compared with the reference measurements (data/). Run from anywhere:
#
#   artifact/quick.sh cpu           ours on 1 core and on all cores (at most 16), and n runs of Dijkstra's algorithm on
#                                   one core, on 7 graphs                                                    (~5 minutes)
#   artifact/quick.sh gpu           ours on the first NVIDIA GPU, on ROME's 16 test graphs that fit in its memory
#                                                                                              (~15 minutes on a B200)
#   artifact/quick.sh gpu --rome    the same, then ROME's three builds on the same GPU, and the speedups over ROME (ROME
#                                   is built the first time)                                   (~30 minutes on a B200)
# Options:
#   --graphs "G67 wing"   these graphs instead (names as in bench/graphs.jl)
#   --reps N              GPU: timed cold calls per graph, for ours and for each ROME build (default 5; the reference runs: 11)
#   --threads N           CPU: the threads of the "all cores" run (default: the physical cores, at most 16)
#
# Needs: Linux, Julia 1.12 (on the PATH, or JULIA=/path/to/julia; "juliaup add 1.12"), curl, Python 3.
#   gpu: an NVIDIA GPU and its driver (Julia downloads the CUDA runtime).
#   --rome: also git, gcc and g++, make, CMake, flex, bison, bc, GNU time (/usr/bin/time), and the CUDA toolkit (nvcc).
# The first run also installs the Julia packages at the reference runs' versions (bench/Manifest.toml, bench/gpu/Manifest.toml)
# and downloads only the graphs it runs. Everything goes under $WORK (default artifact/work). Output: the comparison,
# also in raw/quick/<run>/report.txt; the rows in results/*.csv (as from the Slurm harness; make regress compares them).
set -eo pipefail
cd "$(dirname "$0")"
usage() { awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' quick.sh; exit 1; }
MODE=${1:-}; [ $# -gt 0 ] && shift
case $MODE in cpu|gpu) ;; *) usage ;; esac
GRAPHS=""; REPS=5; THREADS=""; ROME=0
while [ $# -gt 0 ]; do
  case $1 in
    --graphs) GRAPHS=$2; shift ;; --reps) REPS=$2; shift ;; --threads) THREADS=$2; shift ;; --rome) ROME=1 ;;
    *) echo "unknown option $1"; usage ;;
  esac; shift
done
source hpc/env.sh
say() { echo; echo "==== $* ===="; }
need() { command -v $1 > /dev/null || { echo "$1 not found: $2"; exit 1; }; }

# ---- the tools ----
need python3 "install Python 3"; need curl "install curl"
JV=$($JULIA --startup-file=no -e 'print(VERSION.major, ".", VERSION.minor)' 2>/dev/null) \
  || { echo "Julia not found ($JULIA): install Julia 1.12 (https://julialang.org/install), or set JULIA=/path/to/julia"; exit 1; }
[ "$JV" = 1.12 ] || { echo "Julia $JV at $JULIA; these environments need 1.12 (juliaup add 1.12, then JULIA=\"julia +1.12\" $0 ...)"; exit 1; }
NCORES=$(lscpu -b -p=Core,Socket 2>/dev/null | grep -v "^#" | sort -u | wc -l)   # physical cores, at most the CPUs we may use
[ "$NCORES" -gt 0 ] && [ "$NCORES" -le "$(nproc)" ] || NCORES=$(nproc)
if [ $MODE = gpu ]; then
  need nvidia-smi "this needs an NVIDIA GPU and its driver"
  nvidia-smi -L > /dev/null || { echo "nvidia-smi cannot reach a GPU"; exit 1; }
  if [ $ROME = 1 ]; then
    for t in git gcc g++ make cmake flex bison bc; do need $t "ROME's build and runs need it"; done
    [ -x /usr/bin/time ] || { echo "/usr/bin/time not found: the ROME runs need GNU time (e.g. apt install time)"; exit 1; }
    command -v nvcc > /dev/null || [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ] || { echo "nvcc not found: building ROME needs the CUDA toolkit (set CUDA_HOME, or put nvcc on the PATH)"; exit 1; }
    command -v nvcc > /dev/null || export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
  fi
fi

RUN=$(date +%Y%m%d-%H%M%S)_$MODE; RAW=$REPO/raw/quick/$RUN; OUT=$RAW/task_0/out; mkdir -p $OUT
LOG=$RAW/run.log
exec > >(tee -a $LOG) 2>&1
echo "quick benchmark $MODE: run $RUN, library $(libcommit), Julia $($JULIA --startup-file=no -e 'print(VERSION)'), $NCORES cores"

if [ $MODE = cpu ]; then
  say "the Julia environment (bench/: this repository's library, the reference runs' package versions)"
  $JULIA --project=$BENCH -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
  T=${THREADS:-$(( NCORES < 16 ? NCORES : 16 ))}
  AVAIL=$(awk '/^MemAvailable/ {printf "%.1f", $2 / 1e6}' /proc/meminfo)              # GB
  OUTGB=$(python3 -c "print(round(0.6 * $AVAIL, 1))")                                  # the n x n output's budget
  [ -n "$GRAPHS" ] || GRAPHS=$(python3 scripts/quick_report.py plan cpu $OUTGB)
  [ -n "$GRAPHS" ] || { echo "no graph fits in $OUTGB GB"; exit 1; }
  say "the graphs: $GRAPHS"
  bash hpc/fetch_data.sh $GRAPHS
  cat > $RAW/task_0/PROVENANCE.json << JSON
{"commit": "$(libcommit)", "machine": "$(lscpu | sed -n 's/^Model name: *//p' | head -1)", "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "job": "quick-$RUN", "target": "quick cpu"}
JSON
  for t in 1 $T; do
    say "ours and n runs of Dijkstra's algorithm on $t thread(s)"
    ( cd $BENCH && BENCH_RESULTS=$OUT BENCH_NT=$t BENCH_OUT_GB=$OUTGB BENCH_PLACEMENT="quick.sh, unpinned" \
        $JULIA -t $t,0 --project=$BENCH apsp.jl $GRAPHS )
  done
  echo "QUICK cpu DONE"
else
  # CUDA.jl brings its own CUDA toolkit: one on the PATH or LD_LIBRARY_PATH (for nvcc) must not shadow its libraries,
  # which Julia also caches in its depot. ROME's build gets them back; ROME's binaries link no CUDA library dynamically.
  PATH_TOOLKIT=$PATH LDLP_TOOLKIT=${LD_LIBRARY_PATH:-}
  nocuda() { printf %s "$1" | awk -v RS=: 'NF && tolower($0) !~ /cuda/' | paste -sd: -; }
  export PATH=$(nocuda "$PATH") LD_LIBRARY_PATH=$(nocuda "${LD_LIBRARY_PATH:-}")
  say "the Julia environment (bench/gpu: this repository's library with CUDA.jl, the reference runs' package versions)"
  DRV=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -1)        # the newest CUDA the driver runs
  [ -n "$DRV" ] && [ "$DRV" -lt 13 ] && export CUDA_RUNTIME=auto                      # (the reference runs' runtime, 13.0, needs a 13.x driver)
  bash hpc/setup_gpu.sh julia
  INFO=$($JULIA --project=$BENCH_GPU -e 'using CUDA; CUDA.functional() || (CUDA.versioninfo(); exit(1)); d = CUDA.device()
      free = isdefined(CUDA, :free_memory) ? CUDA.free_memory() : isdefined(CUDA, :available_memory) ? CUDA.available_memory() : CUDA.totalmem(d)
      print(CUDA.name(d), "|", round(free / 1e9; digits = 1), "|", CUDA.capability(d).major, CUDA.capability(d).minor)' | tail -1) \
    || { echo "CUDA.jl cannot use this GPU (its report is above)"; exit 1; }
  IFS='|' read GPUNAME GPUFREE ARCH <<< "$INFO"
  echo "GPU: $GPUNAME, $GPUFREE GB free, sm_$ARCH"
  [ -n "$GRAPHS" ] || GRAPHS=$(python3 scripts/quick_report.py plan gpu $GPUFREE)
  [ -n "$GRAPHS" ] || { echo "no graph fits in $GPUFREE GB"; exit 1; }
  say "the graphs: $GRAPHS"
  bash hpc/fetch_data.sh $GRAPHS
  export GPU_CPUS=$(( NCORES < 16 ? NCORES : 16 )) GPU_JULIA_THREADS=$(( NCORES < 8 ? NCORES : 8 ))
  TASK="BENCH_REPS=$REPS BENCH_VARIANTS=auto:hybrid jlg gpu_apsp.jl $GRAPHS || true"
  if [ $ROME = 1 ]; then
    export ROME_DIR=$BASELINES/rome-sm$ARCH; export ROMEBIN=$ROME_DIR/bin
    if ! ls $ROMEBIN/rome_{default,leaf400,leaf800,instr} > /dev/null 2>&1; then
      say "ROME for sm_$ARCH (baselines/rome: ROME 25beef3 with our patches, METIS, Scotch) -> $ROME_DIR"
      PATH=$PATH_TOOLKIT LD_LIBRARY_PATH=$LDLP_TOOLKIT ROME_ARCHS=$ARCH bash hpc/setup_gpu.sh rome
    fi
    TASK="$TASK; ROME_NREP=$REPS rome_runs $GRAPHS; jlg gpu_baselines.jl rome $GRAPHS || true"
  fi
  say "the runs (ours$( [ $ROME = 1 ] && echo ', then ROME'))"
  HPC_DIR=$HPC GPU_TASK="$TASK" GPU_TASK_ID=0 bash hpc/gpu_job.sh gpu_one $RAW
fi

say "results"
python3 scripts/collect.py --out $RAW/csv --results $OUT --scale $OUT > /dev/null      # this run alone, for the report
mkdir -p $REPO/results; bash hpc/collect.sh $RAW > /dev/null                          # merged into results/ (make regress)
say "comparison with the reference runs"
python3 scripts/quick_report.py report $MODE $RAW/csv | tee $RAW/report.txt
echo; echo "raw results and logs: raw/quick/$RUN; rows: results/*.csv"
