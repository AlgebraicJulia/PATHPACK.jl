#!/bin/bash
# One task of a target, on one whole NUMA region (16 cores and that region's memory).
# Submitted by run.sh; do not call directly.  Arguments: TARGET RAW_DIR
set -eo pipefail
TARGET=$1; RAW=$2; I=${SLURM_ARRAY_TASK_ID:-0}
source "${HPC_DIR:-$(dirname "$0")}/env.sh"
module load gcc >/dev/null 2>&1 || true

source "$HPC/baselines.sh"
source "$HPC/targets/$TARGET.sh"                   # defines TASKS (and WHOLE_NODE for whole-node targets)

# ---- where are we? refuse to run on cores scattered over several regions ----
NODES=$(numactl --show | awk '/^nodebind/ {for (i = 2; i <= NF; i++) printf "%s ", $i}')
CORES=$(numactl --show | awk '/^physcpubind/ {for (i = 2; i <= NF; i++) printf "%s ", $i}')
set -- $NODES; REG=${REGIONS:-1}
if [ "${WHOLE_NODE:-0}" = 1 ]; then R=all; MEMPOL="--interleave=all"
elif [ $# -ne $REG ] || [ $(echo $CORES | wc -w) -ne $((16 * REG)) ]; then
    echo "PLACEMENT: expected $REG whole NUMA region(s) of 16 cores, got regions ($NODES) and $(echo $CORES | wc -w) cores: resubmit"; exit 3
elif [ $REG = 1 ]; then R=$1; MEMPOL="--membind=$R"
else R=$(echo $NODES | tr ' ' ','); MEMPOL="--interleave=$R"; fi
export BENCH_PLACEMENT="regions $R, $(echo $CORES | wc -w) cores, memory $MEMPOL"
first() { echo $CORES | tr ' ' '\n' | head -$1 | paste -sd,; }

OUT=$RAW/task_$I/out; mkdir -p $OUT
COMMIT=$(libcommit)
cat > $RAW/task_$I/PROVENANCE.json << JSON
{"commit": "$COMMIT", "machine": "$(lscpu | sed -n 's/^Model name: *//p' | head -1)", "host": "$(hostname -s)",
 "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "job": "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$I", "region": "$R", "target": "$TARGET"}
JSON
echo "target $TARGET, task $I, region $R, cores $CORES, library $COMMIT"

# ---- helpers used by the task lines in targets/*.sh ----
# jl T SCRIPT ARGS...   Julia benchmark on the first T cores of this region, results into $OUT
jl() { local t=$1; shift; ( cd $BENCH && BENCH_RESULTS=$OUT BENCH_NT=$t numactl --physcpubind=$(first $t) $MEMPOL \
       julia -t $t,0 --project=$BENCH "$@" ); }
# c1 CMD...             a C++ baseline on one core
c1() { numactl --physcpubind=$(first 1) $MEMPOL "$@"; }
# c16 CMD...            a C++ baseline on the whole region
c16() { numactl --physcpubind=$(first 16) $MEMPOL "$@"; }
# core I                the I-th core of this allocation (1-based); ncores: how many cores we have
core() { echo $CORES | tr ' ' '\n' | sed -n "${1}p"; }
ncores() { echo $CORES | wc -w; }
# gr G                  path of a DIMACS road network
gr() { echo $BENCH_DATA/gr/USA-road-t_$1.gr; }

TASK=${TASKS[$I]}
echo "task: $TASK"
eval "$TASK"
echo "TASK $TARGET $I DONE"
