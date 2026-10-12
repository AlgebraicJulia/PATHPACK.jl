#!/bin/bash
# One task of a GPU target (submitted by run_gpu.sh; do not call directly). Arguments: TARGET RAW_DIR
# A GPU node is not split into NUMA regions like the CPU runs: the task gets GPU_CPUS cores (or what the target asks
# for) on one node with its GPUs; the cores, the host CPU, the GPUs and their interconnect are recorded.
set -eo pipefail
TARGET=$1; RAW=$2; I=${GPU_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}   # (GPU_TASK_ID: the split jobs of run_gpu_split.sh)
source "${HPC_DIR:-$(dirname "$0")}/env.sh"
OUT=$RAW/task_$I/out; mkdir -p $OUT
export BENCH_EXPORT=${BENCH_EXPORT:-$WORK/gpu-export}; mkdir -p $BENCH_EXPORT
source "$HPC/gpu_baselines.sh"
source "$HPC/targets/$TARGET.sh"                   # defines TASKS (GPUS, CPUS, MEM, TIME: see run_gpu.sh)
# GPU_PIN_LOCAL=N (e.g. on a node of our own: every core, one GPU; --exclusive would hold all 8 GPUs): the task on N cores local to its first GPU (the PCI device's
# local_cpulist), past the first 8 of them (system daemons, interrupts), instead of every core the job holds
if [ -n "${GPU_PIN_LOCAL:-}" ]; then
  BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed -n 1p | sed 's/^0000//' | tr A-F a-f)
  PIN=$(python3 -c 'import sys
L = lambda s: [x for p in s.split(",") for a, _, b in [p.partition("-")] for x in range(int(a), int(b or a) + 1)]
n = int(sys.argv[2]); allowed = set(L(sys.argv[3])); c = [x for x in L(sys.argv[1]) if x in allowed]   # (only cores we hold)
c = c[8:] if len(c) >= 8 + n else c
print(",".join(map(str, (c if len(c) >= n else sorted(allowed))[:n])))' "$(cat /sys/bus/pci/devices/$BUS/local_cpulist)" $GPU_PIN_LOCAL "$(taskset -cp $$ | sed 's/.*: //')")
  taskset -cp $PIN $$ > /dev/null
fi
COMMIT=$(libcommit)
DIRTY=$(libdirty)
CORES=$(taskset -cp $$ | sed 's/.*: //')
GPUS_SEEN=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | sed 's/^ *//' | paste -sd';')
nvidia-smi topo -m > $RAW/task_$I/topology.txt 2>&1 || true
nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,power.limit,pci.bus_id --format=csv > $RAW/task_$I/gpus.csv 2>&1 || true
cat > $RAW/task_$I/PROVENANCE.json << JSON
{"commit": "$COMMIT", "commit_dirty": $DIRTY, "machine": "$(lscpu | sed -n 's/^Model name: *//p' | head -1)",
 "gpus": "$GPUS_SEEN", "host": "$(hostname -s)", "cores": "$CORES", "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "job": "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$I", "slurm_job": "$SLURM_JOB_ID", "partition": "${SLURM_JOB_PARTITION:-}", "target": "$TARGET"}
JSON
export BENCH_PLACEMENT="cores $CORES; $GPUS_SEEN"
echo "target $TARGET, task $I, host $(hostname -s), cores $CORES, GPUs $GPUS_SEEN, library $COMMIT (dirty: $DIRTY)"

# ---- helpers used by the task lines in targets/gpu_*.sh ----
# jlg SCRIPT ARGS...   a GPU benchmark (bench/gpu environment), GPU_JULIA_THREADS threads, results into $OUT
jlg() { ( cd $BENCH && BENCH_RESULTS=$OUT BENCH_NT=$GPU_JULIA_THREADS SEMIRINGGPU_TUNE=${SEMIRINGGPU_TUNE:-off} \
          $JULIA -t $GPU_JULIA_THREADS,0 --project=$BENCH_GPU "$@" ); }

TASK=${TASKS[$I]}
echo "task: $TASK"
eval "$TASK"
echo "TASK $TARGET $I DONE"
