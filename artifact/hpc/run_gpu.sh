#!/bin/bash
# Rerun the GPU experiments on a Slurm cluster (the GPU counterpart of run.sh; one-time setup: hpc/setup_gpu.sh).
#
#   ./hpc/run_gpu.sh                              list the GPU targets
#   ./hpc/run_gpu.sh gpu_kernels                  on every partition in GPU_PARTITIONS (config.sh)
#   ./hpc/run_gpu.sh gpu_apsp --gpu gpu           on one partition
# Options:
#   --gpu PARTITION   only this GPU partition (repeatable)
#   --dry-run         print what would be submitted, submit nothing
#   --tasks 0,3       only these tasks of the target
#   --throttle N      at most N tasks of the target at once per partition (default 2)
#   --after JOB[:JOB] start only after these jobs succeeded (e.g. the gpu_tests job)
#
# Each target's tasks run as one Slurm array per partition (GPUS GPUs, CPUS cores, MEM, TIME from the target file);
# targets with SINGLETON=1 (whole-node, 8-GPU jobs) share one job name, so Slurm runs them one at a time. A collect job
# then merges the results into results/gpu_*.csv (scripts/collect.py, as for the CPU targets). There is no gate: the GPU
# scripts check every result they report (gpu_lib.jl), and the CPU gate does not apply to Float32 GPU results.
set -eo pipefail
source "$(dirname "$0")/env.sh"
export HPC_DIR=$HPC
DRY=0; ONLY=""; THROTTLE=2; PARTS=(); TARGETS=(); AFTER=""
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;; --tasks) ONLY=$2; shift ;; --gpu) PARTS+=("$2"); shift ;; --throttle) THROTTLE=$2; shift ;; --after) AFTER=$2; shift ;;
    -*) echo "unknown option $1"; exit 1 ;; *) TARGETS+=("$1") ;;
  esac; shift
done
if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "usage: ./hpc/run_gpu.sh TARGET... [--gpu PARTITION] [--dry-run] [--tasks 0,3] [--throttle N]"; echo; echo "GPU targets:"
  for f in $HPC/targets/gpu_*.sh; do printf "  %-18s %s\n" $(basename $f .sh) "$(head -1 $f | sed 's/^# //')"; done
  exit 0
fi
[ ${#PARTS[@]} -eq 0 ] && PARTS=($GPU_PARTITIONS)
for t in "${TARGETS[@]}"; do [ -f $HPC/targets/$t.sh ] || { echo "no such target: $t"; exit 1; }; done
[ -f $LIB_GPU/Project.toml ] || { echo "library not found at $LIB_GPU"; exit 1; }
[ -f $BENCH_GPU/Manifest.toml ] || { echo "run ./hpc/setup_gpu.sh julia first"; exit 1; }
COMMIT=$(libcommit); [ "$(libdirty)" = true ] && COMMIT=$COMMIT-dirty
RUN=$(date +%Y%m%d-%H%M%S)_$COMMIT
submit() { if [ $DRY = 1 ]; then echo "  sbatch $*" >&2; echo DRYRUN; else sbatch --parsable "$@"; fi; }
echo "GPU library commit $COMMIT, run $RUN"
for t in "${TARGETS[@]}"; do
  ( source $HPC/env.sh; source $HPC/targets/$t.sh; echo ${#TASKS[@]} "${TIME:-01:00:00}" "${MEM:-64gb}" "${GPUS:-1}" "${CPUS:-$GPU_CPUS}" "${NTASKS:-1}" "${SINGLETON:-0}" ) > ${TMPDIR:-/tmp}/.target_$$
  read N TIME MEM NG CPUS NTASKS SINGLE < ${TMPDIR:-/tmp}/.target_$$; rm -f ${TMPDIR:-/tmp}/.target_$$
  ARR=${ONLY:-0-$((N-1))}
  for part in "${PARTS[@]}"; do
    gv=GPU_GRES_${part//-/_}; gres=${!gv}; [ -n "$gres" ] || { echo "no GPU type for partition $part (config.sh: $gv)"; exit 1; }
    RAW=$REPO/raw/$t/${RUN}_${part}; mkdir -p $RAW
    NAME=$t; [ $SINGLE = 1 ] && NAME=gpu_node
    PLACE="$SLURM_AQ --partition=$part --nodes=1 --gpus=$gres:$NG --ntasks=$NTASKS --cpus-per-task=$((CPUS / NTASKS)) --mem=$MEM"
    D=""; TH=$THROTTLE; [ $SINGLE = 1 ] && { D="singleton"; TH=1; }
    [ -n "$AFTER" ] && D="${D:+$D,}afterok:$AFTER"
    DEP=; [ -n "$D" ] && DEP="--dependency=$D --kill-on-invalid-dep=yes"
    if [ $DRY = 1 ]; then
      echo "target $t on $part: $N tasks, $NG GPU(s), $CPUS cores, $MEM, $TIME"
      ( source $HPC/env.sh; source $HPC/targets/$t.sh; for i in "${!TASKS[@]}"; do echo "  task $i: ${TASKS[$i]}"; done )
    fi
    JOB=$(submit $PLACE --time=$TIME --job-name=$NAME --array=$ARR%$TH $DEP --output=$RAW/slurm_%a.log $HPC/gpu_job.sh $t $RAW)
    COL=$(submit $SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4gb \
          --time=00:15:00 --job-name=collect_$t --dependency=afterany:$JOB --output=$RAW/collect.log \
          --wrap="mkdir -p $REPO/results; flock $REPO/results/.collect.lock $HPC/collect.sh $RAW")          # (one collect at a time writes data/)
    echo "target $t on $part: tasks $ARR -> job $JOB, then collect job $COL; raw results in raw/$t/${RUN}_${part}"
  done
done
[ $DRY = 1 ] && { for t in "${TARGETS[@]}"; do rmdir $REPO/raw/$t/${RUN}_* 2>/dev/null || true; done; echo "(dry run: nothing submitted)"; }
exit 0
