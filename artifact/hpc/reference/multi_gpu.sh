#!/bin/bash
# The reference several-GPU runs: one job per GPU count g = 1..8 holding exactly g GPUs (no idle GPUs), each
# running gpu_multigpu.jl (ordering auto, 8 Julia threads; whole call, closure alone, new weights) on the graphs that fit
# on g GPUs: Fast-APSP's 5, the 4 largest all-pairs graphs, the 3 road networks, delaunay_n19.
#   hpc/reference/multi_gpu.sh [--dry-run]     (COUNTS="1 2 8": only these counts)
set -eo pipefail
source "$(dirname "$0")/../env.sh"
export HPC_DIR=$HPC
GPART=${GPU_PARTITION:-${GPU_PARTITIONS%% *}}; gv=GPU_GRES_${GPART//-/_}; GRES=${!gv:-b200}   # one GPU partition and its GPU type
NODE_CORES=${NODE_CORES:-112}         # every core of one GPU node (the reference runs: 112 on an 8 x B200 node), so no job shares it
mkdir -p $REPO/raw
NB='{ date -Is; squeue -w $(hostname -s) -o "%.10i %.9u %.20j %.12b %.4C %.10M"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; } >> $OUT/../neighbours.txt 2>&1'
BASE="finan512 fe_tooth onera_dual net4-1 598a luxembourg_osm web-sk-2005 usroads-48 fe_ocean"
RUN=$(date +%Y%m%d-%H%M%S)-$$_$(libcommit); RAW=$REPO/raw/gpu_multigpu/${RUN}_b200; [ "$1" = --dry-run ] || mkdir -p $RAW
ALL=""
for g in ${COUNTS:-1 2 3 4 5 6 7 8}; do
  G="$BASE"; [ $g -ge 2 ] && G="$G USA-t-NY"; [ $g -ge 3 ] && G="$G USA-t-BAY"; [ $g -ge 5 ] && G="$G USA-t-COL"; [ $g -ge 7 ] && G="$G delaunay_n19"
  task="$NB; export BENCH_MORDERING=auto GPU_JULIA_THREADS=8; BENCH_NGPUS=$g jlg gpu_multigpu.jl $G || true; $NB"
  t=00:30:00; [ $g -ge 7 ] && t=00:40:00
  echo "g=$g ($t): $G"
  [ "$1" = --dry-run ] && continue
  j=$(GPU_TASK="$task" GPU_TASK_ID=$g GPU_CPUS=24 sbatch --parsable $SLURM_AQ --partition=$GPART --nodes=1 ${DEP:+--dependency=$DEP} --gpus=$GRES:$g \
      --ntasks=1 --cpus-per-task=24 --mem=128gb --time=$t --job-name=final-mg$g --output=$RAW/slurm_$g.log $HPC/gpu_job.sh gpu_one $RAW)
  ALL=$ALL:$j; echo "   job $j"
done
[ "$1" = --dry-run ] && exit 0
c=$(sbatch --parsable $SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4gb --time=00:20:00 \
    --job-name=collect_final --dependency=afterany${ALL} --output=$RAW/collect.log --wrap="mkdir -p $REPO/results; flock $REPO/results/.collect.lock $HPC/collect.sh $RAW")
echo "multi $ALL collect $c $RAW" | tee -a $(dirname $0)/JOBS
