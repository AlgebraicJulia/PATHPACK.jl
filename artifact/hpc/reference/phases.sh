#!/bin/bash
# The reference per-step times: one job on a GPU node of its own (one GPU, 16 GPU-local cores), gpu_phases.jl
# (ours-auto, 7 cold calls per graph under the step timer, per-step medians: code ours-auto-phases) on every graph of the
# one-GPU runs.   hpc/reference/phases.sh [--dry-run]
set -eo pipefail
source "$(dirname "$0")/../env.sh"
export HPC_DIR=$HPC
GPART=${GPU_PARTITION:-${GPU_PARTITIONS%% *}}; gv=GPU_GRES_${GPART//-/_}; GRES=${!gv:-b200}   # one GPU partition and its GPU type
NODE_CORES=${NODE_CORES:-112}         # every core of one GPU node (the reference runs: 112 on an 8 x B200 node), so no job shares it
mkdir -p $REPO/raw
NB='{ date -Is; squeue -w $(hostname -s) -o "%.10i %.9u %.20j %.12b %.4C %.10M"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; } >> $OUT/../neighbours.txt 2>&1'
G="$APSP_ALL 598a web-sk-2005 usroads-48 fe_ocean 144 Dubcova3 wave c-73 cont-300 thermomech_dM"
task="$NB; BENCH_PHREPS=7 jlg gpu_phases.jl $G || true; $NB"
RUN=$(date +%Y%m%d-%H%M%S)-$$Q_$(libcommit); RAW=$REPO/raw/gpu_apsp/${RUN}_b200
echo "== phases job: $RAW"; echo "$task"
[ "$1" = --dry-run ] && exit 0
mkdir -p $RAW
j=$(GPU_TASK="$task" GPU_TASK_ID=1 GPU_CPUS=16 GPU_PIN_LOCAL=16 sbatch --parsable $SLURM_AQ --partition=$GPART --nodes=1 \
    --gpus=$GRES:1 --ntasks=1 --cpus-per-task=$NODE_CORES --mem=256gb --time=01:00:00 --job-name=final-phases2 --output=$RAW/slurm_1.log $HPC/gpu_job.sh gpu_one $RAW)
c=$(sbatch --parsable $SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4gb --time=00:20:00 \
    --job-name=collect_final --dependency=afterany:$j --output=$RAW/collect.log --wrap="mkdir -p $REPO/results; flock $REPO/results/.collect.lock $HPC/collect.sh $RAW")
echo "   job $j, collect $c"; echo "Q $j $c $RAW" >> $REPO/raw/JOBS
