#!/bin/bash
# The reference dense-kernel rates: bench/gpu_kernels.jl (min-plus GEMM, triangular solves, triangular inverse
# against cuBLAS and cuSOLVER, Float32) on one GPU of a node of its own, the task pinned to 16 GPU-local cores.
#   hpc/reference/kernels.sh [--dry-run]       (AFTER=jobid: start after that job)
set -eo pipefail
source "$(dirname "$0")/../env.sh"
export HPC_DIR=$HPC
GPART=${GPU_PARTITION:-${GPU_PARTITIONS%% *}}; gv=GPU_GRES_${GPART//-/_}; GRES=${!gv:-b200}   # one GPU partition and its GPU type
NODE_CORES=${NODE_CORES:-112}         # every core of one GPU node (the reference runs: 112 on an 8 x B200 node), so no job shares it
mkdir -p $REPO/raw
RUN=$(date +%Y%m%d-%H%M%S)_$(libcommit); RAW=$REPO/raw/gpu_kernels/${RUN}_b200
echo "== kernels job: $RAW"
[ "$1" = --dry-run ] && exit 0
mkdir -p $RAW
j=$(GPU_TASK_ID=0 GPU_CPUS=16 GPU_PIN_LOCAL=16 sbatch --parsable $SLURM_AQ --partition=$GPART --nodes=1 ${AFTER:+--dependency=afterany:$AFTER} \
    --gpus=$GRES:1 --ntasks=1 --cpus-per-task=$NODE_CORES --mem=64gb --time=01:00:00 --job-name=final-kernels --output=$RAW/slurm_0.log $HPC/gpu_job.sh gpu_kernels $RAW)
c=$(sbatch --parsable $SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4gb --time=00:20:00 \
    --job-name=collect_final --dependency=afterany:$j --output=$RAW/collect.log --wrap="mkdir -p $REPO/results; flock $REPO/results/.collect.lock $HPC/collect.sh $RAW")
echo "   job $j, collect $c"; echo "K $j $c $RAW" >> $REPO/raw/JOBS
