#!/bin/bash
# The reference one-GPU runs (ours, ROME, and the speedups over ROME): two jobs, A and B, each on a GPU node of its own
# (every core, one GPU; the task pinned to 16 cores local to that GPU): ours (ours-auto and ours-auto-cpufactor: ordering
# auto, hybrid and CPU-only factorization, 11 cold calls each, checked) on its graphs, then ROME's best build per graph
# (RB below: its fastest passing build in earlier runs), 11 runs, then ROME's checks against ours.
#   hpc/reference/one_gpu.sh [--dry-run]      (AFTER=jobid: start after that job)
set -eo pipefail
source "$(dirname "$0")/../env.sh"
export HPC_DIR=$HPC
GPART=${GPU_PARTITION:-${GPU_PARTITIONS%% *}}; gv=GPU_GRES_${GPART//-/_}; GRES=${!gv:-b200}   # one GPU partition and its GPU type
NODE_CORES=${NODE_CORES:-112}         # every core of one GPU node (the reference runs: 112 on an 8 x B200 node), so no job shares it
mkdir -p $REPO/raw
declare -A RB=([144]=leaf800 [598a]=default [CurlCurl_0]=leaf400 [Dubcova3]=default [EB_16384_64]=leaf400 [EB_8192_256]=leaf800
  [G67]=default [OPF_6000]=default [USpowerGrid]=leaf400 [c-42]=default [c-69]=leaf400 [c-73]=leaf400 [cont-300]=leaf800
  [delaunay_n13]=leaf400 [delaunay_n14]=leaf400 [delaunay_n16]=leaf400 [email-Enron]=leaf400 [fe_ocean]=leaf800 [fe_sphere]=leaf400
  [fe_tooth]=leaf800 [finan512]=leaf800 [hypercube_14]=leaf800 [lpl1]=leaf800 [luxembourg_osm]=leaf400 [nd6k]=leaf400 [net4-1]=leaf800
  [oilpan]=leaf400 [onera_dual]=leaf800 [rgg2d_14]=leaf800 [rgg3d_14]=leaf800 [t60k]=leaf400 [usroads-48]=leaf400 [wave]=leaf800
  [web-sk-2005]=leaf400 [wing]=leaf800)
NB='{ date -Is; squeue -w $(hostname -s) -o "%.10i %.9u %.20j %.12b %.4C %.10M"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; } >> $OUT/../neighbours.txt 2>&1'
romepart() {   # the ROME runs and checks of these graphs (those with a passing build)
  local s="" gs=""; for g in "$@"; do [ -n "${RB[$g]}" ] || continue; s="$s ROME_NREP=11 ROME_VARIANTS=${RB[$g]} rome_runs $g;"; gs="$gs $g"; done
  echo "$NB;$s $NB; jlg gpu_baselines.jl rome$gs || true; $NB"
}
# job A: 24 of the 26 all-pairs graphs; job B: the two CPU-heavy dense ones (cpufactor) with the large set (GPU-heavy)
A_G=$(for g in $APSP_ALL; do case $g in EB_16384_64|rgg3d_14) ;; *) printf "%s " $g ;; esac; done)
B_G1="EB_16384_64 rgg3d_14 598a web-sk-2005 usroads-48 fe_ocean"; B_G2="144 Dubcova3 wave c-73 cont-300 thermomech_dM"
TA="$NB; BENCH_REPS=11 BENCH_VARIANTS=\"auto:hybrid auto:cpu\" jlg gpu_apsp.jl $A_G || true; $(romepart $A_G)"
TB="$NB; BENCH_REPS=11 BENCH_VARIANTS=\"auto:hybrid auto:cpu\" jlg gpu_apsp.jl $B_G1 || true; $NB; BENCH_REPS=11 BENCH_VARIANTS=auto:hybrid jlg gpu_apsp.jl $B_G2 || true; $(romepart $B_G1 $B_G2)"
for job in A:"$TA":01:45:00 B:"$TB":02:15:00; do
  name=${job%%:*}; rest=${job#*:}; task=${rest%:*:*:*}; time=${rest#"$task":}
  RUN=$(date +%Y%m%d-%H%M%S)-$$${name}_$(libcommit); RAW=$REPO/raw/gpu_apsp/${RUN}_b200
  echo "== job $name ($time): $RAW"; echo "$task" | tr ';' '\n' | grep -v '^ *{' | sed 's/^ */   /'
  [ "$1" = --dry-run ] && continue
  mkdir -p $RAW
  j=$(GPU_TASK="$task" GPU_TASK_ID=1 GPU_CPUS=16 GPU_PIN_LOCAL=16 sbatch --parsable $SLURM_AQ --partition=$GPART --nodes=1 \
      ${AFTER:+--dependency=afterok:$AFTER} --gpus=$GRES:1 --ntasks=1 --cpus-per-task=$NODE_CORES --mem=256gb --time=$time --job-name=final-$name --output=$RAW/slurm_1.log $HPC/gpu_job.sh gpu_one $RAW)
  c=$(sbatch --parsable $SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4gb --time=00:20:00 \
      --job-name=collect_final --dependency=afterany:$j --output=$RAW/collect.log --wrap="mkdir -p $REPO/results; flock $REPO/results/.collect.lock $HPC/collect.sh $RAW")
  echo "   job $j, collect $c"; echo "$name $j $c $RAW" >> $REPO/raw/JOBS
done
