#!/bin/bash
# Run one or more of the CPU experiments on a Slurm cluster (one-time setup: hpc/setup.sh).
#
#   ./hpc/run.sh                     list the targets
#   ./hpc/run.sh queries             the single-source queries
#   ./hpc/run.sh dense diffcon       several targets at once
#   ./hpc/run.sh all                 everything
# Options:
#   --dry-run      print what would be submitted, submit nothing
#   --tasks 0,3    only these tasks of the target (e.g. to resubmit one that failed)
#
# What happens: a short gate job checks the library; the target's tasks run as region-sized jobs once the gate
# passes; a collect job then merges the results into results/*.csv (compare with the reference measurements in data/: scripts/regress.py).
set -eo pipefail
source "$(dirname "$0")/env.sh"
export HPC_DIR=$HPC          # sbatch runs a spooled copy of job.sh; this tells it where the repository is
DRY=0; ONLY=""; TARGETS=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;; --tasks) ONLY=$2; shift ;;
    -*) echo "unknown option $1"; exit 1 ;; *) TARGETS+=("$1") ;;
  esac; shift
done
ALL=$(for f in $HPC/targets/*.sh; do t=$(basename $f .sh); [ $t = gate ] || [[ $t == gpu_* ]] || grep -q "^WHOLE_NODE=1\|^REGIONS=" $f || echo $t; done)   # multi-region targets: submit them by name
if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "usage: ./hpc/run.sh TARGET... [--dry-run] [--tasks 0,3]"; echo; echo "targets:"
  for t in $ALL all; do case $t in all) d="every target above";; *) d=$(head -1 $HPC/targets/$t.sh | sed 's/^# //');; esac; printf "  %-15s %s\n" $t "$d"; done
  echo; echo "several NUMA regions (not in all; submit by name):"
  echo; echo "GPU targets: see ./hpc/run_gpu.sh"
  for f in $(grep -l "^REGIONS=" $HPC/targets/*.sh); do printf "  %-15s %s\n" $(basename $f .sh) "$(head -1 $f | sed 's/^# //')"; done
  exit 0
fi
[ "${TARGETS[*]}" = all ] && TARGETS=($ALL)
for t in "${TARGETS[@]}"; do [ -f $HPC/targets/$t.sh ] || { echo "no such target: $t (run ./hpc/run.sh to list them)"; exit 1; }; done
[ -f $LIB/Project.toml ] || { echo "library not found at $LIB"; exit 1; }
[ -f $BENCH/Manifest.toml ] || { echo "run ./hpc/setup.sh first"; exit 1; }
COMMIT=$(libcommit)
RUN=$(date +%Y%m%d-%H%M%S)_$COMMIT
SB="$SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=16 --ntasks-per-socket=1 --sockets-per-node=1 --cores-per-socket=16"
submit() { if [ $DRY = 1 ]; then echo "  sbatch $*" >&2; echo DRYRUN; else sbatch --parsable "$@"; fi; }
echo "library commit $COMMIT, run $RUN"

NEEDGATE=0; for t in "${TARGETS[@]}"; do grep -q "^NOGATE=1" $HPC/targets/$t.sh || NEEDGATE=1; done
GRAW=$REPO/raw/gate/$RUN; mkdir -p $GRAW; GATE=
if [ $NEEDGATE = 1 ]; then
  GATE=$(submit $SB --mem=64gb --time=00:30:00 --job-name=gate --array=0 --output=$GRAW/slurm_%a.log $HPC/job.sh gate $GRAW)
  echo "gate job: $GATE"
fi
for t in "${TARGETS[@]}"; do
  ( source $HPC/env.sh; source $HPC/targets/$t.sh; echo ${#TASKS[@]} "${TIME:-01:00:00}" "${MEM:-100gb}" "${WHOLE_NODE:-0}" "${REGIONS:-1}" ) > ${TMPDIR:-/tmp}/.target_$$
  read N TIME MEM WHOLE REG < ${TMPDIR:-/tmp}/.target_$$; rm -f ${TMPDIR:-/tmp}/.target_$$
  PLACE="$SB --mem=$MEM"                                           # one NUMA region
  [ $REG -gt 1 ] && PLACE="$SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=$((16 * REG)) --sockets-per-node=$REG --cores-per-socket=16 --distribution=block:block --mem=$MEM"   # block: fill a region before the next (default is cyclic)
  [ $WHOLE = 1 ] && PLACE="$SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=128 --exclusive --mem=950gb"
  ARR=${ONLY:-0-$((N-1))}
  RAW=$REPO/raw/$t/$RUN; mkdir -p $RAW
  if [ $DRY = 1 ]; then
    echo "target $t: $N tasks"
    ( source $HPC/env.sh; source $HPC/targets/$t.sh; for i in "${!TASKS[@]}"; do echo "  task $i: ${TASKS[$i]}"; done )
  fi
  DEP="--dependency=afterok:$GATE --kill-on-invalid-dep=yes"; grep -q "^NOGATE=1" $HPC/targets/$t.sh && DEP=
  JOB=$(submit $PLACE --time=$TIME --job-name=$t --array=$ARR $DEP \
        --output=$RAW/slurm_%a.log $HPC/job.sh $t $RAW)
  COL=$(submit $SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4gb \
        --time=00:15:00 --job-name=collect_$t --dependency=afterany:$JOB --output=$RAW/collect.log $HPC/collect.sh $RAW)
  echo "target $t: tasks $ARR -> job $JOB, then collect job $COL; raw results in raw/$t/$RUN"
done
[ $DRY = 1 ] && { rmdir $GRAW 2>/dev/null || true; for t in "${TARGETS[@]}"; do rmdir $REPO/raw/$t/$RUN 2>/dev/null || true; done; echo "(dry run: nothing submitted)"; exit 0; }
cat << MSG

Submitted. Watch with:   squeue -u \$USER
When the collect jobs are done, compare your results with the reference measurements:
  python3 scripts/regress.py    # results/*.csv against data/*.csv
MSG
