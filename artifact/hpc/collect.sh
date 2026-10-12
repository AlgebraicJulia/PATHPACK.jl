#!/bin/bash
# Merge one run's raw results into results/*.csv (data/ holds the reference measurements; compare: scripts/regress.py).  Runs automatically after each target (see run.sh);
# safe to rerun by hand:  ./hpc/collect.sh raw/TARGET/RUN
set -eo pipefail
source "${HPC_DIR:-$(dirname "$0")}/env.sh"
RAW=$(cd "$1" && pwd)
mkdir -p $REPO/results
case $RAW in   # two targets with files of their own
  */gpu_semirings/*) python3 $REPO/bench/semirings/build_csv.py "$(dirname $RAW)" $REPO/results; exit 0 ;;
  */gpu_timeline/*)  cp $RAW/task_0/out/timeline/trace.csv $REPO/results/gpu_timeline_trace.csv; echo "results/gpu_timeline_trace.csv"; exit 0 ;;
esac
DIRS=$(ls -d $RAW/task_*/out 2>/dev/null | paste -sd: || true)
[ -n "$DIRS" ] || { echo "no results in $RAW"; exit 1; }
mkdir -p $REPO/results
python3 $REPO/scripts/collect.py --out $REPO/results --results $DIRS --scale $DIRS
LOGS=$(ls $RAW/task_*/out/baselines_*.txt 2>/dev/null || true)
[ -n "$LOGS" ] && python3 $REPO/scripts/parse_baselines.py --out $REPO/results $LOGS
echo; echo "tasks that did not finish:"
for f in $RAW/slurm_*.log $RAW/run.log; do [ -f $f ] || continue; grep -q "DONE" $f || echo "  $(basename $f): $(tail -1 $f)"; done
echo "(none listed = all finished)"
