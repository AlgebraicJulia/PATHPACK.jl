#!/bin/bash
# The GPU targets as many small independent jobs, all submitted at once (for when the results are needed soon and the
# group's GPUs are free): one job per (graph, code) on one GPU, per (graph, GPU count) for several GPUs, each asking
# for exactly the GPUs, cores, memory and time it needs. Same scripts, checks and result files as run_gpu.sh's targets;
# each job is one task of gpu_job.sh (target gpu_one), and a collect job per partition and target merges them.
#
#   ./hpc/run_gpu_split.sh fapbatch [--gpu P] --ngpu N --items "k:G ..." [--ftime T] [--fmem M] [--pre CMDS] [--prefix ENV]
#   ./hpc/run_gpu_split.sh rome     [--gpu P] --base "G..." [--otime T]
#   ./hpc/run_gpu_split.sh apsp     [--gpu P] [--ours "G..."] [--base "G..."] [--codes "rome cugraph ..."] [--dry-run]
#   ./hpc/run_gpu_split.sh multi    [--gpu P] [--groups "G G,G G,..."] [--ks "1 2 ... 8"] [--mtime T]
#   ./hpc/run_gpu_split.sh fastapsp [--gpu P] [--graphs "G..."] [--ks "1 2 4 8"] [--after-multi FILE] [--ftime T]
#
#   apsp:     ours (gpu_apsp.jl: all variants) per graph in --ours; ROME and each other baseline per graph in --base, after
#             ours on that graph and GPU when ours is submitted here (its row hashes and fingerprints check them)
#   multi:    ours on 1..8 GPUs (gpu_multigpu.jl), one 8-GPU job per group of graphs, all counts in one process
#   fastapsp: Fast-APSP on 1, 2, 4, 8 GPUs, one 8-GPU job per graph (K from the earlier tuning when known, FAP_KPRIOR;
#             else swept), after ours on that graph (--after-multi FILE: the JOBS file of a multi submission)
set -eo pipefail
source "$(dirname "$0")/env.sh"
export HPC_DIR=$HPC
WHAT=$1; shift
DRY=0; PARTS=(); OURS=""; BASE=""; CODES="rome cugraph gunrock adds fw ecl fastapsp1"; GRAPHS=""; KS=""; AFTERM=""; MGROUPS=""; MTIME=""; FTIME=""; RAWS=""; AFTER=""; OTIME=""; PREFIX=""; ITEMS=""; NGPU=""; FMEM=""; PRE=""; NOPARSE=""
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;; --gpu) PARTS+=("$2"); shift ;; --ours) OURS=$2; shift ;; --base) BASE=$2; shift ;; --codes) CODES=$2; shift ;;
    --graphs) GRAPHS=$2; shift ;; --ks) KS=$2; shift ;; --after-multi) AFTERM=$2; shift ;; --groups) MGROUPS=$2; shift ;;
    --mtime) MTIME=$2; shift ;; --ftime) FTIME=$2; shift ;; --raws) RAWS=$2; shift ;; --prefix) PREFIX="$2 "; shift ;; --otime) OTIME=$2; shift ;; --after) AFTER=$2; shift ;;
    --noparse) NOPARSE=1 ;;   # (fapbatch: no Julia in the GPU job; parse later with recheck)
    --items) ITEMS=$2; shift ;; --ngpu) NGPU=$2; shift ;; --fmem) FMEM=$2; shift ;; --pre) PRE=$2; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac; shift
done
[ ${#PARTS[@]} -eq 0 ] && PARTS=($GPU_PARTITIONS)
COMMIT=$(libcommit); [ "$(libdirty)" = true ] && COMMIT=$COMMIT-dirty
RUN=$(date +%Y%m%d-%H%M%S)-$$_$COMMIT          # (the pid: submissions in the same second get folders of their own)
export FAP_KPRIOR=${FAP_KPRIOR:-$REPO/baselines/gpu/fastapsp_kprior.txt}   # Fast-APSP's K per graph, GPU and count, as tuned for the reference runs

# n of a graph: from its exported file, else bench/graphs.jl's reader is needed (unknown: 0)
nof() { local f=$BENCH_EXPORT/$1.general.mtx; [ -f $f ] || f=$BENCH_EXPORT/$1.rome.mtx; [ -f $f ] && grep -v '^%' $f | head -1 | awk '{print $1}' || echo 0; }
declare -A NKNOWN=([598a]=110971 [luxembourg_osm]=114599 [web-sk-2005]=121422 [usroads-48]=126146 [fe_ocean]=143437 [t60k]=60005 [wing]=62032
  [delaunay_n16]=65536 [c-69]=67458 [oilpan]=73752 [finan512]=74752 [fe_tooth]=78136 [onera_dual]=85567 [net4-1]=88343
  [CurlCurl_0]=11083 [EB_16384_64]=16384 [EB_8192_256]=8192 [G67]=10000 [OPF_6000]=29902 [USpowerGrid]=4941 [c-42]=10471
  [delaunay_n13]=8192 [delaunay_n14]=16384 [email-Enron]=33696 [fe_sphere]=16386 [hypercube_14]=16384 [lpl1]=32460 [nd6k]=18000
  [rgg2d_14]=16384 [rgg3d_14]=16384 [delaunay_n17]=131072 [rgg_n_2_17_s0]=131072 [USA-t-NY]=264346 [USA-t-BAY]=321270 [USA-t-COL]=435666)
gn() { local n=${NKNOWN[$1]:-}; [ -n "$n" ] && echo $n || nof $1; }
EXP='[ -f $BENCH_EXPORT/G.general.mtx ] && [ -f $BENCH_EXPORT/G.probes ] || BENCH_EXPORT_ONLY=1 jlg gpu_apsp.jl G; '   # (graph files first)

ID=0; declare -A JOBS
sub() {   # sub ID PART RAW NAME NGPU NTASKS CPUS MEM TIME DEP TASK   -> job id (ID: the task's number in RAW)
  local id=$1 part=$2 raw=$3 name=$4 ng=$5 nt=$6 cpus=$7 mem=$8 time=$9 dep=${10} task=${11}
  local gv=GPU_GRES_${part//-/_}; local gres=${!gv}
  if [ $DRY = 1 ]; then echo "  [$part] #$id $name gpus=$ng tasks=$nt cpus=$cpus mem=$mem time=$time ${dep:+dep=$dep} :: $task" >&2; echo DRY$id; return; fi
  GPU_TASK="$task" GPU_TASK_ID=$id GPU_CPUS=$cpus sbatch --parsable $SLURM_AQ --partition=$part --nodes=1 --gpus=$gres:$ng \
    --ntasks=$nt --cpus-per-task=$((cpus / nt)) --mem=$mem --time=$time --job-name=$name ${dep:+--dependency=$dep} \
    --output=$raw/slurm_$id.log $HPC/gpu_job.sh gpu_one $raw
}
collect() {   # collect PART RAW JOBIDS
  [ $DRY = 1 ] && return; [ -z "$3" ] && return
  sbatch --parsable $SLURM_AQ --partition=$PARTITION --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4gb --time=00:20:00 \
    --job-name=collect_split --dependency=afterany:$3 --output=$2/collect.log --wrap="mkdir -p $REPO/results; flock $REPO/results/.collect.lock $HPC/collect.sh $2"
}
for part in "${PARTS[@]}"; do
  k=$part; ALL=""
  case $WHAT in
  ours)
    # ours on one GPU: one job running every graph of --ours (all variants) in one Julia process, smallest first
    RAW=$REPO/raw/gpu_apsp/${RUN}_$k; mkdir -p $RAW
    ID=$((ID + 1)); j=$(sub $ID $part $RAW "ours1" 1 1 8 48gb ${OTIME:-02:00:00} "" "${PREFIX}jlg gpu_apsp.jl $OURS"); ALL=$ALL:$j ;;
  apsp)
    RAW=$REPO/raw/gpu_apsp/${RUN}_$k; mkdir -p $RAW
    declare -A OJ=()
    for g in $OURS; do
      ID=$((ID + 1)); j=$(sub $ID $part $RAW "go_$g" 1 1 8 48gb 01:30:00 "" "jlg gpu_apsp.jl $g"); OJ[$g]=$j; ALL=$ALL:$j
    done
    for g in $BASE; do
      n=$(gn $g); dep=${OJ[$g]:+afterany:${OJ[$g]}}; e=${EXP//G/$g}
      for c in $CODES; do
        case $c in
          rome) ID=$((ID + 1)); j=$(sub $ID $part $RAW "rome_$g" 1 1 16 32gb 00:45:00 "" "${e}rome_runs $g; jlg gpu_baselines.jl rome $g || true") ;;
          fastapsp1) [ $n -gt 0 ] && [ $n -lt 60000 ] || continue
                     ID=$((ID + 1)); j=$(sub $ID $part $RAW "fap1_$g" 1 1 16 56gb 00:45:00 "$dep" "${e}OTHERS=$c others_runs $g; OTHERS=$c jlg gpu_baselines.jl others $g || true") ;;
          fw|ecl) ID=$((ID + 1)); j=$(sub $ID $part $RAW "${c}_$g" 1 1 8 48gb 01:00:00 "$dep" "${e}OTHERS=$c others_runs $g; OTHERS=$c jlg gpu_baselines.jl others $g || true") ;;
          *)  ID=$((ID + 1)); j=$(sub $ID $part $RAW "${c}_$g" 1 1 8 48gb 00:45:00 "$dep" "${e}OTHERS=$c others_runs $g; OTHERS=$c jlg gpu_baselines.jl others $g || true") ;;
        esac
        ALL=$ALL:$j
      done
    done ;;
  rome)
    # ROME on several graphs (--base) in one 1-GPU job: the same runs and checks as apsp's per-graph ROME jobs, one after
    # another (a re-run of its timings, e.g. when co-located jobs made them noisy); ROME_NREP, ROME_VARIANTS as usual
    RAW=$REPO/raw/gpu_apsp/${RUN}_$k; mkdir -p $RAW; e=""
    for g in $BASE; do e="$e${EXP//G/$g}"; done
    ID=$((ID + 1)); j=$(sub $ID $part $RAW "rome_batch" 1 1 16 48gb ${OTIME:-01:00:00} "" "${e}${PREFIX}rome_runs $BASE; jlg gpu_baselines.jl rome $BASE || true"); ALL=$ALL:$j ;;
  multi)
    # one 8-GPU job per group of graphs (--groups "G1 G2,G3 G4,..."; default: the two multi-GPU sets), one Julia process
    # (compiled once) running every count in --ks for each graph, smallest graphs first
    RAW=$REPO/raw/gpu_multigpu/${RUN}_$k; mkdir -p $RAW
    IFS=',' read -ra GROUPS_ <<< "${MGROUPS:-$MULTI_GRAPHS}"
    for grp in "${GROUPS_[@]}"; do
      ID=$((ID + 1)); j=$(sub $ID $part $RAW "gm_$(echo $grp | awk '{print $1}')" 8 1 24 64gb ${MTIME:-01:00:00} "" "${PREFIX}BENCH_NGPUS=\"${KS:-1 2 3 4 5 6 7 8}\" jlg gpu_multigpu.jl $grp")
      ALL=$ALL:$j; for g in $grp; do echo "$part $g $j" >> $RAW/JOBS; done
    done ;;
  fastapsp)
    # one job per graph and count k, with exactly k GPUs (1 GPU: fap_driver; k GPUs: k MPI ranks); its compiled startup
    # is seconds. K from the earlier tuning when known (FAP_KPRIOR), else swept; checked against ours afterwards (recheck)
    RAW=$REPO/raw/gpu_fastapsp/${RUN}_$k; mkdir -p $RAW
    for g in ${GRAPHS:-$FASTAPSP}; do for kk in ${KS:-1 2 4 8}; do
      n=$(gn $g); e=${EXP//G/$g}
      if [ $kk = 1 ]; then nt=1; cpus=16; mem=$(( 13 * n / 1000 * n / 1000000 + 24 ))gb; t=00:50:00
      else nt=$kk; cpus=$((4 * kk)); mem=$(( 10 * n / 1000 * n / 1000000 + 24 ))gb; t=00:30:00; fi
      ID=$((ID + 1)); j=$(sub $ID $part $RAW "fap${kk}_$g" $kk $nt $cpus $mem ${FTIME:-$t} "" "${e}${PREFIX}fap_runs $kk $g; jlg gpu_baselines.jl fastapsp $g || true"); ALL=$ALL:$j
    done; done ;;
  fapbatch)
    # Fast-APSP at several (count, graph) points in one job of --ngpu GPUs (--items "k:G k:G ...", run in that order, each
    # k <= ngpu ranks on k of the job's GPUs, one at a time, FAP_THREADS cores per rank), its results parsed once at
    # the end; --pre: commands run first (e.g. a fit-only pass of gpu_multigpu.jl)
    RAW=$REPO/raw/gpu_fastapsp/${RUN}_$k; mkdir -p $RAW; e=""; runs=""; gs=""
    for it in $ITEMS; do kk=${it%%:*}; g=${it#*:}; runs="$runs fap_runs $kk $g;"
      case " $gs " in *" $g "*) ;; *) gs="$gs $g"; e="$e${EXP//G/$g}" ;; esac; done
    ID=$((ID + 1)); j=$(sub $ID $part $RAW "fapbatch" ${NGPU:-8} ${NGPU:-8} $((4 * ${NGPU:-8})) ${FMEM:-500gb} ${FTIME:-04:00:00} "" "${e}${PRE:+$PRE; }export FAP_THREADS=4; ${PREFIX}${runs} ${NOPARSE:+true ||} jlg gpu_baselines.jl fastapsp$gs || true"); ALL=$ALL:$j ;;
  recheck)
    # the baselines' checks again, once ours has written its row hashes and fingerprints (gpu_baselines.jl over every
    # task folder of the given raw runs of this partition), in one job on one GPU of the partition (compiled once), after
    # the given jobs; then each raw run collected again. Paths absolute: the Julia scripts run in bench/.
    RR=""; for raw in $RAWS; do case $raw in *_$k) RR="$RR $(cd $raw && pwd)" ;; esac; done
    [ -n "$RR" ] || continue
    # (the task folders are listed when the job runs: those of jobs still pending do not exist yet)
    RAW=$(echo $RR | awk '{print $1}')   # (task number: past every task folder there, so nothing is overwritten)
    ID=$(( $(ls -d $RAW/task_* 2>/dev/null | sed 's/.*task_//' | sort -n | tail -1) + 1 ))
    j=$(sub $ID $part $RAW "recheck" 1 1 8 32gb ${OTIME:-00:45:00} "${AFTER:+afterany:$AFTER}" "BENCH_RECHECK_DIRS=\$(ls -d $(for r in $RR; do printf '%s/task_*/out ' $r; done)| paste -sd:) jlg gpu_baselines.jl recheck; for r in $RR; do flock $REPO/data/.collect.lock $HPC/collect.sh \$r; done")
    echo "recheck on $part: job $j over$RR"; continue ;;
  *) echo "what: ours | apsp | rome | multi | fastapsp | fapbatch | recheck"; exit 1 ;;
  esac
  col=$(collect $part $RAW ${ALL#:})
  echo "$WHAT on $part: $(echo ${ALL#:} | tr ':' '\n' | grep -c .) jobs (${ALL#:}), collect ${col:-none}; raw: $RAW"
done
