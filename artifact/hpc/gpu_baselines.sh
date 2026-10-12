# The GPU baselines, as task helpers (sourced by gpu_job.sh). Each writes its logs under $OUT; bench/gpu_baselines.jl
# then checks every run against the CPU's Float64 distances (or our row hashes) and turns the logs into result files.
# Graph files come from gpu_apsp.jl / gpu_multigpu.jl (export_graph): $BENCH_EXPORT/<graph>.rome.mtx, .general.mtx,
# .sample (the 64 checked vertices), .rowhash (integer weights: the hashes of every row of our result).
ROMEBIN=${ROMEBIN:-$BASELINES/rome/bin}   # (another build set, e.g. sm_89: ROMEBIN=$BASELINES/rome-l4/bin)
FAP=$FASTAPSP_BUILD

# ---- ROME (Luo et al., PPoPP'26): one process per run; its timers are tree build (Scotch ordering, elimination tree),
#      allocation and initialization (the n x n device matrix, the graph, the metadata buffers), computing. The result stays
#      on the device (ROME_DEVICE_ONLY: no host copy, outside its timers); run 1 of each build dumps the sampled rows.
#      Builds: default, and Scotch leaf sizes 400 and 800 (the tuning that helped it most); one instrumented run per graph
#      splits the allocation (cudaMallocPitch of the n x n matrix) from the initialization.
ROME_VARIANTS=${ROME_VARIANTS:-"default leaf400 leaf800"}
rome_one() {   # rome_one GRAPH VARIANT RUN
  local g=$1 v=$2 r=$3 d=$OUT/rome; mkdir -p $d
  local log=$d/$g.$v.r$r.log dump=
  [ $r = 1 ] && dump="ROME_DUMP_SAMPLE=$d/$g.$v.dump ROME_DUMP_SAMPLE_ROWS=$BENCH_EXPORT/$g.sample"
  local s=$(date +%s.%N)
  timeout --signal=KILL ${ROME_TIMEOUT:-1200} env ROME_DEVICE_ONLY=1 ROME_SKIP_CHECK=1 OMP_NUM_THREADS=$GPU_CPUS CUDA_MODULE_LOADING=EAGER $dump \
    /usr/bin/time -f "TIME maxrss_kb %M user_s %U sys_s %S wall_s %e" $ROMEBIN/rome_$v $BENCH_EXPORT/$g.rome.mtx ignore > $log 2>&1
  local rc=$?
  echo "PROCESS_WALL $(echo "$(date +%s.%N) - $s" | bc) rc=$rc" >> $log
  printf "rome %-16s %-8s run %s rc=%s %s\n" $g $v $r $rc "$(grep -E -m3 'tree build|of computing|allocation and init' $log | sed -E 's/.*(tree build|computing|initialization)[^0-9]*([0-9.]+).*/\1 \2/' | paste -sd' ')"
  return $rc
}
rome_runs() {  # rome_runs GRAPH...   (NREP runs per build, interleaved across builds; then the instrumented run)
  for g in "$@"; do
    [ -f $BENCH_EXPORT/$g.rome.mtx ] || { echo "rome $g: no exported graph (run gpu_apsp.jl first)"; continue; }
    for r in $(seq ${ROME_NREP:-5}); do for v in $ROME_VARIANTS; do rome_one $g $v $r || true; done; done
    rome_one $g instr 1 || true
  done 2>&1 | tee -a $OUT/gpubase_rome.txt
  return 0
}

# ---- Fast-APSP (Yang et al., ICS'23), the authors' code through the drivers of our earlier comparison: fap_driver (one
#      GPU) and fap_mpi_driver (one MPI rank per GPU). Its timer ("canonical"): conversion from our CSR to its graph, METIS
#      partitioning, preconditioning, solve; the result ends in host memory (its design: per-call device buffers, copies
#      to the host), together with a path matrix it always computes. The partition count K is tuned per graph and GPU
#      count: one run per K, doubling from the smallest K it accepts (n/K must fit its 32-bit indexing), until a K is
#      slower than the best; then NREP more runs at the best K (as in the reference runs).
#      Its check: the row hashes of its whole result against ours ($BENCH_EXPORT/<graph>.rowhash; integer weights only),
#      linked where the driver reads them ($GB_ROWHASH_DIR/apsp_<graph>.rowhash), and its checksum.
FAP_REF=$BENCH_EXPORT/rowhash          # (the drivers read <name>.rowhash there: GB_ROWHASH_DIR)
fap_prepare() {  # graph files for Fast-APSP's reader (graph/weight-undirected/apsp_<g>.mtx, relative to its working folder)
  local g=$1 d=$OUT/fastapsp; mkdir -p $d/work/graph
  [ -f $d/work/graph/weight-undirected/apsp_$g.mtx ] || $FAP/mtx2fap $BENCH_EXPORT/$g.general.mtx $d/work/graph apsp_$g > /dev/null
  mkdir -p $FAP_REF; [ -f $BENCH_EXPORT/$g.rowhash ] && ln -sf $BENCH_EXPORT/$g.rowhash $FAP_REF/apsp_$g.rowhash
  return 0
}
fap_one() {      # fap_one GRAPH NGPU K TAG   -> $OUT/fastapsp/<g>.g<NGPU>.k<K>.<TAG>.log
  local g=$1 c=$2 K=$3 tag=$4 d=$OUT/fastapsp; local log=$d/$g.g$c.k$K.$tag.log s=$(date +%s.%N) fp=
  rm -f $d/$g.g$c.k$K.$tag.fp*; [ $tag = run1 ] && fp="GB_FP_PROBES=$BENCH_EXPORT/$g.probes GB_FP_OUT=$d/$g.g$c.k$K.$tag.fp"
  if [ $c = 1 ]; then
    ( cd $d/work && { [ -z "$fp" ] || export $fp; } && export GB_ROWHASH_DIR=$FAP_REF && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=$GPU_CPUS timeout --signal=KILL ${FAP_TIMEOUT:-1500} /usr/bin/time -f "MAXRSS_KB %M" \
        $FAP/fap_driver -f apsp_$g -k $K -direct false -weight true -code fastapsp -mtx $BENCH_EXPORT/$g.general.mtx ) > $log 2>&1
  else
    local th=${FAP_THREADS:-$(( ${SLURM_CPUS_ON_NODE:-$GPU_CPUS} / c ))}; [ $th -lt 1 ] && th=1   # (FAP_THREADS: per rank, fixed)
    ( cd $d/work && { [ -z "$fp" ] || export $fp; } && export GB_ROWHASH_DIR=$FAP_REF && OMP_NUM_THREADS=$th OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm UCX_TLS=self,sm timeout --signal=KILL ${FAP_TIMEOUT:-1500} \
        srun --overlap --propagate=CORE --mpi=${HPC_PMIX:-pmix} -n $c --cpus-per-task=$th $FAP/fap_mpi_driver \
        -f apsp_$g -k $K -direct false -weight true -code fastapsp_mg$c -mtx $BENCH_EXPORT/$g.general.mtx ) > $log 2>&1
  fi
  local rc=$?
  echo "PROCESS_WALL $(echo "$(date +%s.%N) - $s" | bc) rc=$rc" >> $log
  local t=$(grep -m1 '^RESULT ' $log | python3 -c 'import json,sys; l=sys.stdin.read(); print(json.loads(l[7:]).get("canonical","-") if l else "-")' 2>/dev/null)
  printf "fastapsp %-15s %d GPU K=%-4s %-6s rc=%s canonical=%s\n" $g $c $K $tag $rc "${t:--}" >&2
  echo "${t:--}"
}
fap_runs() {     # fap_runs NGPU GRAPH...
  local c=$1; shift
  module load gcc/12.2.0 openmpi/5.0.7 > /dev/null 2>&1 || true
  for g in "$@"; do
    [ -f $BENCH_EXPORT/$g.general.mtx ] || { echo "fastapsp $g: no exported graph (run gpu_multigpu.jl first)"; continue; }
    fap_prepare $g
    local n=$(grep -v '^%' $BENCH_EXPORT/$g.general.mtx | head -1 | awk '{print $1}')
    # K: c * 2^j (every rank gets the same number of parts: its loadBalance gives rank P-1 the remainder of K / P; for
    # c = 1, 2, 4, 8 these are the powers of two of the earlier runs), above the sizes its 32-bit indices reach: a part's
    # Floyd matrix (n/K)^2 and its min-plus product, (rows of a part) x n, both < 2^31 (the latter: an illegal memory
    # access at minplus.cu:230, e.g. USA-t-NY with K = 8 and 16)
    local K=$c; [ $K -lt 2 ] && K=2
    while [ $(echo "1.15 * $n / $K >= 46340" | bc) = 1 ]; do K=$((K * 2)); done
    while [ $(echo "$n * $n / $K >= 2^31" | bc) = 1 ]; do K=$((K * 2)); done
    local best= bestK=
    # K tuned in the earlier comparison for this graph, GPU and count (FAP_KPRIOR: "graph gpu count K"): no sweep
    local kind=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p | grep -q B200 && echo b200 || echo rtx)
    local kp=$( [ -f "${FAP_KPRIOR:-}" ] && awk -v g=$g -v k=$kind -v c=$c '$1 == g && $2 == k && $3 == c {print $4; exit}' $FAP_KPRIOR )
    if [ -n "$kp" ]; then
      echo "fastapsp $g $c GPUs: K=$kp from the earlier tuning (no sweep)"
      local t1=$(fap_one $g $c $kp run1) nrep=${FAP_NREP:-3}
      [ "$t1" != - ] && [ "$t1" != null ] && [ $(echo "$t1 < 60" | bc) = 1 ] && nrep=${FAP_NREP_FAST:-5}   # (5 runs, 3 over 60 s)
      for r in $(seq 2 $nrep); do fap_one $g $c $kp run$r > /dev/null; done
      continue
    fi
    local ns=0   # (FAP_KSWEEP: at most that many K values swept)
    while [ $K -le ${FAP_KMAX:-128} ] && [ $ns -lt ${FAP_KSWEEP:-99} ]; do
      local t=$(fap_one $g $c $K sweep); ns=$((ns + 1))
      [ "$t" = - ] || [ "$t" = null ] && { [ -n "$best" ] && break; K=$((K * 2)); continue; }
      if [ -z "$best" ] || [ $(echo "$t < $best" | bc) = 1 ]; then best=$t; bestK=$K; else break; fi
      K=$((K * 2))
    done
    echo "fastapsp $g $c GPUs: best K=$bestK (sweep $best s)"
    local nrep=${FAP_NREP:-3}; [ -n "$best" ] && [ $(echo "$best < 60" | bc) = 1 ] && nrep=${FAP_NREP_FAST:-5}
    [ -n "$bestK" ] && for r in $(seq $nrep); do fap_one $g $c $bestK run$r > /dev/null; done
  done 2>&1 | tee -a $OUT/gpubase_fastapsp.txt
  module unload openmpi gcc > /dev/null 2>&1 || true
  return 0
}

# ---- the other single-GPU baselines (setup_gpu.sh others: the earlier comparison's drivers, with fingerprints), on the
#      benchmark graphs and Float32 weights ($BENCH_EXPORT/<graph>.general.mtx). One process per run (cold), NREP runs.
#   cugraph   RAPIDS cuGraph (ROME's single-GPU baseline): od_shortest_distances, batches of sources, all n of them
#   gunrock   Gunrock SSSP from every source;  adds  ADDS (float weights) from every source
#             (these two, and cuGraph if it would not finish: when a full run is estimated above SSSP_BUDGET s, a sample
#             of sources, ~SSSP_SAMPLE_TARGET s, extrapolated to n and marked so)
#   fw        dense blocked Floyd-Warshall, the tuned kernel of the earlier comparison (float), O(n^3)
#   ecl       ECL-APSP (dense FW, int distances: integer-weight graphs only)
#   fastapsp1 Fast-APSP on one GPU (its host result needs ~12 n^2 bytes: graphs that fit FAP1_GB)
#   Each run's rows are fingerprinted (GB_FP_*) and compared with ours by gpu_baselines.jl; integer-weight graphs are also
#   row-hashed against ours (GB_ROWHASH_DIR). Logs: $OUT/others/<graph>.<code>.r<k>.log (+ .fp of run 1).
OTHERS=${OTHERS:-"cugraph gunrock adds fw ecl fastapsp1"}
GBP=$BASELINES/gpu/build
others_one() {   # others_one CODE GRAPH RUN
  local c=$1 g=$2 r=$3 d=$OUT/others cwd=.; mkdir -p $d
  local log=$d/$g.$c.r$r.log mtx=$BENCH_EXPORT/$g.general.mtx fp=
  rm -f $d/$g.$c.r$r.fp; [ $r = 1 ] && fp="GB_FP_PROBES=$BENCH_EXPORT/$g.probes GB_FP_OUT=$d/$g.$c.r$r.fp"
  local hashdir=$BENCH_EXPORT/rowhash; mkdir -p $hashdir; [ -f $BENCH_EXPORT/$g.rowhash ] && ln -sf $BENCH_EXPORT/$g.rowhash $hashdir/$g.general.rowhash
  local cmd
  case $c in
    cugraph) cmd="env LD_LIBRARY_PATH=$BASELINES/gpu/envs/rapids/lib CUGRAPH_MODE=od SSSP_BUDGET=${SSSP_BUDGET:-60} SSSP_SAMPLE_TARGET=${SSSP_SAMPLE_TARGET:-12} $GBP/cugraph/cugraph_apsp $mtx" ;;
    gunrock) cmd="env SSSP_BUDGET=${SSSP_BUDGET:-60} SSSP_SAMPLE_TARGET=${SSSP_SAMPLE_TARGET:-12} $GBP/gunrock/gunrock_apsp $mtx" ;;
    adds)    cmd="env SSSP_BUDGET=${SSSP_BUDGET:-60} SSSP_SAMPLE_TARGET=${SSSP_SAMPLE_TARGET:-12} $GBP/adds/float/adds_apsp_float $mtx" ;;
    fw)      cmd="$GBP/fw/fwbench $mtx --algo tuned --budget ${FW_BUDGET:-400}" ;;
    ecl)     [ -f $BENCH_EXPORT/$g.rowhash ] || { echo "others $g ecl: real weights, ECL-APSP has int distances only: not run" | tee $log; return 0; }
             cmd="$GBP/fw/ecl_apsp $mtx --budget ${FW_BUDGET:-400}" ;;
    fastapsp1)
             local n=$(grep -v '^%' $mtx | head -1 | awk '{print $1}')
             [ $(echo "12 * $n * $n / 1000000000 < ${FAP1_GB:-48}" | bc) = 1 ] || { echo "others $g fastapsp1: needs ~$((12 * n * n / 1000000000)) GB of host memory, over this job's: not run here (see gpu_fastapsp)" | tee $log; return 0; }
             fap_prepare $g; cwd=$OUT/fastapsp/work
             cmd="env CUDA_VISIBLE_DEVICES=0 $FAP/fap_driver -f apsp_$g -k ${FAP1_K:-8} -direct false -weight true -code fastapsp -mtx $mtx" ;;
  esac
  local s=$(date +%s.%N)
  ( cd $cwd && timeout --signal=KILL ${OTHERS_TIMEOUT:-900} env GB_ROWHASH_DIR=$hashdir $fp OMP_NUM_THREADS=$GPU_CPUS /usr/bin/time -f "MAXRSS_KB %M" $cmd ) > $log 2>&1
  local rc=$?
  echo "PROCESS_WALL $(echo "$(date +%s.%N) - $s" | bc) rc=$rc" >> $log
  printf "others %-15s %-9s run %s rc=%s %s\n" $g $c $r $rc "$(grep -m1 '^RESULT' $log | grep -oE '"(inproc|status|extrapolated)":[^,]*' | paste -sd' ')"
  return 0
}
others_runs() {  # others_runs GRAPH...
  for g in "$@"; do
    [ -f $BENCH_EXPORT/$g.general.mtx ] || { echo "others $g: no exported graph (run gpu_apsp.jl first)"; continue; }
    for c in $OTHERS; do for r in $(seq ${OTHERS_NREP:-3}); do others_one $c $g $r || true; done; done
  done 2>&1 | tee -a $OUT/gpubase_others.txt
  return 0
}
