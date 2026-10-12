# The C++ baselines, as task helpers (sourced by job.sh). Each appends to a log in $OUT that
# scripts/parse_baselines.py turns into rows of data/external.csv.
GAPD=$BASELINES/gapbs/sssp_double; PHAST=$BASELINES/RoutingKit/phast_driver; CCH=$BASELINES/RoutingKit/cch_driver
STREAM=$BASELINES/STREAM/stream_c; BOOSTW=$BASELINES/bottleneck_boost; GAPAPSP=$BASELINES/gapbs/gap_apsp
# drivers of ours are rebuilt when their source is newer than the binary (no need to rerun setup.sh)
fresh() { [ -x "$1" ] && [ "$1" -nt "$2" ]; }
fresh $GAPAPSP $REPO/baselines/gap_apsp.cpp || g++ -std=c++11 -O3 -march=native -fopenmp -I $BASELINES/gapbs/src $REPO/baselines/gap_apsp.cpp -o $GAPAPSP
fresh $BOOSTW $REPO/baselines/bottleneck_boost.cpp || g++ -std=c++17 -O3 -march=native -I$BASELINES/boost-1.86.0 $REPO/baselines/bottleneck_boost.cpp -o $BOOSTW

# ---- PHAST, CCH, STREAM (unchanged) ----
phast_single() { for g in $DIMACS; do echo "## phast 1 thread | $g"
                 OMP_NUM_THREADS=1 c1 $PHAST $(gr $g) 32 1 | tail -1; done 2>&1 | tee -a $OUT/baselines_phast_single.txt; }
phast_region() { for g in $DIMACS; do echo "## phast | 16 threads, one region | $g"
                 c16 $PHAST $(gr $g) 32 16 | tail -1; done 2>&1 | tee -a $OUT/baselines_phast_region.txt; }
cch_single()   { for g in $DIMACS; do echo "## cch | $g"
                 c1 $CCH $(gr $g) $BENCH_DATA/co/USA-road-d.$g.co 1000 | tail -1; done 2>&1 | tee -a $OUT/baselines_cch.txt; }
stream_region(){ { echo "## stream | 1 thread"; OMP_NUM_THREADS=1 c1 $STREAM | grep "^Triad"
                   echo "## stream | 16 threads, one region"; OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores c16 $STREAM | grep "^Triad"
                 } 2>&1 | tee -a $OUT/baselines_stream.txt; }

# ---- GAP's Delta-stepping, built with double weights (setup.sh), on exactly our graphs and weights ----
# gap_on T FILE ARGS...  GAP on the first T cores of this allocation, T OpenMP threads bound to them
gap_on() { local t=$1; shift; OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores numactl --physcpubind=$(first $t) $MEMPOL $GAPD "$@"; }
# gap_deltas FILE       candidate Deltas relative to the graph's own weights: mean arc weight x 2^k, k = -6..9
gap_deltas() { case $1 in *.gr) awk '$1 == "a" {s += $4; n++} END {m = s / n; for (k = -6; k <= 9; k++) printf "%g ", m * 2^k}' $1 ;;
                          *)    awk 'NF >= 3 {s += $3; n++} END {m = s / n; for (k = -6; k <= 9; k++) printf "%g ", m * 2^k}' $1 ;; esac; }
# gap_best_delta T FILE  the candidate Delta with the smallest average time (4 random sources each)
gap_best_delta() { local t=$1 f=$2 best= bt=
    for d in $(gap_deltas $f); do x=$(gap_on $t -f $f -n 4 -d $d | awk '/Average Time/ {print $3}')
        if [ -z "$bt" ] || awk -v a=$x -v b=$bt 'BEGIN { exit !(a < b) }'; then best=$d; bt=$x; fi; done; echo $best; }
# gap_apsp G...  all-pairs baseline: GAP's Delta-stepping from all n sources, writing the n x n output (baselines/gap_apsp.cpp),
#   on one core and on P threads (P = all cores of the allocation, or gap_apsp_p P G...).
gap_apsp_p() { local P=$1; shift; for g in "$@"; do f=$OUT/apsp_$g.wel; [ -f $f ] || continue
        d=$(gap_best_delta 1 $f)
        echo "## gap_apsp | 1 thread | $g | delta $d | full matrix"; numactl --physcpubind=$(first 1) $MEMPOL $GAPAPSP $f $d 1 | grep "total_s"
        echo "## gap_apsp | $P threads | $g | delta $d | full matrix"; numactl --physcpubind=$(first $P) $MEMPOL $GAPAPSP $f $d $P | grep "total_s"
    done 2>&1 | tee -a $OUT/baselines_gap_apsp.txt; }
gap_apsp() { gap_apsp_p $(ncores) "$@"; }
# gap_queries G...  single queries on 1 and 16 threads, best Delta for each. Directed graphs: our exact sources (-r),
#   one per invocation; otherwise GAP's own picker, which is uniform there (strongly connected or undirected graphs).
gap_queries() { for g in "$@"; do
        case $g in USA-t-*) f=$(gr ${g#USA-t-}) ;; *) f=$OUT/queries_$g.wel ;; esac; [ -f $f ] || continue
        for t in 1 16; do d=$(gap_best_delta $t $f); echo "## gap_query | $t threads | $g | delta $d"
            if [[ " $DIRECTED " == *" $g "* ]]; then for s in $(head -32 $OUT/queries_$g.src); do gap_on $t -f $f -r $s -n 1 -d $d | grep "Average Time"; done
            else gap_on $t -f $f -n 64 -d $d | grep "Average Time"; fi
        done; done 2>&1 | tee -a $OUT/baselines_gap_query.txt; }

# ---- Boost's generic Dijkstra for widest paths, on the graphs and targets bottleneck.jl wrote; checked against our closure ----
boost_widest() { for g in "$@"; do [ -f $OUT/bottleneck_$g.wel ] || continue
        echo "## boost_widest | 1 thread | $g"
        c1 $BOOSTW $OUT/bottleneck_$g.wel $OUT/bottleneck_$g.src $OUT/bottleneck_$g.boost
        python3 -c "import sys; a, b = ([float(x) for x in open(p)] for p in sys.argv[1:]); print('boost check:', 'PASS' if a == b else 'FAIL')" $OUT/bottleneck_$g.boost $OUT/bottleneck_$g.ref
    done 2>&1 | tee -a $OUT/baselines_boost_widest.txt; }
