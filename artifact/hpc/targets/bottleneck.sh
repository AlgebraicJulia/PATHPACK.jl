# Genericity: all-pairs widest paths (max, min), directed random capacities, one core: our closure vs n x widest-path
# Dijkstra and Boost's generic Dijkstra.  -> bottleneck.csv, external.csv (boost_widest)
TIME=05:00:00
MEM=120gb
TASKS=('BENCH_OUT_GB=100 jl 1 bottleneck.jl $APSP_A $APSP_B && boost_widest $APSP_A $APSP_B'
       'BENCH_OUT_GB=100 jl 1 bottleneck.jl $APSP_C && boost_widest $APSP_C')
