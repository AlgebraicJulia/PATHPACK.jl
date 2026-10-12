# Main all-pairs plot and phase breakdown: our closure (phases) and n x Dijkstra on 1 and 16 cores; GAP n-SSSP on one core
# and with one process per core.  -> apsp.csv, external.csv (gap_apsp)
TIME=05:00:00
MEM=120gb
TASKS=('BENCH_OUT_GB=100 jl 1 apsp.jl $APSP_A && gap_apsp $APSP_A' 'BENCH_OUT_GB=100 jl 16 apsp.jl $APSP_A'
       'BENCH_OUT_GB=100 jl 1 apsp.jl $APSP_B && gap_apsp $APSP_B' 'BENCH_OUT_GB=100 jl 16 apsp.jl $APSP_B'
       'BENCH_OUT_GB=100 jl 1 apsp.jl $APSP_C && gap_apsp $APSP_C' 'BENCH_OUT_GB=100 jl 16 apsp.jl $APSP_C')
