# Reuse: single queries (1 and 16 cores) and blocks of 1..1024 right-hand sides vs Dijkstra; GAP single queries (1, 16 threads).
TIME=05:00:00
MEM=120gb
TASKS=('BENCH_RHS_GB=40 jl 16 queries.jl $SSSP_A && gap_queries $SSSP_A' 'BENCH_RHS_GB=40 jl 16 queries.jl $SSSP_B && gap_queries $SSSP_B'
       'BENCH_RHS_GB=40 jl 16 queries.jl $SSSP_C && gap_queries $SSSP_C' 'BENCH_RHS_GB=40 jl 16 queries.jl $SSSP_D && gap_queries $SSSP_D')
