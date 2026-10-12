# Strong scaling within one region (SuperFW's Fig. 7 graphs): our closure and n x Dijkstra on T threads, GAP with T processes.
TIME=05:00:00
MEM=120gb
TASKS=('BENCH_OUT_GB=100 jl 1 apsp.jl $SCALING && gap_apsp_p 1 $SCALING'  'BENCH_OUT_GB=100 jl 2 apsp.jl $SCALING && gap_apsp_p 2 $SCALING'
       'BENCH_OUT_GB=100 jl 4 apsp.jl $SCALING && gap_apsp_p 4 $SCALING'  'BENCH_OUT_GB=100 jl 8 apsp.jl $SCALING && gap_apsp_p 8 $SCALING'
       'BENCH_OUT_GB=100 jl 16 apsp.jl $SCALING && gap_apsp_p 16 $SCALING')
