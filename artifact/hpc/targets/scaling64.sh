# Strong scaling across four NUMA regions (64 cores, memory interleaved over all four).
REGIONS=4
TIME=04:00:00
MEM=480gb
TASKS=('BENCH_OUT_GB=400 jl 64 apsp.jl $SCALING && gap_apsp_p 64 $SCALING')
