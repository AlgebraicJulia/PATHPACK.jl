# Strong scaling across two NUMA regions (32 cores, memory interleaved over both).
REGIONS=2
TIME=04:00:00
MEM=240gb
TASKS=('BENCH_OUT_GB=200 jl 32 apsp.jl $SCALING && gap_apsp_p 32 $SCALING')
