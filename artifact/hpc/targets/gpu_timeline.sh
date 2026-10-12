# One GPU: the timeline of one call on fe_tooth (bench/timeline/measure2.jl): a host trace of every step, CUDA events
# for the GPU's work, and the call under the step timer, interleaved.  -> gpu_timeline_trace.csv
TIME=01:00:00
MEM=64gb
GPUS=1
TASKS=('jlg timeline/measure2.jl $BENCH_DATA/mtx/fe_tooth/fe_tooth.mtx $OUT/timeline 11 $(libcommit)')
