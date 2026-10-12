# One GPU: every semiring the GPU closure supports (bench/semirings/gpu_semirings.jl): the GEMM kernels' instruction
# counts, the kernels against cuBLAS, and the closure on 18 graphs; then (same job, on its CPU cores) each closure's
# sampled columns checked against the CPU closure in the same semiring, Float64 (bench/semirings/sr_check.jl).
#   -> gpu_semirings.csv, gpu_semirings_kernels.csv (collect.sh runs bench/semirings/build_csv.py)
TIME=10:00:00
MEM=64gb
GPUS=1
TASKS=('SR_BENCH=$BENCH SR_TAG=artifact jlg semirings/gpu_semirings.jl $SR_GRAPHS; SR_BENCH=$BENCH SR_RAW=$OUT jlg semirings/sr_check.jl $SR_GRAPHS')
