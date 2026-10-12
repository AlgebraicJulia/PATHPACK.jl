# One GPU: dense kernels (min-plus GEMM, triangular solves, triangular inverse) vs cuBLAS / cuSOLVER, Float32.  -> gpu_kernels.csv
TIME=01:00:00
MEM=32gb
GPUS=1
TASKS=('jlg gpu_kernels.jl')
