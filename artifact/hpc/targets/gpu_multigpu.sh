# Several GPUs of one node, every count 1-8: ours (whole call and closure alone, per-GPU times) on Fast-APSP's graphs and
# the all-pairs graphs with n >= 60,000.  -> gpu_multigpu.csv
# A whole 8-GPU node; one such job at a time (SINGLETON).
SINGLETON=1
TIME=02:30:00
MEM=200gb
GPUS=8
CPUS=24
NTASKS=8
TASKS=('jlg gpu_multigpu.jl $FASTAPSP $APSP_LARGE')
