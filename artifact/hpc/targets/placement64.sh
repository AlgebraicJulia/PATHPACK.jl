# Test: can Slurm give us four whole NUMA regions? Prints the placement; job.sh refuses anything else.
NOGATE=1
REGIONS=4
TIME=00:05:00
MEM=8gb
TASKS=('numactl --show; echo "placement OK: $BENCH_PLACEMENT"')
