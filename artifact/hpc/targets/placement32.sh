# Test: can Slurm give us two whole NUMA regions? Prints the placement; job.sh refuses anything else.
NOGATE=1
REGIONS=2
TIME=00:05:00
MEM=8gb
TASKS=('numactl --show; echo "placement OK: $BENCH_PLACEMENT"')
