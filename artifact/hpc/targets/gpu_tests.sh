# Precondition for the GPU targets: the library's own GPU test suite (test/gpu, quick) and the GPU gate (gpu_gate.jl:
# every entry point bitwise identical to reference row hashes on integer-weight graphs, 1 and 2 GPUs). Logs only; the
# job fails when the gate fails (submit the other targets with --after JOB to run them only then).
TIME=02:30:00
MEM=64gb
GPUS=2
TASKS=('$JULIA --project=$WORK/gpu-testenv $LIB_GPU/test/gpu/runtests.jl quick --logdir=$OUT/testlogs && echo "TEST SUITE PASSED" || echo "TEST SUITE FAILED (exit $?)"; { [ -n "$BENCH_GATE_MTX" ] && jlg gpu_gate.jl $GATE_GPU || echo "GPU gate skipped (set GPU_GATE_MTX and GPU_GATE_REF in hpc/config.sh)"; }')
