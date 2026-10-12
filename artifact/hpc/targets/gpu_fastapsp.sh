# Several GPUs of one node: Fast-APSP (the authors' code) on 1-8 GPUs, on the multi-GPU graphs, K tuned per graph and
# count, every run's rows checked against ours (row hashes, or fingerprints with real weights; run gpu_apsp and
# gpu_multigpu first: they write them).  -> gpu_multigpu.csv
# A whole 8-GPU node per task (one GPU count each), one at a time (SINGLETON). Its result lives in host memory: on one
# GPU it needs about 12 n^2 bytes (fe_ocean: ~250 GB, over this job's memory: reported as a failure).
SINGLETON=1
TIME=02:30:00
MEM=200gb
GPUS=8
CPUS=24
NTASKS=8
TASKS=()
for c in 1 2 3 4 5 6 7 8; do TASKS+=("fap_runs $c \$FASTAPSP \$APSP_LARGE; jlg gpu_baselines.jl fastapsp \$FASTAPSP \$APSP_LARGE || true"); done
