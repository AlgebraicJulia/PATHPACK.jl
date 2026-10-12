# One GPU: ours (BestFill and AMF orderings, hybrid and CPU-only factorization; phases, cold/warm, checks), ROME, and the
# other GPU baselines (cuGraph, Gunrock, ADDS, dense FW, ECL-APSP, Fast-APSP) on the 26 all-pairs graphs.  -> gpu_apsp.csv
# Per task, on one node, a third of the graphs: ours first (gpu_apsp.jl also writes the graphs, row hashes and
# fingerprints the baselines are checked against), then ROME's builds, then the others, each checked (gpu_baselines.jl).
TIME=04:00:00
MEM=64gb
GPUS=1
T='jlg gpu_apsp.jl $S || true; rome_runs $S; jlg gpu_baselines.jl rome $S || true; others_runs $S; jlg gpu_baselines.jl others $S || true'
TASKS=("S=\"$APSP_A\"; $T" "S=\"$APSP_B\"; $T" "S=\"$APSP_C\"; $T")
