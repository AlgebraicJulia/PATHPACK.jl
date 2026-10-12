# Sourced by every script: paths, Julia settings, and the graph sets used by the targets.
HPC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HPC")"
source "$HPC/config.sh"
# libcommit: the library version recorded with every result (git, or artifact/VERSION in a download without git)
# (git only if this repository is a checkout of its own, not a folder inside another one)
libgit() { local t; t=$(git -C "$LIB" rev-parse --show-toplevel 2>/dev/null) && [ "$(cd "$t" && pwd -P)" = "$(cd "$LIB" && pwd -P)" ]; }
libcommit() { { libgit && git -C "$LIB" rev-parse --short HEAD; } || sed -n 1p "$REPO/VERSION"; }
libdirty() { libgit && [ -n "$(git -C "$LIB" status --porcelain --untracked-files=no -- src ext 2>/dev/null)" ] && echo true || echo false; }
# sbatch account/QOS options, if set in config.sh
SLURM_AQ="${ACCOUNT:+--account=$ACCOUNT} ${QOS:+--qos=$QOS}"
export LIB="$(dirname "$REPO")"                        # the library: this repository (artifact/ is inside it)
export BENCH=$REPO/bench
export BENCH_DATA=${BENCH_DATA:-$WORK/data}
export BASELINES=$WORK/baselines
export JULIA_DEPOT_PATH=$WORK/julia-depot:
export OPENBLAS_NUM_THREADS=1
export PATH="$(dirname "$JULIA"):$PATH"

# ---- graph sets (the same lists as bench/graphs.jl; split for task size) ----
APSP_A="USpowerGrid delaunay_n13 G67 c-42 CurlCurl_0 fe_sphere nd6k delaunay_n14 OPF_6000 EB_8192_256 EB_16384_64 rgg2d_14 rgg3d_14 hypercube_14"
APSP_B="lpl1 email-Enron t60k wing delaunay_n16 c-69 oilpan"
APSP_C="finan512 fe_tooth onera_dual net4-1 luxembourg_osm"
SSSP_A="USA-t-NY USA-t-BAY USA-t-COL USA-t-FLA USA-t-NW USA-t-NE delaunay_n16 wing c-69 email-Enron soc-Epinions1"
SSSP_B="USA-t-CAL USA-t-E USA-t-W"
SSSP_C="USA-t-CTR"
SSSP_D="USA-t-USA"
SCALING="finan512 net4-1 email-Enron wing"
GATE="USpowerGrid wiki-Vote G67 delaunay_n14 soc-Epinions1 USA-t-NY"   # wiki-Vote: closure with many components
DIRECTED="soc-Epinions1"                 # GAP's own source picker is biased on these: pass our sources with -r
DIMACS="NY BAY COL FLA NW NE CAL E W CTR USA"

# ---- GPU experiments ----
export LIB_GPU=${LIB_GPU_DIR:-$LIB}                      # the library with its CUDA extension (the same tree)
export BENCH_GPU=${BENCH_GPU_ENV:-$BENCH/gpu}          # the GPU benchmarks' Julia environment (bench/gpu/Project.toml)
# (another version of the library side by side: LIB_GPU_DIR = its tree, BENCH_GPU_ENV = an environment that develops it,
#  BENCH_EXPORT = its own graph/hash folder)
APSP_ALL="$APSP_A $APSP_B $APSP_C"
FASTAPSP="598a luxembourg_osm web-sk-2005 usroads-48 fe_ocean"
APSP_LARGE="t60k wing delaunay_n16 c-69 oilpan finan512 fe_tooth onera_dual net4-1"   # the all-pairs graphs with n >= 60,000 (not in FASTAPSP)
MULTI_GRAPHS="finan512 fe_tooth onera_dual net4-1 598a luxembourg_osm web-sk-2005 usroads-48 fe_ocean"   # several GPUs: Fast-APSP's 5 and the 4 largest all-pairs graphs, smallest first
LARGE1="598a luxembourg_osm web-sk-2005 usroads-48 fe_ocean delaunay_n17 rgg_n_2_17_s0"
LARGE2="144 Dubcova3 wave c-73 cont-300 thermomech_dM"   # one GPU near its limit (n ~ 145k-205k)
HUGE="delaunay_n19"                                     # 7-8 GPUs only
ROADS="USA-t-NY USA-t-BAY USA-t-COL"
SR_GRAPHS=${SR_GRAPHS:-"USpowerGrid delaunay_n13 G67 c-42 CurlCurl_0 fe_sphere nd6k lpl1 email-Enron t60k wing delaunay_n16 c-69 oilpan finan512 fe_tooth luxembourg_osm 598a"}   # semirings
GATE_GPU="power ca-GrQc delaunay_n14 grid3d-25 bcsstk30 kron_g500-logn16 luxembourg_osm"
export BENCH_GATE_MTX=$GPU_GATE_MTX BENCH_GATE_REF=$GPU_GATE_REF
