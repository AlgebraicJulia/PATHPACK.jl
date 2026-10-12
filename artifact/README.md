# Benchmarks

This folder benchmarks the solver, the package in the parent folder, against state-of-the-art codes for shortest paths:
- **On GPUs:** ROME, Fast-APSP, cuGraph, Gunrock, ADDS, ECL-APSP, and a dense Floyd-Warshall.
- **On CPUs:** n runs of Dijkstra's algorithm, GAP's Delta-stepping, PHAST, CCH, and Boost.

It holds three things:
- the harness that runs the benchmarks;
- the reference measurements, from nodes of 8 NVIDIA B200 GPUs and a node with two 64-core AMD EPYC 7702 processors;
- the scripts that turn the measurements into derived numbers (LaTeX macros), tables, and figure data.

For installation and a quick start of the solver itself, see [SOLVER.md](../SOLVER.md).

There are four ways to check the results, from seconds to days. Commands run from the repository root unless they say
`cd artifact`, or the section says to run them from `artifact/`.

1. **Recompute every derived number from the reference measurements** (Python, a few seconds):
   `make -C artifact check`.
2. **Benchmark on your own machine** (minutes, no cluster, nothing to configure): `bash artifact/quick.sh cpu`, and on
   a machine with an NVIDIA GPU, `bash artifact/quick.sh gpu --rome`. These run the benchmark drivers on a few graphs
   and compare your results with the reference measurements ([below](#benchmark-on-your-machine)).
3. **Run the solver:** the quick starts in [SOLVER.md](../SOLVER.md) and the test suites.
4. **Rerun the benchmarks** on a Slurm cluster (`hpc/`), then compare your results with the reference measurements
   (`scripts/regress.py`).

## Contents

| Path | What it is |
|---|---|
| `quick.sh` | The benchmark on one machine, without Slurm: a few graphs, the benchmark drivers, and a comparison with the reference measurements ([below](#benchmark-on-your-machine)). |
| `bench/` | The Julia benchmark drivers, one per experiment ([bench/README.md](bench/README.md)), and their environments (`Project.toml`, `gpu/Project.toml`). `graphs.jl` holds the graph sets and weights; `lib.jl` and `gpu_lib.jl` hold the timing, the checks, and the result files. |
| `hpc/` | The Slurm harness. `config.sh` is the one file to edit; `setup.sh` and `setup_gpu.sh` are one-time setup; `fetch_data.sh` downloads the graphs; `run.sh` (CPU) and `run_gpu.sh` (GPU) submit the experiments. |
| `hpc/targets/` | One file per experiment: its tasks, time, and memory. |
| `hpc/reference/` | The exact submissions behind the reference GPU measurements. |
| `baselines/` | The competitors' drivers and our patches to them ([baselines/README.md](baselines/README.md)). |
| `data/` | The reference measurements, as CSV. `*.csv` are the CPU runs and `gpu/*.csv` the GPU runs. `legacy/*.csv` are earlier CPU runs: `scripts/csv_to_tex.py` reads them, but every number they give is redefined from `data/*.csv`. |
| `scripts/` | Measurements to numbers, tables, and figure data (`gpu_data.py`, `cpu_data.py`, `csv_to_tex.py`, `fig_apsp_combined.py`, `table_graphs.py`). Raw results to CSV (`collect.py`, `parse_baselines.py`). Comparisons: `regress.py` (your results against `data/`), `rome_report.py` (per graph, against ROME), `compare_macros.py`, `quick_report.py` (`quick.sh`'s plan and report). |
| `reference/` | The derived numbers of the reference measurements (LaTeX macros), which `make check` compares with. |
| `Makefile` | `make numbers`, `make check`, `make regress`, `make clean`. |
| `VERSION` | The label of this repository's library version, `lib-g`. |

## Requirements

- **Checking the numbers:** Python 3 with numpy and pandas (tested with Python 3.13, pandas 2.3, numpy 2.4) and make.
- **Benchmarking on your machine:** see [below](#benchmark-on-your-machine).
- **Rerunning the benchmarks:**
  - a Slurm cluster and Julia 1.12;
  - for the CPU runs, nodes with 8 NUMA regions of 16 cores (`hpc/job.sh` refuses any other placement);
  - for the GPU runs, nodes with 8 NVIDIA GPUs, CUDA 12.9, and CMake 3.30 or newer to build the GPU baselines;
  - git, curl, gcc with OpenMP, and conda or micromamba for the cuGraph environment.

## Recompute the derived numbers

    make -C artifact check

The `check` target first runs `make numbers`, which writes these files from the CSVs in `data/`:
- **Every derived number:** 670 LaTeX macros, in `build/data/{macros,results,gpu_results}.tex`. Some are diagnostics.
- **The tables:** `build/tables/`, among them the test graphs (`tab-graphs.tex`), the speedups over the other GPU codes
  (`tab-gpu-baselines.tex`), and the runs on several GPUs (`tab-gpu-multigpu.tex`).
- **The data of every figure:** `build/data/*.dat`, `build/data/gpu_phases_1col.tex`, and `build/figures/`.

It then compares the macros with `reference/`:

    build/data/gpu_results.tex: 460 macros, 460 in the reference; 0 differ
    build/data/results.tex: 67 macros, 67 in the reference; 0 differ
    build/data/macros.tex: 143 macros, 143 in the reference; 0 differ

The scripts print warnings about the rows they skip and why; these are expected.

The speedups over ROME, per graph and per set:

    python3 artifact/scripts/rome_report.py artifact/data/gpu/gpu_apsp.csv

This prints our time, ROME's fastest build, the ratio and the checks for each graph, then a summary:

    ROME's 16 : geomean 3.12x over 16 graphs, wins 16/16, min 1.66 max 5.53
    all 26    : geomean 2.83x over 26 graphs, wins 26/26, min 1.59 max 5.53
    large set : geomean 6.33x over 10 graphs, wins 10/10, min 3.60 max 11.68

## Benchmark on your machine

    bash artifact/quick.sh cpu            # ~5 minutes: ours and n runs of Dijkstra's algorithm, 7 graphs
    bash artifact/quick.sh gpu            # ~15 minutes: ours on the first NVIDIA GPU, ROME's 16 graphs (those that fit)
    bash artifact/quick.sh gpu --rome     # ~30 minutes: the same, then ROME on the same GPU (built the first time)

These run on one machine without Slurm, and there is nothing to configure. The script:
- installs the Julia packages at the versions of the reference runs (`bench/Manifest.toml`, `bench/gpu/Manifest.toml`);
- downloads only the graphs it runs;
- runs the benchmark drivers (`bench/apsp.jl`, `bench/gpu_apsp.jl`, and ROME through `hpc/gpu_baselines.sh` and
  `bench/gpu_baselines.jl`) with the timing conventions and checks of the reference runs;
- prints a comparison with the reference measurements.

Everything it installs, downloads, or builds goes under `artifact/work/` (set `WORK` to change this). The comparison is
also saved in `raw/quick/<run>/report.txt`, and the rows are added to `results/*.csv`, as with the Slurm harness, so
`make -C artifact regress` compares them too.

| Mode | What runs | What it compares |
|---|---|---|
| `cpu` | `bench/apsp.jl` on 1 thread and on every physical core (at most 16): ours, and n runs of Dijkstra's algorithm, on 7 all-pairs graphs whose Dijkstra baseline takes the reference machine under 25 s | the speedup over n runs of Dijkstra's algorithm on one core |
| `gpu` | `bench/gpu_apsp.jl` on one GPU: ours with the default ordering (`ours-auto`), 5 cold calls per graph, on ROME's 16 test graphs that fit in the GPU's free memory | the time against the reference B200's |
| `gpu --rome` | then ROME's three builds (default, leaf sizes 400 and 800), 5 runs each, on the same GPU; the fastest passing build per graph | the speedup over ROME on the same GPU |

Options: `--graphs "G67 wing"` (any graphs of `bench/graphs.jl`), `--reps N` (the GPU's timed calls per graph and code;
the reference runs used 11), `--threads N` (the CPU's multithreaded run).

**Needs:**
- Linux, curl, and Python 3.
- Julia 1.12. Install it with `curl -fsSL https://install.julialang.org | sh`. If 1.12 is not your default version,
  run `juliaup add 1.12` and call the script as `JULIA="julia +1.12" bash artifact/quick.sh ...`.
- For `gpu`: an NVIDIA GPU and its driver. Julia downloads the CUDA runtime: that of the reference runs (13.0), or with
  a driver older than CUDA 13, the newest runtime the driver supports.
- For `--rome`: git, gcc and g++, make, CMake, flex, bison, bc, GNU time (`/usr/bin/time`), and the CUDA toolkit
  (`nvcc` on the PATH, or `CUDA_HOME`). ROME is built for your GPU's architecture
  ([baselines/README.md](baselines/README.md)), with `/usr/bin/gcc` and `/usr/bin/g++`; if your nvcc does not support
  that compiler, set `ROME_CC` and `ROME_CXX` to one it supports.
- Disk: under 1 GB for `cpu`. For `gpu`, a few GB more, mostly the CUDA libraries.

**What to expect on other hardware:**
- **The same:** n, m, and nnz(L+U) of every graph, as in the reference rows of this version (the column `exact`): the
  symbolic analysis gives the same factor on every machine. There is one exception. On the graphs with several
  components (lpl1, net4-1, thermomech_dM), the components are ordered concurrently. METIS, one of the two orderings that
  `BestFill(AMF(), METIS())` tries for the reported factor, is not deterministic when it runs on several threads at
  once, so nnz(L+U) can differ there by up to about 1% (shown as `~`). The timed GPU runs use the ordering `auto`, which
  does not call METIS.
- **Passing:** every result is checked:
  - ours against Dijkstra's algorithm on 64 columns, and on the GPU also against the CPU's star in `Float64`;
  - ROME's sampled rows against Dijkstra's algorithm.
- **Comparable:** speedups, each measured on one machine: over n runs of Dijkstra's algorithm on one core (CPU), and over
  ROME on the same GPU. They depend on the hardware less than times do. `data/gpu/gpu_apsp.csv` also has speedups over
  ROME on an RTX PRO 6000 and an L4.
- **Different:** the times. The report shows the reference times beside yours, for scale.

The times are those of a fresh install on one B200 node, most of them installing Julia's CUDA packages; later runs
skip the installation and ROME's build. An example, `bash artifact/quick.sh cpu` on a laptop, from a fresh download
(under 3 minutes):

    This machine: AMD Ryzen AI 7 350 w/ Radeon 860M, ours on 1 and 8 threads, library lib-g.
    Reference runs: AMD EPYC 7702, 1 core and one NUMA region of 16 cores, library lib-a (data/apsp.csv).
    Times in seconds; speedup = n runs of Dijkstra's algorithm on one core / ours (symbolic + numeric + star);
    exact = n, m and nnz(L+U) as in the reference rows of this version.

                                      ------------------- this machine ------------------   ---- reference ----
    graph                  n  exact    Dijkstra    ours 1    ours 8     x, 1     x, 8        x, 1     x, 16
    USpowerGrid        4,941  yes         0.912    0.0407    0.0364   22.40x   25.06x      30.96x    37.13x
    delaunay_n13       8,192  yes         2.828    0.1961    0.1180   14.42x   23.96x      16.25x    43.90x
    G67               10,000  yes         2.948    0.3826    0.1908    7.71x   15.45x      11.09x    39.98x
    c-42              10,471  yes         6.280    0.2551    0.1690   24.61x   37.16x      29.88x    65.36x
    CurlCurl_0        11,083  yes         8.052    1.2548    0.4288    6.42x   18.78x       8.65x    48.58x
    delaunay_n14      16,384  yes        12.165    0.8436    0.5156   14.42x   23.59x      16.64x    53.86x
    fe_sphere         16,386  yes        10.727    1.5169    0.6229    7.07x   17.22x       9.56x    50.22x

    Exact: 7/7 graphs. Every result was checked against Dijkstra's algorithm on 64 columns (a failed check stops the run).

Its speedups are lower than the reference runs'. On one core, this laptop runs Dijkstra's algorithm about twice as fast
as the reference machine, and our code about 1.5 times as fast. Multithreaded, it has 8 cores instead of 16, and less
memory bandwidth.

## Versions of the library in the data

Every row of `data/` records the version of the library it was measured with, as a label in the column `commit` (and
`provenance.pathpack_commit`):
- `lib-a` to `lib-e` are earlier states of this repository's code.
- `lib-f` is `lib-e` plus three changes that equal the last three of `lib-g`; it has the same GPU kernels as `lib-g`.
- `lib-g` is the code in this repository, with one line added after the reference runs: when the GPU extension loads,
  `reset_caches!()` (`ext/PATHPACKCUDAExt.jl`) also empties `SIDE_STREAMS`, its cache of one CUDA stream per device.
  Without that line, an extension precompiled on a machine with a working GPU kept the precompiling process's stream in
  its package image, and later sessions crashed. The reference runs used images precompiled without a GPU, so the cache
  was empty there too.

| Label | Rows |
|---|---|
| `lib-a` | CPU: all-pairs (`apsp.csv`), widest paths (`bottleneck.csv`), dense kernels (`kernels.csv`) |
| `lib-b` | CPU: queries (`queries.csv`), including the DIMACS road networks |
| `lib-c` | none of ours; it labels baseline rows (GAP) of `external.csv` |
| `lib-d` | GPU: B200 and RTX PRO 6000, orderings AMF and BestFill; several GPUs (which GPU counts can hold a road network's output) |
| `lib-e` | GPU: B200 and L4, ordering `auto`; several B200s |
| `lib-f` | GPU: every semiring (`gpu_semirings*.csv`) |
| `lib-g` | GPU: one B200, with ROME run right after ours on the same node; steps; kernels; several B200s; one traced call |

Baseline rows (ROME, Fast-APSP, cuGraph, Gunrock, ADDS, dense Floyd-Warshall, ECL-APSP, GAP, PHAST, CCH, Boost,
STREAM) carry the label of the job that ran them; the baselines do not depend on it. The rows of the GPU baselines other
than ROME, and Fast-APSP's rows on several GPUs, were run in the `lib-d` jobs.

The Makefile selects the reference rows: `GPU_COMMIT=lib-g`, `CPU_COMMITS=lib-b,lib-a` (two CPU versions that differ
only in the renumbering step, which the timings exclude), and `GPU_HEAD=ours-auto`. The scripts take the other
versions' rows from settings documented at the top of `scripts/gpu_data.py`.

## Hardware and software of the reference runs

| | GPU runs | CPU runs |
|---|---|---|
| Machine | nodes of 8 NVIDIA B200 (180 GB each) with Intel Xeon Platinum 8570 hosts (112 cores) | one node with two 64-core AMD EPYC 7702 processors: 8 NUMA regions of 16 cores, 1 TB of memory |
| Placement | one job per experiment, holding only the GPUs it uses; 16 host cores local to the GPU, 8 Julia threads | whole NUMA regions; up to 16 threads use their region's memory, more use interleaved memory |
| Software | Julia 1.12.6, CUDA.jl 6.1.0 (CUDA runtime 13.0), CliqueTrees 1.19.7, `Float32` | Julia 1.12.7, CliqueTrees 1.19.7, `Float64` |

Two other GPUs appear in `data/gpu/gpu_apsp.csv`, with earlier versions:
- RTX PRO 6000 Blackwell on AMD EPYC 9555 hosts (`lib-d`);
- L4 on AMD EPYC 9655P hosts (`lib-e`).

All of this is recorded per row: the columns `machine`, `gpu.*`, and `provenance.*`.

**Timing:**
- **Scope:** wall-clock, from the graph in memory to the full n × n distance matrix in memory, symbolic analysis
  included.
- **Output:** allocated before the timer (on CPUs, also first touched). The outputs take 4n² bytes on GPUs (`Float32`).
- **GPU:** the headline time is the median of cold calls (`bench/gpu_lib.jl`). Before each call, CUDA.jl's memory pool
  is returned to the driver and the pinned host memory is freed. Julia's compilation, up to 36 s for a graph's first
  call, is excluded by an untimed first call.
- **CPU:** the renumbering to the original labels is excluded (`t_perm`).

## Rerunning the benchmarks

### 1. Configure

Edit `hpc/config.sh`. Every setting has a default and can also be set in the environment, or in `hpc/config.local.sh`,
which is read last:

| Setting | What it is |
|---|---|
| `ACCOUNT`, `QOS` | Slurm account and QOS; empty means not passed to sbatch |
| `PARTITION` | the CPU partition |
| `GPU_PARTITIONS` | the GPU partition(s) |
| `GPU_GRES_<partition>` | the `--gpus` type of each GPU partition (default `b200`) |
| `WORK` | scratch space for the graphs, the Julia depot, and the baselines (default `artifact/work`) |
| `JULIA` | the Julia binary |

### 2. Set up

Run these from `artifact/`:

    ./hpc/setup.sh          # CPU: the bench/ environment (this repository's library), GAP, RoutingKit, Boost, STREAM, the graphs
    ./hpc/setup_gpu.sh      # GPU: the bench/gpu environment with CUDA.jl, the GPU tests' environment, ROME, the other GPU baselines

- `setup.sh` ends with `hpc/fetch_data.sh`. It downloads the graphs from SuiteSparse, SNAP, Network Repository, and the
  9th DIMACS challenge into `$WORK/data`, about 5 GB; the synthetic graphs are generated at run time.
- `setup_gpu.sh` takes one step as an argument (`julia`, `tests`, `rome`, `others`, `check`). Both scripts are safe to
  rerun.
- The baselines are built from their authors' sources at pinned versions, with our patches
  ([baselines/README.md](baselines/README.md)).

### 3. Run

Run these from `artifact/` too.

**Targets.** Each experiment is a target, a file in `hpc/targets/`. `./hpc/run.sh` and `./hpc/run_gpu.sh` with no
arguments list them. Both take `--dry-run` (it prints the jobs and submits nothing) and `--tasks`.

CPU targets, `./hpc/run.sh TARGET...`:

| Target | What it runs |
|---|---|
| `apsp` | all-pairs: our star and n runs of Dijkstra's algorithm on 1 and 16 cores; n runs of GAP's Delta-stepping |
| `scaling`, `scaling32`, `scaling64` | strong scaling on 1-16 cores of one NUMA region, on two regions, and on four |
| `queries` | single queries and blocks of 1 to 1024 right-hand sides against Dijkstra and GAP |
| `external` | PHAST and CCH on the DIMACS road networks |
| `bottleneck` | all-pairs widest paths: our star, n runs of widest-path Dijkstra, and Boost |
| `kernels` | dense semiring kernels against OpenBLAS `dgemm`, one core |
| `bandwidth` | STREAM triad on one NUMA region |
| `gate` | determinism: bit-identical results on 1, 2, 4, and 8 threads; runs before every target |

GPU targets, `./hpc/run_gpu.sh TARGET... [--gpu PARTITION]`:

| Target | What it runs |
|---|---|
| `gpu_tests` | the library's GPU test suite and the GPU gate (the gate needs `GPU_GATE_MTX` and `GPU_GATE_REF`) |
| `gpu_apsp` | one GPU: ours, ROME, and the other GPU baselines on the 26 all-pairs graphs |
| `gpu_kernels` | one GPU: min-plus GEMM, triangular solves, and triangular inverse against cuBLAS and cuSOLVER |
| `gpu_multigpu` | one to eight GPUs of one node: ours |
| `gpu_fastapsp` | one to eight GPUs of one node: Fast-APSP |
| `gpu_semirings` | one GPU: every semiring, 64 saved columns per graph; then, in the same job on its CPU cores, `bench/semirings/sr_check.jl` checks them against the CPU's star in `Float64` and marks the rows; the collect job runs `bench/semirings/build_csv.py`. `SR_GRAPHS` and `SR_PARTS` select a subset |
| `gpu_timeline` | one GPU: a traced call on fe_tooth |

`gpu_apsp` runs the orderings BestFill and AMF unless `BENCH_VARIANTS` says otherwise. The reference runs used the
ordering `auto`.

**The reference GPU runs** (ordering `auto`, each job on a GPU node of its own) are the scripts in `hpc/reference/`:

| Script | What it measures |
|---|---|
| `hpc/reference/one_gpu.sh` | ours and ROME on one GPU, and the speedups over ROME |
| `hpc/reference/phases.sh` | the time per step |
| `hpc/reference/multi_gpu.sh` | one to eight GPUs |
| `hpc/reference/kernels.sh` | the dense kernels |

**One job per graph and code.** `hpc/run_gpu_split.sh` submits the same scripts this way, which is how the other GPU
baselines were run:
- `run_gpu_split.sh apsp --base "G..." --codes "cugraph gunrock adds fw ecl fastapsp1"`;
- `run_gpu_split.sh fastapsp` or `fapbatch` for Fast-APSP on several GPUs. Fast-APSP's K comes from the values tuned in
  the reference runs, `baselines/gpu/fastapsp_kprior.txt` (lines `graph gpu count K`; set `FAP_KPRIOR` to another file,
  or to an empty one to sweep K).

**Where results go.** Every job writes its raw results to `raw/<target>/<run>/` and records:
- the library version, from git, or from `VERSION` in a download without git;
- the machine, the cores, and the GPUs.

A collect job then merges the results into `results/*.csv`, in the same format as `data/`. `data/` is never written.

### 4. Compare

    make -C artifact regress          # or: cd artifact && python3 scripts/regress.py [--tol 0.15] [--all]

`scripts/regress.py` compares every experiment in `results/` with the same experiment in `data/`: same code, graph, GPU,
GPU count, kernel, and size. It checks three things:
- **Exact quantities:** n, m, and nnz(L+U) must be identical. On graphs with several components, nnz(L+U) may differ by
  up to 1%, which is reported as a note: their components are ordered concurrently, and METIS is not deterministic then
  (see [above](#benchmark-on-your-machine)).
- **Correctness:** every new row must pass its own check (`check.ok`, `status`).
- **Timings:** a rerun more than `--tol` slower than the reference (default 15%, about the node-to-node noise between
  B200 nodes) fails. A faster one is reported but does not fail.

It compares only the reference rows. The exit status is 1 if anything failed, and `results/regress.csv` holds every
comparison.

**Expected differences:**
- The CPU code here is `lib-g`'s; the reference CPU rows are `lib-a` and `lib-b`. Same algorithms, but later performance
  work, so CPU reruns may be faster.
- `cd artifact && python3 scripts/rome_report.py` (default input: `results/gpu_apsp.csv`) prints the per-graph comparison
  with ROME for your runs.

## Experiments, data, and scripts

All paths are relative to `artifact/`. Each table and figure-data file is made by its script during `make numbers`.

| Experiment | Measurements | Made by | Rerun with |
|---|---|---|---|
| The test graphs: n, m, fill | `data/gpu/gpu_apsp.csv` (n, m, fill), `data/queries.csv` (DIMACS) | `scripts/table_graphs.py` | the runs below |
| All-pairs on CPUs and on one GPU: the speedup over n Dijkstra runs on one core | `data/apsp.csv`, `data/external.csv` (`gap_apsp`), `data/gpu/gpu_apsp.csv` (`ours-auto`, `rome-*` at `lib-g`) | `scripts/fig_apsp_combined.py` | `hpc/run.sh apsp`; `hpc/reference/one_gpu.sh` |
| Ours against ROME, and one GPU against 16 CPU cores | `data/gpu/gpu_apsp.csv`, `data/apsp.csv` | `scripts/gpu_data.py` (`\GpuRome...`, `\GpuVsCpuSixteen...`); `scripts/rome_report.py` | `hpc/reference/one_gpu.sh` |
| The speedup over the other GPU codes | `data/gpu/gpu_apsp.csv` (`rome-*`, `fw`, `cugraph`, `ecl`, `fastapsp1`, `adds`, `gunrock`) | `scripts/gpu_data.py` | `hpc/reference/one_gpu.sh` (ROME); `hpc/run_gpu_split.sh apsp` (the others) |
| The min-plus kernel on one CPU core | `data/kernels.csv` | `scripts/cpu_data.py` | `hpc/run.sh kernels` |
| The dense kernels on one B200 | `data/gpu/gpu_kernels.csv` | `scripts/gpu_data.py` | `hpc/reference/kernels.sh` |
| The time per step on one B200 | `data/gpu/gpu_apsp.csv` (`ours-auto-phases`; `ours-auto-cpufactor` for the CPU-only factorization) | `scripts/gpu_data.py` | `hpc/reference/phases.sh`; `hpc/reference/one_gpu.sh` |
| Several GPUs: ours and Fast-APSP | `data/gpu/gpu_multigpu.csv` (`ours`, `fastapsp`) | `scripts/gpu_data.py` | `hpc/reference/multi_gpu.sh`; `hpc/run_gpu_split.sh fastapsp` |
| Strong scaling on CPUs | `data/apsp.csv` (2 to 64 threads), `data/external.csv` (`gap_apsp`, `stream`) | `scripts/cpu_data.py` | `hpc/run.sh scaling bandwidth`; `scaling32`, `scaling64` |
| Queries on the DIMACS road networks | `data/queries.csv`, `data/external.csv` (`gap_query`, `phast`, `cch`) | `scripts/cpu_data.py` | `hpc/run.sh queries external` |
| Widest paths on CPUs | `data/bottleneck.csv`, `data/external.csv` (`boost_widest`) | `scripts/cpu_data.py` | `hpc/run.sh bottleneck` |
| Other semirings on the GPU | `data/gpu/gpu_semirings.csv`, `data/gpu/gpu_semirings_kernels.csv` | `scripts/gpu_data.py` (`\GpuSr...`) | `hpc/run_gpu.sh gpu_semirings` |
| An RTX PRO 6000 and an L4 against ROME | `data/gpu/gpu_apsp.csv` (`gpu.kind` `rtx`, `nvidial4`) | `scripts/gpu_data.py` | `hpc/run_gpu.sh gpu_apsp --gpu PARTITION` |
| One traced call, step by step | `data/gpu/gpu_timeline_trace.csv` | `scripts/fig_gpu_timeline.py` | `hpc/run_gpu.sh gpu_timeline` |

The output of each script:

| Script | Writes |
|---|---|
| `scripts/table_graphs.py` | `build/tables/tab-graphs.tex` |
| `scripts/fig_apsp_combined.py` | `build/figures/fig-apsp-combined.dat` |
| `scripts/cpu_data.py` | `build/data/kernels.dat`, `scaling_*.dat`, `reuse.dat`, and the macros `\Bottleneck...` |
| `scripts/gpu_data.py` | `build/data/gpu_kernels.dat`, `build/data/gpu_phases_1col.tex`, `build/tables/tab-gpu-baselines.tex`, `build/tables/tab-gpu-multigpu.tex` |

## Data dictionary

The common columns: `file` (the raw result file), `commit` (the library version label), `machine`, `date`, `job`, and
`provenance.*` (Julia and CliqueTrees versions, threads, placement). The other columns the scripts read:

| File | Columns |
|---|---|
| `gpu/gpu_apsp.csv`, one GPU | see below |
| `gpu/gpu_multigpu.csv`, one to eight GPUs | `code` (`ours`, `fastapsp`), `graph`, `gpu.count`. Times: `t.median` (whole call), `t_closure` (the star alone), `t_reuse` (new weights on the same graph). `closure_imbalance`; `status`, `check.ok`; Fast-APSP's `K` |
| `gpu/gpu_kernels.csv` | `op` (`gemm`, `trsmL`, `trsmU`, `trtri`, `copy`), `impl` (`ours`, `cublas`, `cusolver`, `cuda`), `semiring`, `T`, `n`, `time` (s), `gmuladd` (G multiply-adds/s) |
| `gpu/gpu_semirings.csv` | `semiring`, `eltype`, `graph`, `t_cold.median`, `check.ok`; `tag` and `variant` (the reference rows: tags `sr`/`sr2` with variant `pair1`, which is `lib-g`'s default) |
| `gpu/gpu_semirings_kernels.csv` | `op`, `impl`, `semiring`, `T`, `n`, `time`, `gmuladd`, `tag`, `variant` |
| `apsp.csv`, all-pairs on CPUs | `graph`, `n`, `m`, `nnzLU`, `fill`, `ncc` (number of components). Times (s): `t_sym`, `t_num`, `t_inverse`, `t_solve`, `t_closure` (inverse + solve), `t_alloc` (first touch of the output), `t_perm` (renumbering; excluded), `t_dijkstra_all` (n Dijkstra runs). `file` ends in `_nt<threads>.json` |
| `queries.csv` | `graph`, `n`, `m`, `nnzLU`, `fill`, `t_sym`, `t_num.1`, `t_num.16`, `single.{source,target}_nt{1,16}.median`, `dijkstra_{source,target}.median`, `blocks.k<k>_nt<t>.{general,subtree}_{col,row}` (per block) |
| `bottleneck.csv` | as `apsp.csv`, in the (max, min) semiring with random capacities (`capacity_seed`) |
| `kernels.csv` | `semiring` (`MinPlus`, `MaxMin`, `sgemm`, `dgemm`), `T`, `n`, `time`, `gmuladd` |
| `external.csv` | `tool` (`gap_apsp`, `gap_query`, `phast`, `cch`, `boost_widest`, `stream`), `graph`, `threads`, `placement`, `variant`, `metric`, `value` |

The columns of `gpu/gpu_apsp.csv`:
- **Identity:** `code`, `graph`, `n`, `m` (arcs), `nnzLU`, `fill` = nnz(L+U)/nnz(A), `eltype`, `ordering`, `auto_choice`
  (the ordering `auto` picked), `gpu.kind`.
- **Status:** `status` and `check.ok` (passing rows only are used); `extrapolated` (true: a sampled run scaled to n,
  never used).
- **Codes:**
  - ours: `ours-<ordering>`, `-cpufactor` (the factorization on the CPU only), `-phases` (per-step medians);
  - ROME's builds: `rome-default`, `rome-leaf400`, `rome-leaf800`;
  - the other baselines: `fw`, `cugraph`, `ecl`, `fastapsp1`, `adds`, `gunrock`.
- **Times (s):**
  - ours: `t_cold.median` (also `.min`, `.max`, `.all`);
  - ROME: `t_match.median` (its tree, its allocations without the output, its computation); `t_rome.median` (ROME's own
    scope, without allocations); `t_full.median` (with the output's allocation);
  - the other baselines: `t.median`;
  - `phases.*` (`symbolic`, `ordering`, `numeric_setup`, `numeric_cpu`, `numeric_gpu`, `transfer`, `solve_setup`,
    `inverse`, `solve`, `relabel`, `other`, `total`).

Graphs (`bench/graphs.jl`):
- **SuiteSparse matrices** keep SuperFW's weights, |a_ij|, with 1 for pattern matrices.
- **The DIMACS networks** keep their travel times.
- **The others** get uniform random weights in [1, 100].
- **SNAP graphs** are reduced to their largest weakly connected component, so email-Enron has n = 33,696 here; ROME's
  authors list 36,692.

## Regression check

Before release we reran the benchmarks with this folder, set up from scratch with the scripts above, on the same kind of
nodes as the reference runs (Oct 2026), and compared the results with `scripts/regress.py`. Ratios are rerun /
reference.

| Experiment | Rerun | Result |
|---|---|---|
| GPU test suite (`gpu_tests`) | 11 test files, two B200s | all pass, including the register-spill check (with `bench/gpu/tests.Manifest.toml`, the reference versions) |
| GPU dense kernels (`hpc/reference/kernels.sh`) | 70 kernels and sizes | 1.006 (0.94 to 1.07) |
| GPU all-pairs, ours (`hpc/reference/one_gpu.sh`) | 36 graphs, ordering `auto` | 1.03 (0.96 to 1.09); n and m identical, nnz(L+U) identical on 34 graphs and within 0.5% on net4-1 and thermomech_dM (METIS, see above) |
| GPU all-pairs, ROME | 35 graphs, its fastest build | n of 60,000 or more: 0.91 to 1.04; smaller graphs down to 0.44 (USpowerGrid; see below) |
| Speedup over ROME | | large set 6.33x, as in the reference; ROME's 16 graphs 2.63x (reference 3.12x); all 26 2.50x (reference 2.83x) |
| GPU time per step (`hpc/reference/phases.sh`) | 36 graphs | the GPU steps as in the reference (inverse 0.94, solve 0.97, numeric 1.06); the CPU steps 1.2 to 1.4 (below) |
| GPU semirings | 140 kernel timings; 48 closures | 1.001 (1.00 to 1.05); closures, as their cost relative to min-plus: 0.97 (0.85 to 1.08) |
| CPU all-pairs and queries | 28 and 11 graph runs | 0.91 and 1.05 against `lib-a` and `lib-b`; nnz(L+U) differs by -10% to +3.5% (the versions order differently; see "Versions of the library in the data") |
| `quick.sh cpu` | a laptop, from a fresh download | 7 of 7 graphs exact, every check passing, under 3 minutes |
| `quick.sh gpu --rome` | one B200, a fresh install | 16 of 16 graphs exact, every check passing, ROME passing on all 16; 2.69x over ROME (reference 3.12x), faster on 16 of 16 |
| The quick starts of `SOLVER.md` | one B200 | both pass |

These one-GPU reruns held 16 cores of a node that other users' jobs shared. The reference runs held the whole node. On
the shared nodes, the CPU-side steps (symbolic analysis, the CPU's part of the factorization, transfers) were 20 to 40%
slower, and so was the CPU-only factorization on two graphs. The GPU-side steps and the headline times were not.

ROME's time on the small graphs varies from run to run. In the reference job, ROME's runs on c-42 fell into two groups,
25 to 30 ms and 52 to 57 ms, and the median was in the slower one; in the rerun it was 29 ms. On USpowerGrid, ROME took
11 to 17 ms in the reference job and 4 to 8 ms in the rerun. Our times did not vary this way. With ROME's fastest median
from any of the reference B200 jobs instead of the last one's, the speedups over ROME's 16 graphs and all 26 are 2.76x
and 2.58x (`GPU_ROME_RULE=best`, `scripts/fig_apsp_combined.py`).

## Caveats

- **Timing noise:** about 15% from one B200 node to another. In the reference runs, ROME ran right after ours on the
  same node.
- **ROME's builds:** `hpc/reference/one_gpu.sh` runs ROME's fastest passing build per graph, from earlier runs (its array
  `RB`). `scripts/gpu_data.py` takes the fastest passing build at `lib-g`.
- **ROME's failures:** ROME's distances fail our check on delaunay_n17, rgg_n_2_17_s0, and thermomech_dM; the
  comparisons with ROME exclude these graphs.
- **Baselines that do not complete:**
  - ADDS is killed after 900 s on 7 graphs.
  - ECL-APSP computes integer distances only, so it runs only on graphs with integer weights.
  - The comparisons with the other GPU codes use only graphs each code completes with all n sources timed.
- **Precision:** GPU results are `Float32`. 64 sampled columns per graph are checked against Dijkstra's algorithm and the
  CPU star in `Float64` (the largest relative error over the `lib-g` one-GPU rows is 3.8e-7, `check.dijkstra_maxrel`);
  integer weights are checked exactly by row hashes.
- **Graph sizes:** the largest outputs need several GPUs. USA-t-COL takes 759 GB on 5 or more B200s, and delaunay_n19
  1.1 TB on 7 or 8.
