# PATHPACK benchmarks

PATHPACK solves algebraic path problems (all-pairs shortest paths, widest paths, reachability, and other semirings)
by a sparse LU factorization over the semiring, on CPUs and NVIDIA GPUs. This repository benchmarks it against
state-of-the-art codes:

- **GPUs:** ROME, Fast-APSP, cuGraph, Gunrock, ADDS, ECL-APSP, and a dense Floyd-Warshall.
- **CPUs:** n runs of Dijkstra's algorithm, GAP's Delta-stepping, PHAST, CCH, and Boost.

It holds the solver as measured (`src/`, `ext/`; see [SOLVER.md](SOLVER.md)), the benchmark harness, and the reference
measurements (`artifact/`), from nodes of 8 NVIDIA B200 GPUs and a node with two 64-core AMD EPYC 7702 processors.

## Check the numbers (seconds)

    make -C artifact check

This recomputes every derived number (670 of them), table, and figure-data file from `artifact/data/`, and compares the
numbers with `artifact/reference/`: none differ. It needs Python 3 with numpy and pandas.

    python3 artifact/scripts/rome_report.py artifact/data/gpu/gpu_apsp.csv

This prints the speedup over ROME on one B200, per graph and per set of graphs.

## Benchmark on your machine (minutes)

    bash artifact/quick.sh cpu            # ~5 min: the solver and n runs of Dijkstra's algorithm, 7 graphs
    bash artifact/quick.sh gpu --rome     # ~30 min: the solver and ROME on your NVIDIA GPU, ROME's 16 test graphs

These run on one machine, without Slurm or configuration. The script installs the Julia packages at the reference
runs' versions, downloads only the graphs it runs, runs the benchmark drivers with their checks, and prints your results
next to the reference ones. It needs Linux, Julia 1.12, curl, and Python 3; `--rome` also needs git, gcc, CMake, and the
CUDA toolkit ([details](artifact/README.md#benchmark-on-your-machine)).

## Rerun everything (a Slurm cluster)

1. Set the account, the partitions, and the scratch space in `artifact/hpc/config.sh`.
2. `cd artifact && bash hpc/setup.sh && bash hpc/setup_gpu.sh`: the Julia environments, the baselines (built from their
   authors' sources with our patches), and the graphs (about 5 GB).
3. Submit `bash hpc/run.sh TARGET` and `bash hpc/run_gpu.sh TARGET` (without a target, they list them), or the scripts
   of the reference GPU runs in `hpc/reference/`.
4. Compare with the reference measurements: `make -C artifact regress`.

[artifact/README.md](artifact/README.md) has the details: the targets, the timing conventions, the versions behind each
measurement, the data dictionary, and our own rerun of these steps.

## Layout

| Path | Contents |
|---|---|
| `src/` | the solver's CPU code (`src/CPU/`) and the declarations of its GPU functions (`src/GPU/GPU.jl`) |
| `ext/` | the solver's GPU code: a CUDA.jl package extension, which Julia loads with `using CUDA` ([SOLVER.md](SOLVER.md#layout)) |
| `test/` | the solver's tests; `test/gpu/`: the GPU tests |
| `artifact/quick.sh` | the benchmark on one machine |
| `artifact/bench/` | the benchmark drivers, and their environments at the reference runs' package versions |
| `artifact/hpc/` | the Slurm harness: settings, setup, one file per experiment, the reference GPU runs |
| `artifact/baselines/` | the competitors' drivers and our patches to them |
| `artifact/data/` | the reference measurements, as CSV |
| `artifact/scripts/`, `artifact/reference/` | measurements to numbers, tables, and figure data; the reference values |
