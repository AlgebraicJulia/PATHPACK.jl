# Benchmark drivers

The targets in `../hpc/targets/` run these drivers; each driver's header gives its exact protocol. Every driver writes
one JSON file per measurement into `$BENCH_RESULTS`, and `../scripts/collect.py` turns those files into CSV rows.

**Environments:**
- `Project.toml`: the CPU drivers, with this repository's library;
- `gpu/Project.toml`: the GPU drivers, with CUDA.jl as well.

`../hpc/setup.sh` and `../hpc/setup_gpu.sh` instantiate them.

## CPU drivers

| Driver | Measures |
|---|---|
| `apsp.jl` | all-pairs shortest paths, phase by phase (symbolic, numeric, inverse U*, solve L*), against n runs of Dijkstra's algorithm; 64 columns checked against Dijkstra |
| `queries.jl` | single-source and single-target queries, and blocks of right-hand sides in both layouts, against Dijkstra |
| `bottleneck.jl` | all-pairs widest paths in the (max, min) semiring, with a random capacity per arc, against n widest-path Dijkstra runs |
| `kernels.jl` | dense semiring kernel rates (`sgemx!`) on one core, against OpenBLAS `dgemm` |
| `gate.jl` | determinism: every timed result bit-identical on 1, 2, 4, and 8 threads |

## GPU drivers

| Driver | Measures |
|---|---|
| `gpu_apsp.jl` | all-pairs on one GPU through `apsp_gpu!`. Variants are orderings and hybrid or CPU-only factorization. Each result is checked against Dijkstra, against the CPU star, and by row hashes |
| `gpu_phases.jl` | the time per step of the one-GPU call: the median of 7 synchronized cold calls per graph |
| `gpu_multigpu.jl` | all-pairs on one to eight GPUs of one node: the whole call, the star alone, and new weights |
| `gpu_kernels.jl` | the GPU's dense kernels against cuBLAS and cuSOLVER, `Float32` |
| `gpu_baselines.jl` | the GPU baselines' runs, checked against our results and turned into result files |
| `gpu_gate.jl` | every entry point bit for bit against reference row hashes, on integer-weight graphs |
| `semirings/gpu_semirings.jl` | the closure on every semiring and element type |
| `semirings/sr_check.jl` | the CPU check of the semiring runs |
| `semirings/build_csv.py` | the CSVs of the semiring runs |
| `timeline/measure2.jl` | the timeline of one call: a host trace of every step, CUDA events for the GPU's work |

## Shared files

- `graphs.jl`: the graph sets, readers, and weight conventions.
- `lib.jl`: environment, timing, provenance, result files, and the Dijkstra reference.
- `gpu_lib.jl`: the GPU's description, the timing conventions (warm and cold calls), the checks, and the graph export for
  the baselines.

A driver can also run without Slurm: `../quick.sh` does this for `apsp.jl` and `gpu_apsp.jl` (and ROME). By hand, from
this folder, with the graphs that `../hpc/fetch_data.sh [GRAPH...]` downloads in the data folder:

    BENCH_DATA=<data folder> BENCH_RESULTS=<output folder> BENCH_NT=16 julia -t 16 --project=. apsp.jl wing
    BENCH_DATA=<data folder> BENCH_RESULTS=<output folder> BENCH_NT=8 julia -t 8,0 --project=gpu gpu_apsp.jl wing
