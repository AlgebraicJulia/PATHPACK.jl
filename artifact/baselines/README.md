# Baselines

Every code the benchmarks compare against, with exactly our changes. Third-party code is not copied here: it is fetched
from upstream at the commit we used. This folder holds only our drivers, patches and build scripts.

| Where | What |
|---|---|
| `gap_apsp.cpp`, `bottleneck_boost.cpp`, `phast_driver.cpp`, `cch_driver.cpp` | CPU drivers (built by `hpc/setup.sh`, `hpc/baselines.sh`) |
| `rome/` | ROME: `fair.patch`, `instr.patch`, `rome_patch.py`, `prepare_rome.sh` |
| `gpu/` | the other GPU baselines: drivers and build scripts (`src/`, `jobs/`), `fetch_ext.sh`, `envs/`, `gpu_fingerprint.py` |

Toolchain of the reference runs: CUDA 12.9.1, CMake 3.30.5, the system GCC 11.5 for ROME and its libraries, GPU code for
sm_100 (B200) and sm_120. Every baseline's result is checked against ours: GPU codes write per-row fingerprints after
their timers (`gpu/gpu_fingerprint.py` adds them: unreachable count, sum, sampled columns), which
`bench/gpu_baselines.jl` compares with ours under the benchmarks' relative tolerance; integer-weight graphs are also
row-hashed (reference hashes in `GB_ROWHASH_DIR`). A run whose result disagrees is reported as failed, not timed.

## ROME (GPU)

- Upstream: <https://github.com/LyleLuo/ROME> at `25beef3260a52725803405208dd64f76dd385f0b`, with its submodules METIS
  5.1.0.3 (<https://github.com/scivision/metis> at `7f8f9b0d765c15efe3eefe3e9db0a9faf9a5f480`) and Scotch v7.0.10
  (<https://gitlab.inria.fr/scotch/scotch> at `7fd71f5eee1911b3b1c000dad1f880c0a0a41c2c`).
- `fair.patch`: weights read from the input file (upstream ignores them and draws random weights 1 to 10); the CUDA
  architectures as a CMake option `ROME_ARCHS` (upstream: sm_80 to sm_90 only); `ROME_LOWMEM` (one n x n pinned host
  buffer instead of two); `ROME_SKIP_CHECK` (no CPU reference check); `ROME_ROWHASH` and `ROME_DUMP_ROWS` (verification
  output, after all timers); a separate end-to-end timer. ROME's timers, kernels and launch logic are unchanged.
- `rome_patch.py` (applied to each copy at build time): `ROME_DEVICE_ONLY=1`, the result stays in GPU memory as ours does
  (ROME's "data transfer to host" timer then times nothing), and `ROME_DUMP_SAMPLE`, sampled rows in the original labels
  for the check. Both act after all of ROME's timers.
- `instr.patch` (on top of `fair.patch`; one extra run per graph): host timers, per-level counters and NVTX ranges, used
  only to split ROME's allocation time. Kernels unchanged.
- Builds: Scotch with `CMAKE_BUILD_TYPE=Release` (-O3; the README's build sets no build type), otherwise the README's
  options. Four binaries: `rome_default` (Scotch separator leaf size 150, as shipped), `rome_leaf400`, `rome_leaf800`
  (the leaf size edited in `src/ETree/ETreeOptimized.cpp`; the comparisons use the fastest of the three per graph) and
  `rome_instr`.
- Build: `rome/prepare_rome.sh DIR` (clones, patches, builds METIS and Scotch; prints `ROME_SRC`, `ROME_SRC_INSTR`,
  `ROME_MODULES`, `ROME_SCOTCH`), then `hpc/setup_gpu.sh rome`.
- Run and check: target `gpu_apsp` (`rome_runs` in `hpc/gpu_baselines.sh`), then `bench/gpu_baselines.jl rome`. Timed
  by ROME's own timers (tree build, allocation and initialization, computing).

## Fast-APSP (GPU, 1 to 8 GPUs)

- Upstream: <https://github.com/Liu-xiandong/FastAPSP> at `eec2af2b8939e6cbdc10ae6bbfabfa6d1231a0d3`.
- `gpu/src/fastapsp/fastapsp.patch`: phase timers, three missing `return` statements (undefined behavior at -O2), and two
  fixes: the SSSP kernel reads `n + 1` row offsets but `n` were allocated, and the `numFrom * vertexs` size of its
  result buffer (allocation and copy) overflowed 32-bit integers.
- Our drivers: `fap_driver` (one GPU) and `fap_mpi_driver` (one MPI rank per GPU, as its `MoreTaskMpiExample`), and
  `mtx2fap` (graph conversion). Timed scope: conversion from CSR, METIS partitioning, preconditioning and solve, with the
  output allocated inside the timer. Fast-APSP indexes with 32-bit integers, so it needs at least n^2/2^31 partitions K;
  K is swept per graph and GPU count and the fastest run is reported (`fap_runs` in `hpc/gpu_baselines.sh`).
- Build: `gpu/jobs/build_fastapsp.sh` (CUDA 12.9.1, METIS 5.1.0; the MPI driver with GCC 12.2.0 and OpenMPI 5.0.7).
- Run and check: targets `gpu_fastapsp` and `gpu_apsp` (one GPU), then `bench/gpu_baselines.jl`.

## cuGraph (GPU)

- RAPIDS libcugraph 26.08.00 for CUDA 12.9 (conda). `gpu/envs/rapids.explicit.txt` is the exact package list of the
  environment we built and ran against; `gpu/envs/make_rapids_env.sh PREFIX` recreates it.
- `gpu/src/sssp/cugraph/cugraph_apsp.cu`: `od_shortest_distances` from batches of origins to all vertices until all n
  rows are done (`CUGRAPH_MODE=od`), RMM pool allocator, float weights.
- Build: `gpu/src/sssp/cugraph/build_cugraph.sh`. Run: target `gpu_apsp` (`others_runs`).

## Gunrock (GPU)

- Upstream: <https://github.com/gunrock/gunrock> at `748f79e66cc26762cd3f99422b028e742a737bde` (v2.2.0-13, essentials
  API), with cxxopts v3.0.0 (`c74846a891b3cc3bfa992d588b1295f528d43039`) and nlohmann/json v3.11.3
  (`9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03`).
- `gpu/src/sssp/gunrock/gunrock_apsp.cu`: Gunrock's SSSP from every source. Built with Gunrock's own release flags;
  `SM_TARGET=90` because its launch-parameter table stops at sm_90 (every operator uses the fallback parameters, so the
  launch configuration is the same on both GPUs).
- Build: `build_gunrock.sh`. Run: target `gpu_apsp`. When a full run is estimated above `SSSP_BUDGET` seconds, a sample
  of sources is timed and extrapolated to n, and the result is marked so (also for ADDS and cuGraph).

## ADDS (GPU)

- Upstream: <https://github.com/UCLA-VAST/adds> at `9ef8166e87d3f13cb458ce2f128bc862e841200b` (float version).
- Porting edits only (`build_adds.sh`): `MAX_TB` 512 (the B200's 148 and the RTX PRO 6000's 188 SMs need more thread
  blocks than its cap), the unused Boost include dropped, and the CUDA toolkit's CUB instead of the bundled CUB 1.8.0,
  which does not compile with CUDA 12.9. Kernels, worklist, Delta heuristic and launch configuration unchanged; our
  driver `adds_apsp.cu` loops over sources with its exact per-source sequence.
- Build: `build_adds.sh`. Run: target `gpu_apsp`.

## Dense Floyd-Warshall (GPU)

- Ours: `gpu/src/fw/fwbench.cu --algo tuned`, a tuned blocked min-plus Floyd-Warshall in float, O(n^3). It is compiled
  with cuASR (<https://github.com/hpcgarage/cuASR> at `45b34c7631d1324bf15a6fd907063f2334f8ed88`, CUTLASS v2.3.0 as its
  submodule) because the same file also holds cuASR and Lund-Smith variants, which the benchmarks do not use.
- Build: `gpu/src/fw/build.sh`. Run: target `gpu_apsp`.

## ECL-APSP (GPU)

- ECL-APSP v1.0 from the authors' page, <https://userweb.cs.txstate.edu/~burtscher/research/ECL-APSP/>
  (`ECL-APSP_10.cu`, `ECLgraph.h`; `fetch_ext.sh` checks their SHA-1).
- `gpu/src/fw/ecl/make_ecl_kernels.py` extracts its kernels with 64-bit indices (its int indices overflow once
  n > 46,340) and a grid-stride initialization; same tiles, kernels and order. Integer distances only, so it runs on the
  integer-weight graphs.
- Build: `gpu/src/fw/ecl/build_ecl.sh`. Run: target `gpu_apsp`.

## GAP Delta-stepping (CPU)

- Upstream: <https://github.com/sbeamer/gapbs> at `2972aeb2703165bafd921222f4ed7196f542d3a8`. `hpc/setup.sh` makes the
  weights `double` and makes the builder stop on unweighted input (instead of inserting random integer weights).
- `gap_apsp.cpp`: GAP's `DeltaStep` (from `sssp.cc`, unmodified) from every source, each result copied into its row of an
  n x n output allocated and first-touched before the timer; n single-threaded runs spread over the threads. Delta tuned
  per graph over the mean arc weight times 2^k, k = -6..9 (`hpc/baselines.sh`). Queries use GAP's own `sssp` binary.

## Dijkstra (CPU)

- The solver's own generic label-setting code, n runs, one per target (`bench/lib.jl`, `bench/apsp.jl`). Not third-party.

## Boost (CPU, widest paths)

- Boost 1.86.0 (release archive `boost-1.86.0-b2-nodocs.tar.gz`, header-only Graph library).
- `bottleneck_boost.cpp`: Boost's generic `dijkstra_shortest_paths`, unmodified, with combine = min and compare = greater,
  on its compressed-sparse-row graph. Build: `hpc/setup.sh`, `hpc/baselines.sh`.

## PHAST and CCH (CPU, RoutingKit)

- Upstream: <https://github.com/RoutingKit/RoutingKit> at `54d49bb0cdea56dde182357522e4e86a03c57852`.
- `phast_driver.cpp`: PHAST on RoutingKit's contraction hierarchy (upward search, then one linear downward sweep; k-source
  layout), validated against Dijkstra. `cch_driver.cpp`: customizable contraction hierarchies with an inertial-flow
  nested-dissection order; the four phases timed separately. Build: `hpc/setup.sh` (copied into RoutingKit, `make`).

## STREAM (CPU)

- Upstream: <https://github.com/jeffhammond/STREAM> at `6703f7504a38a8da96b353cadafa64d3c2d7a2d3`, unmodified;
  `gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20`. Triad, 1 and 16 threads in one NUMA region
  (target `bandwidth`).

## Rebuilding the GPU baselines

```bash
baselines/rome/prepare_rome.sh $WORK/rome-src        # then set ROME_SRC, ROME_SRC_INSTR, ROME_MODULES, ROME_SCOTCH
./hpc/setup_gpu.sh rome                              # rome_default, rome_leaf400, rome_leaf800, rome_instr
./hpc/setup_gpu.sh others                            # fetch_ext.sh, cuGraph environment, cuGraph, Gunrock, ADDS,
                                                     # dense FW, ECL-APSP, Fast-APSP
./hpc/setup_gpu.sh check
```

The earlier comparison these drivers come from also built other dense Floyd-Warshall codes; the benchmarks do not use
them, and they are not needed here.
