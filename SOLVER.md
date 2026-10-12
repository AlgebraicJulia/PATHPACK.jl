# The solver: PATHPACK.jl

PATHPACK computes the star of a sparse matrix over a semiring, A* = I ⊕ A ⊕ A² ⊕ ⋯ (the algebraic path problem). If
A[i, j] is the weight of the arc i → j, then in min-plus arithmetic A*[i, j] is the distance from i to j. Other semirings
give widest paths, path sums, reachability, and more.

PATHPACK factorizes the star, A* = U*L*, by a sparse multifrontal LU factorization over the semiring: a fill-reducing
ordering and a symbolic analysis, then the numeric factorization of each strongly connected component. From the factors
it computes either the whole star (an inverse X ← U*, then a solve X ← XL*) or single-source and single-target queries.

- The CPU code, `PATHPACK.CPU`, is multithreaded.
- The GPU backend, `PATHPACK.GPU`, is a package extension that loads with CUDA.jl. CPU threads factorize the many small
  fronts and the GPU the large ones; the star is computed on one GPU or on several.

This file describes the solver as measured. For the benchmarks, see the [README](README.md).

## Install

You need Julia 1.12; the reference runs used 1.12.6 and 1.12.7. In this folder:

    julia --project=. -e 'using Pkg; Pkg.instantiate()'

or add the package to another environment with `pkg> dev /path/to/this/folder`. The dependencies come from the General
registry: CliqueTrees, Graphs, SIMD, and standard libraries.

For the GPU backend, add CUDA.jl 6 to the same environment (`pkg> add CUDA`) and load both packages. It needs an NVIDIA
GPU. The reference runs used CUDA.jl 6.1.0 on three GPUs:

| GPU | Compute capability |
|---|---|
| B200 | 10.0 |
| RTX PRO 6000 Blackwell | 12.0 |
| L4 | 8.9 |

## Quick start: CPU

The first call of each semiring in a session compiles the solver, about half a minute on a laptop; later calls take
only their own time.

```julia
using PATHPACK, SparseArrays, Random
using PATHPACK.CPU: MinPlus, MaxMin, mstar

Random.seed!(1)
n = 300
A = sprand(n, n, 0.01)                    # A[i, j]: the weight of the arc i -> j
nonzeros(A) .= 1 .+ 99 .* nonzeros(A)     # weights in [1, 100]
D = mstar(MinPlus(), A)                   # D[i, j]: the distance from i to j (Inf: unreachable)

# the same with Floyd-Warshall
F = fill(Inf, n, n); foreach(i -> F[i, i] = 0, 1:n)
for (i, j, w) in zip(findnz(A)...); F[i, j] = min(F[i, j], w); end
for k in 1:n, j in 1:n, i in 1:n; F[i, j] = min(F[i, j], F[i, k] + F[k, j]); end
@assert isinf.(D) == isinf.(F) && maximum(abs, D[isfinite.(F)] - F[isfinite.(F)]) < 1e-9

W = mstar(MaxMin(), A)                    # W[i, j]: the largest capacity of a path from i to j (-Inf: unreachable)
```

The `CPU` module provides the following functions (`src/CPU/abstract_slu.jl`):

| Function | Returns |
|---|---|
| `mstar(s, A)` | the star |
| `mlu(s, A)` | its factorization `F`, with `Matrix(F) == mstar(s, A)` |
| `lmul!(F, B)`, `rmul!(B, F)` | the solves `B ← A*B` and `B ← BA*` |

All of them take `nt`, the number of threads (default: all of Julia's). `mstar` and `mlu` also take `alg`, the ordering
(a CliqueTrees elimination algorithm or a permutation).

## Quick start: GPU

The first call in a session also compiles GPU kernels for the graph at hand, which can take up to a minute.

```julia
using PATHPACK, CUDA, SparseArrays, Random
using PATHPACK.CPU: MinPlus, mstar
using PATHPACK.GPU: apsp_gpu

Random.seed!(1)
n = 5000
I = rand(1:n, 2n); J = rand(1:n, 2n); keep = I .!= J
W = Float32.(rand(1:100, count(keep)))                                    # integer weights: exact in Float32
A = sparse([I[keep]; J[keep]], [J[keep]; I[keep]], [W; W], n, n, min)    # undirected
D = apsp_gpu(A; output = :host)                                           # a Matrix{Float32}
@assert D == mstar(MinPlus(), A)                                          # the CPU star, bit for bit
```

Other calls (docstrings in `src/GPU/GPU.jl` and `ext/PATHPACKCUDAExt/api.jl`):

| Call | Result |
|---|---|
| `apsp_gpu(A)` | the star as a `CuMatrix`, the default `output = :device` |
| `apsp_gpu(A, sources)` | only the rows of these sources |
| `apsp_gpu(A; devices = collect(CUDA.devices()))` | the rows split over several GPUs |
| `apsp_gpu!(D, A)` | the star written into a matrix you allocated |
| `P = apsp_plan(A); apsp_gpu(P, A2)` | new weights on the same graph; whatever depends only on the graph is reused |
| `apsp_gpu(A; semiring = MaxMin())` | another semiring |

Limits on the GPU:
- **Element types:** `Float32` or `Float64`. Integers are accepted only where the semiring's zero absorbs, as for
  `MaxMin` and `OrAnd` with `Int32`; min-plus with integers is rejected, because its infinity overflows.
- **Graphs:** an undirected graph, or a directed one whose strongly connected components do not reach one another. For a
  directed graph with arcs between its components, use the CPU solver.

## Semirings

All are exported by `PATHPACK.CPU`; `src/CPU/semiring/` defines them.

| Semiring | Set, ⊕, ⊗ | A*[i, j] |
|---|---|---|
| `MinPlus` (tropical) | [-∞, ∞], min, + | the shortest distance |
| `MaxPlus` | [-∞, ∞], max, + | the largest path weight |
| `MaxMin` (bottleneck) | [-∞, ∞], max, min | the largest capacity of a path |
| `MinMax` | [-∞, ∞], min, max | the smallest largest arc weight on a path |
| `PlusProd` (probability) | [0, ∞], +, × | the sum over all paths of their products |
| `MinProd`, `MaxProd` | [0, ∞], min or max, × | the smallest or largest path product |
| `OrAnd`, `AndOr` | subsets, ∪ and ∩ (`AndOr`: ∩ and ∪) | reachability, one bit per set element (use integer matrices, e.g. `Int32`; the CPU's vector kernels do not take `Bool`) |
| `MinPlusLaw`, `MaxPlusLaw` | [0, ∞], min, +; [-∞, 0], max, + | Lawvere-quantale variants of the tropical semirings |
| `MinProdLaw`, `MaxProdLaw` | [1, ∞], min, ×; [0, 1], max, × | Lawvere-quantale variants of the product semirings |

`artifact/data/gpu/gpu_semirings.csv` holds the GPU runs of these semirings:
- `Float32`: the tropical, product, bottleneck, probability, and Lawvere semirings;
- `Float64`: `MinPlus`, `MaxMin`, and `PlusProd`;
- `Int32`: `MaxMin`, `OrAnd`, and `AndOr`.

All pass their checks except `MinProdLaw`, which the GPU rejects: in `Float32` its zero does not absorb under ⊗.

## Layout

```
src/PATHPACK.jl            the package: modules CPU and GPU
src/CPU/                   the CPU solver
  semiring/                the semirings (splus, sprod, sstar, szero, sone, ...)
  dense/                   dense kernels: sgemx! (C ← C ⊕ A ⊗ B), sgetrf!, strsx!, strtri!, sgetri!
  sparse/                  sparse-times-dense products
  fast_orderings.jl        orderings AutoOrder, HubAMF, BFSND (AMF and METIS come from CliqueTrees)
  chordal_ssymbolic.jl     symbolic analysis: components, clique tree (ssymbolic)
  parallel_symbolic.jl     the symbolic analysis on several threads
  chordal_slu.jl           the factorization ChordalSLU: copyto!, sgetrf!, sgetrs!, sgetre!, sgetri!
  chordal/                 one strongly connected component: multifrontal factorization, solves, star
  blocked/                 solves and the star across components
  dense_slu.jl             DenseSLU, the dense case
  abstract_slu.jl          mlu, mstar, pstar; lmul!, rmul!, ldiv!, rdiv!
  dijkstra.jl, bellman.jl  generic Dijkstra and Bellman-Ford (references and baselines)
src/GPU/GPU.jl             the GPU API, declared here; its methods are in ext/
ext/PATHPACKCUDAExt/        the GPU backend
  runtime/                 settings (GPUConfig, with_config), device profile, CUDA graphs, precompilation
  dense/                   semiring GEMM, dense LU, triangular solves
  batched/                 kernels over every front of a tree level in one launch
  chordal/                 hybrid factorization, the star's sweeps
  multigpu.jl              several GPUs
  api.jl                   apsp_gpu, apsp_gpu!, apsp_plan
test/                      tests; test/gpu: the GPU suite
artifact/                  the benchmarks
```

## Where each part of the method is

GPU paths below are under `ext/PATHPACKCUDAExt/`.

| Part | Code |
|---|---|
| semirings and stars | `src/CPU/semiring/` |
| ordering and symbolic factorization | `src/CPU/fast_orderings.jl` (`AutoOrder`, `HubAMF`, `BFSND`), `src/CPU/chordal_ssymbolic.jl` (`ssymbolic`), `src/CPU/parallel_symbolic.jl` |
| numeric factorization: BlockGetrf, ChordalGetrf, DenseGetrf | `sgetrf!` in `src/CPU/chordal_slu.jl`, `src/CPU/chordal/sgetrf.jl`, `src/CPU/dense/sgetrf.jl` |
| solves: BlockGetrs, ChordalTrsm, DenseTrsm | `sgetrs!` in `src/CPU/chordal_slu.jl` and `src/CPU/blocked/sgetrs.jl`, `src/CPU/chordal/strsx.jl`, `src/CPU/dense/strsx.jl` |
| the star, X ← U* then X ← XL* | `sgetri!`: `src/CPU/chordal/sgetri.jl` (`strtri!`, then `strsx!`), `src/CPU/blocked/sgetri.jl` (across components); `mstar` in `src/CPU/abstract_slu.jl` |
| queries | `sgetre!` (`src/CPU/chordal/sgetre.jl`, `src/CPU/blocked/sgetre.jl`); blocks: `sgetrs!` with a `TDWorkspace` (`src/CPU/chordal_slu.jl`) |
| parallelism on CPUs | the `_mt!` functions of the files above; dense kernels in `src/CPU/dense/` (GEMM: `sgemx.jl`) |
| GPU: hybrid factorization | `chordal/sgetrf.jl` (`FactorPlan`, `factorize!`), `chordal/sgetrf_top.jl` (the top of the tree, level by level), `batched/sgetrf.jl`, `dense/sgetrf.jl` |
| GPU: semiring GEMM | `dense/sgemx.jl`, `dense/sgemx_simt.jl`, `dense/sgemx_rows.jl` |
| GPU: inverse X ← U* | `chordal/sgetrs.jl` (`closure_gpu!`, `sssp_gpu!`), `chordal/toprows.jl` (a top front updates only the rows that need it), `batched/sweep.jl` |
| GPU: solve X ← XL* | `chordal/sgetrs.jl`, `chordal/layered.jl` (layers of subtrees), `chordal/amalgamate.jl` (relaxed supernodes), `dense/strsx.jl`, `dense/strsx_diag.jl` |
| GPU: several GPUs | `multigpu.jl` (`closure_multigpu!`) |
| GPU: the whole call | `api.jl` (`apsp_gpu`, `apsp_gpu!`, `apsp_plan`) |
| benchmarks | `artifact/` ([README.md](README.md), [`artifact/README.md`](artifact/README.md)) |

## Tests

    julia --project=. -e 'using Pkg; Pkg.test()'

runs Aqua's code-quality checks and the CPU tests (`test/core.jl`: the star in min-plus, max-min, and Boolean arithmetic
against Floyd-Warshall on random directed graphs). Without a working GPU the GPU test set is reported as broken.

The GPU suite needs an NVIDIA GPU. It runs each test file in a fresh Julia process and writes one log per file:

    julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'
    julia --project=test/gpu test/gpu/runtests.jl quick        # or full; --list shows what runs

The benchmark harness also checks every result it times:
- `artifact/bench/apsp.jl` checks 64 columns of every star it times against Dijkstra's algorithm;
- `artifact/bench/gate.jl` requires bit-identical results on 1, 2, 4, and 8 threads.
