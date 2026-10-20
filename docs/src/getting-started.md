# Get Started

## Install

PATHPACK.jl needs Julia 1.10 or later. In the Julia REPL, press `]` and run

```
pkg> add https://github.com/AlgebraicJulia/PATHPACK.jl
```

For NVIDIA GPUs, also add CUDA.jl (`pkg> add CUDA`); the GPU solver loads when both packages are loaded.

To try it without installing anything, open the repository in GitHub Codespaces: a VS Code in your browser with Julia
and PATHPACK.jl ready, and every tutorial runnable as is. It uses the free Codespaces hours of your GitHub account.

```@raw html
<p><a href="https://codespaces.new/AlgebraicJulia/PATHPACK.jl?quickstart=1"><img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub Codespaces" style="height: 32px"></a></p>
```

## Your First Star

A graph is a sparse matrix: `A[i, j]` is the weight of the arc `i → j`. `mstar` takes a semiring and the matrix, and
returns the dense matrix of answers for every pair of vertices.

```@example start
using PATHPACK, SparseArrays, LinearAlgebra
using PATHPACK.CPU: MinPlus, MaxMin, mstar, mlu

I = [1, 1, 2, 3, 4, 4]            # arcs i → j
J = [2, 3, 4, 2, 1, 5]
V = [3.0, 8.0, 1.0, 4.0, 2.0, 7.0]  # their weights
A = sparse(I, J, V, 5, 5)
D = mstar(MinPlus(), A)
```

`D[i, j]` is the length of a shortest path from `i` to `j`; `Inf` means that `j` cannot be reached from `i`. An
undirected graph is a symmetric matrix: store each edge in both directions.

## Choose the Question

The semiring decides what "best path" means. The call stays the same: pick one below.

```@raw html
<CodeExplorer id="semirings" />
```

| Semiring | ⊕, ⊗ | `A*[i, j]` |
|---|---|---|
| [`MinPlus`](@ref PATHPACK.CPU.MinPlus) | min, + | the length of a shortest path |
| [`MaxPlus`](@ref PATHPACK.CPU.MaxPlus) | max, + | the length of a longest path (no positive cycles) |
| [`MaxMin`](@ref PATHPACK.CPU.MaxMin) | max, min | the capacity of a widest path |
| [`MinMax`](@ref PATHPACK.CPU.MinMax) | min, max | the smallest largest weight on a path |
| [`MaxProd`](@ref PATHPACK.CPU.MaxProd) | max, × | the probability of the most reliable path |
| [`PlusProd`](@ref PATHPACK.CPU.PlusProd) | +, × | the sum over all walks, `(I - A)⁻¹` |
| [`OrAnd`](@ref PATHPACK.CPU.OrAnd) | ∪, ∩ | the labels a path carries; reachability |

## Only Some Rows

The result has `n²` entries. If you need a few sources, or many queries on one graph, factorize once and solve for
what you need.

The unit row vector at a vertex (the semiring's one, `0`, there; its zero, `Inf`, elsewhere) becomes that vertex's row
of `A*`. Pick the source:

```@raw html
<CodeExplorer id="rows" />
```

`rmul!(B, F)` computes rows (`B A*`) and `lmul!(F, B)` columns (`A* B`), for many right-hand sides at once.

## From Graphs.jl

Any weighted graph becomes a sparse matrix. With Graphs.jl and SimpleWeightedGraphs.jl, for an undirected graph:

```julia
using Graphs, SimpleWeightedGraphs
g = SimpleWeightedGraph(10)            # ... add_edge!(g, u, v, w) ...
A = sparse(weights(g))                 # symmetric: A[i, j] = A[j, i] = the weight of the edge
D = mstar(MinPlus(), A)
```

For a directed graph, make sure `A[i, j]` holds the weight of the arc `i → j` (some libraries store the transpose).

For an unweighted graph, use the adjacency matrix with weight 1 on every arc: `A = Float64.(adjacency_matrix(g))`.

## Threads

The solver uses every thread Julia was started with. Start Julia with `julia --threads=auto`, or pass `nt` to choose:
`mstar(MinPlus(), A; nt = 8)`.

## GPUs

```julia
using CUDA, PATHPACK
using PATHPACK.GPU: apsp_gpu

D = apsp_gpu(A)                                         # every pair, on the GPU (a CuMatrix)
D = apsp_gpu(A; output = :host)                         # ... copied to a Matrix
D = apsp_gpu(A; devices = collect(CUDA.devices()))      # rows split over every GPU
```

The [GPU tutorial](generated/gpus.md) walks through one and several GPUs, and [`apsp_gpu`](@ref PATHPACK.GPU.apsp_gpu) in the library reference lists sources, plans and supported types.

## How Big Can a Graph Be?

The time grows far slower than `n³`: road networks with hundreds of thousands of intersections take well under a second
on GPUs. The limit is usually memory: the full answer has `n²` entries, `8n²` bytes in `Float64` and `4n²` in
`Float32`. That is 20 GB at 50,000 vertices and 80 GB at 100,000 (in `Float64`). Beyond that, compute blocks of rows
with `mlu` and `rmul!`, or split the rows over several GPUs.

## When to Use Something Else

- **One or a few sources on a very large graph.** A single Dijkstra or Δ-stepping run touches each edge once; it needs
  no factorization. PATHPACK.jl pays off when you need many sources, all pairs, or repeated queries.
- **Unweighted grids.** Breadth-first search is already fast: on a warehouse floor grid we measured PATHPACK.jl only
  1.15× faster than graph-tool and NetworKit on the same cores.
- **Negative cycles** (in min-plus): the shortest paths are not defined.
- **`PlusProd` with negative entries**: outside that semiring; use a linear solver.

## Getting Help

Open an [issue](https://github.com/AlgebraicJulia/PATHPACK.jl/issues) on GitHub.
