<h1 align="center">
    PATHPACK.jl
    <p align="center">
        <a href="https://algebraicjulia.github.io/PATHPACK.jl/dev">
            <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="documentation (development)">
        </a>
        <a href="https://github.com/AlgebraicJulia/PATHPACK.jl/actions/workflows/julia_ci.yml">
            <img src="https://github.com/AlgebraicJulia/PATHPACK.jl/actions/workflows/julia_ci.yml/badge.svg" alt="build status">
        </a>
        <a href="https://github.com/JuliaTesting/Aqua.jl">
            <img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg" alt="Aqua">
        </a>
    </p>
</h1>

PATHPACK.jl solves *algebraic path problems* in Julia: shortest paths, widest paths, reachability, and their relatives,
between all pairs of vertices of a graph or from a few of them. It computes the *star* A\* = I ⊕ A ⊕ A² ⊕ ⋯ of a sparse
matrix over a [semiring](https://en.wikipedia.org/wiki/Semiring) with a sparse LU factorization over that semiring. The
solver runs on multithreaded CPUs and, through a [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) extension, on NVIDIA
GPUs.

## Getting Help

If you have a question about the library, feel free to open an [issue](https://github.com/AlgebraicJulia/PATHPACK.jl/issues).

## Installation

To install PATHPACK.jl, enter the Pkg REPL by typing `]` and run the following command.

```
pkg> add https://github.com/AlgebraicJulia/PATHPACK.jl
```

The CPU solver is multithreaded: start Julia with several threads, e.g. `julia --threads auto`. For the GPU solver, also
add CUDA.jl and load it next to PATHPACK.

```
pkg> add CUDA
```

## Algebraic Path Problems

Let A be the adjacency matrix of a weighted directed graph: A[i, j] is the weight of the arc i → j. Path problems ask,
for every pair of vertices, for the best path between them. Which path is best, and what "best" means, depends on the
problem, but many problems share one structure: the weights live in a *semiring*, a set with two operations ⊕ and ⊗
that behave like addition and multiplication. The weight of a path is the ⊗-product of its arcs' weights, and the
answer for a pair of vertices is the ⊕-sum over all paths between them. This answer is the entry A\*[i, j] of the star

    A* = I ⊕ A ⊕ A² ⊕ A³ ⊕ ⋯

Different semirings give different problems.

| Problem | Semiring | ⊕, ⊗ | A\*[i, j] |
|---|---|---|---|
| shortest paths | `MinPlus` | min, + | the length of a shortest path from i to j |
| longest paths | `MaxPlus` | max, + | the length of a longest path from i to j |
| widest paths | `MaxMin` | max, min | the largest capacity of a path from i to j |
| minimax paths | `MinMax` | min, max | the smallest largest arc weight of a path from i to j |
| most reliable paths | `MaxProd` | max, × | the largest product of the arcs' weights along a path |
| path sums | `PlusProd` | +, × | the sum over all paths of the products of their weights |
| reachability | `OrAnd` | ∪, ∩ | the set of labels that some path carries from i to j |

PATHPACK.jl computes stars by Gaussian elimination over the semiring. It factorizes the star as A\* = U\*L\*, where U
and L are triangular: first it orders the vertices to reduce fill (with
[CliqueTrees.jl](https://github.com/AlgebraicJulia/CliqueTrees.jl)) and analyzes the factor's structure; then it
factorizes each strongly connected component with the multifrontal method, which turns the sparse factorization into
many small dense ones. From the factors it computes either the whole star or a few of its rows and columns.

## Computing Stars

Stars are computed by the function `mstar`. It takes a semiring and a matrix.

```julia-repl
julia> using PATHPACK, SparseArrays, LinearAlgebra

julia> using PATHPACK.CPU: MinPlus, MaxMin, mstar, mlu

julia> A = sparse([1, 1, 2, 3, 4, 4], [2, 3, 4, 2, 1, 5], [3.0, 8.0, 1.0, 4.0, 2.0, 7.0], 5, 5)
5×5 SparseMatrixCSC{Float64, Int64} with 6 stored entries:
  ⋅   3.0  8.0   ⋅    ⋅
  ⋅    ⋅    ⋅   1.0   ⋅
  ⋅   4.0   ⋅    ⋅    ⋅
 2.0   ⋅    ⋅    ⋅   7.0
  ⋅    ⋅    ⋅    ⋅    ⋅

julia> mstar(MinPlus(), A)
5×5 Matrix{Float64}:
  0.0   3.0   8.0   4.0  11.0
  3.0   0.0  11.0   1.0   8.0
  7.0   4.0   0.0   5.0  12.0
  2.0   5.0  10.0   0.0   7.0
 Inf   Inf   Inf   Inf    0.0
```

In the min-plus semiring, the entry A\*[i, j] is the distance from i to j, and `Inf` means that j cannot be reached from
i. With the semiring `MaxMin`, the same call computes the capacity of the widest path between every pair of vertices.

```julia-repl
julia> mstar(MaxMin(), A)
5×5 Matrix{Float64}:
  Inf     4.0    8.0    1.0   1.0
   1.0   Inf     1.0    1.0   1.0
   1.0    4.0   Inf     1.0   1.0
   2.0    2.0    2.0   Inf    7.0
 -Inf   -Inf   -Inf   -Inf   Inf
```

## Factorizations and Queries

The star of a graph with n vertices has n² entries. If you only need some of them, factorize the matrix once with the
function `mlu`, and then solve for the rows or columns you need.

```julia-repl
julia> F = mlu(MinPlus(), A);

julia> b = fill(Inf, 5); b[5] = 0.0;    # the unit vector at vertex 5

julia> lmul!(F, b)                      # column 5 of A*: the distances to vertex 5
5-element Vector{Float64}:
 11.0
  8.0
 12.0
  7.0
  0.0

julia> B = fill(Inf, 1, 5); B[1] = 0.0;    # the unit row vector at vertex 1

julia> rmul!(B, F)                         # row 1 of A*: the distances from vertex 1
1×5 Matrix{Float64}:
 0.0  3.0  8.0  4.0  11.0
```

The function `lmul!(F, B)` overwrites B with the least solution X of the fixpoint equation X = AX ⊕ B, which is A\*B; the
function `rmul!(B, F)` overwrites B with BA\*. Both take a matrix of right-hand sides at once, and `Matrix(F)` is the
whole star. All of these functions take the keyword argument `nt`, the number of threads (by default, all of Julia's).

## Orderings

The keyword argument `alg` of `mlu` and `mstar` chooses the fill-reducing ordering. It can be any elimination algorithm
of CliqueTrees.jl, for example `AMF()` (approximate minimum fill) or `METIS()` (nested dissection), or a permutation of
the vertices. To name one, add CliqueTrees.jl to your environment (`pkg> add CliqueTrees`).

```julia-repl
julia> using CliqueTrees: AMF

julia> F = mlu(MinPlus(), A; alg = AMF());
```

## GPUs

The GPU solver computes the whole star on one GPU or on several. It factorizes the many small fronts of the factorization
with CPU threads and the large ones on the GPU.

```julia
using PATHPACK, CUDA, SparseArrays, Random
using PATHPACK.CPU: MinPlus, mstar
using PATHPACK.GPU: apsp_gpu

Random.seed!(1)
n = 5000
I = rand(1:n, 2n); J = rand(1:n, 2n); keep = I .!= J
W = Float32.(rand(1:100, count(keep)))                                    # integer weights: exact in Float32
A = sparse([I[keep]; J[keep]], [J[keep]; I[keep]], [W; W], n, n, min)    # an undirected graph
D = apsp_gpu(A; output = :host)                                           # a Matrix{Float32}
D == mstar(MinPlus(), A)                                                  # true: the CPU star, bit for bit
```

| Call | Result |
|---|---|
| `apsp_gpu(A)` | the star as a `CuMatrix` |
| `apsp_gpu(A, sources)` | only the rows of these sources |
| `apsp_gpu(A; devices = collect(CUDA.devices()))` | the rows split over several GPUs |
| `apsp_gpu!(D, A)` | the star written into a matrix you allocated |
| `P = apsp_plan(A); apsp_gpu(P, A2)` | new weights on the same graph; what depends only on the graph is reused |
| `apsp_gpu(A; semiring = MaxMin())` | another semiring |

The GPU solver takes `Float32` and `Float64` matrices (plus-times: `Float64`; max-min: also `Int32`). It handles undirected graphs, and
directed graphs whose strongly connected components do not reach one another; for other directed graphs, use the CPU
solver.

## Semirings

All semirings are exported by `PATHPACK.CPU`.

| Semiring | Set | ⊕ | ⊗ |
|---|---|---|---|
| `MinPlus` | [-∞, ∞] | min | + |
| `MaxPlus` | [-∞, ∞] | max | + |
| `MaxMin` | [-∞, ∞] | max | min |
| `MinMax` | [-∞, ∞] | min | max |
| `PlusProd` | [0, ∞] | + | × |
| `MinProd`, `MaxProd` | [0, ∞] | min, max | × |
| `OrAnd`, `AndOr` | sets of bits, stored in integers | ∪, ∩ | ∩, ∪ |
| `MinPlusLaw`, `MaxPlusLaw` | [0, ∞], [-∞, 0] | min, max | + |
| `MinProdLaw`, `MaxProdLaw` | [1, ∞], [0, 1] | min, max | × |

The last four are the Lawvere-quantale variants of the tropical and product semirings.
