# PATHPACK.jl

```@meta
CurrentModule = PATHPACK
```

PATHPACK.jl solves *algebraic path problems*: shortest paths, widest paths, reachability, and their relatives, between
all pairs of vertices of a graph or from a few of them. It computes the *star* ``A^* = I \oplus A \oplus A^2 \oplus
\cdots`` of a sparse matrix over a semiring with a sparse LU factorization over that semiring, on multithreaded CPUs
(`PATHPACK.CPU`) and, through a CUDA.jl extension, on NVIDIA GPUs (`PATHPACK.GPU`).

```julia
using PATHPACK, SparseArrays
using PATHPACK.CPU: MinPlus, mstar

A = sparse([1, 2, 3], [2, 3, 1], [1.0, 2.0, 3.0], 3, 3)
mstar(MinPlus(), A)    # all-pairs shortest paths
```

See the [example](generated/shortest_paths.md) for queries from a factorization, and the
[README](https://github.com/AlgebraicJulia/PATHPACK.jl) for the semirings and the GPU solver.
