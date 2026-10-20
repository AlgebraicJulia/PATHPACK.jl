# Release Notes

## Unreleased

The first public version of PATHPACK.jl. Install it from GitHub for now:

```julia
using Pkg
Pkg.add(url = "https://github.com/AlgebraicJulia/PATHPACK.jl")
```

### Solver

- `mstar`, `mlu` and `pstar`: the star of a sparse matrix over a semiring, its factorization, and some of its rows,
  by a sparse direct method with a fill-reducing order from CliqueTrees.jl (`AMF()` by default).
- `lmul!` and `rmul!` on a factorization: single-source and single-target queries without the whole star.
- Semirings `MinPlus`, `MaxPlus`, `MaxMin`, `MinMax`, `MinProd`, `MaxProd`, `PlusProd`, `OrAnd` and `AndOr`, and the
  scalar interface for defining others.
- Multithreaded on the CPU through the keyword `nt`.

### GPUs

- With CUDA.jl loaded: `apsp_gpu` and `apsp_gpu!` for all pairs on one NVIDIA GPU or several, `apsp_plan` to reuse
  the symbolic work, and `sssp_gpu!` for single sources.

### Documentation

- This site: a guide, 13 tutorials with downloadable notebooks, the benchmarks, and the library reference.
