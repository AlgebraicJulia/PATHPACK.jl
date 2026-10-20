# Library Reference

Everything you call is in two modules: `PATHPACK.CPU` (the solver and the semirings) and `PATHPACK.GPU` (the GPU solver,
available once `CUDA` is loaded). Import what you use:

```julia
using PATHPACK
using PATHPACK.CPU: MinPlus, MaxMin, mstar, mlu
using PATHPACK.GPU: apsp_gpu       # after `using CUDA`
```

| You want | Call |
|---|---|
| every pair, on the CPU | [`mstar(s, A)`](@ref PATHPACK.CPU.mstar) |
| some rows or columns, or many queries | [`mlu(s, A)`](@ref PATHPACK.CPU.mlu), then `rmul!(B, F)` (rows) or `lmul!(F, B)` (columns) |
| every pair, on one GPU or several | [`apsp_gpu(A; semiring, devices)`](@ref PATHPACK.GPU.apsp_gpu) |
| a few sources, on the GPU | `apsp_gpu(A, sources)` |
| new weights on the same graph, on the GPU | [`apsp_plan(A)`](@ref PATHPACK.GPU.apsp_plan), then `apsp_gpu(plan, A2)` |

## Computing Stars

```@docs
PATHPACK.CPU.mstar
PATHPACK.CPU.mlu
PATHPACK.CPU.pstar
```

`lmul!(F, B)` overwrites `B` with `A* B`, the least solution of `X = A X ⊕ B`; `rmul!(B, F)` overwrites `B` with
`B A*`. Both take a matrix of right-hand sides at once and the keyword `nt`, and `Matrix(F)` is the whole star.

## Semirings

```@docs
PATHPACK.CPU.MinPlus
PATHPACK.CPU.MaxPlus
PATHPACK.CPU.MaxMin
PATHPACK.CPU.MinMax
PATHPACK.CPU.MaxProd
PATHPACK.CPU.MinProd
PATHPACK.CPU.PlusProd
PATHPACK.CPU.OrAnd
PATHPACK.CPU.AndOr
```

## Semiring Algebra

The scalar layer under the solver, for defining a semiring of your own or calling the kernels directly. A semiring is
a subtype of `AbstractSemiring`; the solver reaches it only through these functions.

```@docs
PATHPACK.CPU.AbstractSemiring
PATHPACK.CPU.szero
PATHPACK.CPU.sone
PATHPACK.CPU.splus
PATHPACK.CPU.sprod
PATHPACK.CPU.smuladd
PATHPACK.CPU.sstar
PATHPACK.CPU.slte
PATHPACK.CPU.sgte
PATHPACK.CPU.sdot
PATHPACK.CPU.saxpy!
PATHPACK.CPU.sger!
```

Semirings built from others, and families:

```@docs
PATHPACK.CPU.DualQuantale
PATHPACK.CPU.NegativeQuantale
PATHPACK.CPU.Lattice
PATHPACK.CPU.MinPlusLaw
PATHPACK.CPU.MaxPlusLaw
PATHPACK.CPU.MinProdLaw
PATHPACK.CPU.MaxProdLaw
PATHPACK.CPU.LawvereQuantale
PATHPACK.CPU.TropicalSemiring
```

## GPUs

```@docs
PATHPACK.GPU.apsp_gpu
PATHPACK.GPU.apsp_gpu!
PATHPACK.GPU.apsp_plan
PATHPACK.GPU.sssp_gpu!
```

## GPU Tuning and Internals

You do not need these to use the solver. They expose its configuration, device profiles and phase timings.

```@docs
PATHPACK.GPU.config
PATHPACK.GPU.with_config
PATHPACK.GPU.device_profile
PATHPACK.GPU.with_phases
PATHPACK.GPU.measure!
PATHPACK.GPU.extension
PATHPACK.GPU.mlu_gpu
PATHPACK.GPU.rmul_gpu!
PATHPACK.GPU.closure_gpu
PATHPACK.GPU.closure_gpu!
PATHPACK.GPU.closure_multigpu!
PATHPACK.GPU.factorize!
PATHPACK.GPU.precompute_ops!
PATHPACK.GPU.sgemx_gpu!
PATHPACK.GPU.sgetrf_gpu!
```
