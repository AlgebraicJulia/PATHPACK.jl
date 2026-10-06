module PATHPACKCUDAExt

# GPU kernels for PATHPACK.CPU.
#
# sgemx_gpu!(s, C, A, B) computes C ← C ⊕ A ⊗ B in the semiring s, with
# the same semantics as the CPU kernel CPU.sgemx!. The kernel is
# generic: it only calls smuladd, szero, and sone, so every semiring
# whose operations compile for the GPU works unchanged.
#
# Layout (see GPU_MERGE_PLAN.md):
#
#   runtime/   device profile, settings, graph capture, precompilation
#   dense/     semiring GEMM (sgemx), dense LU (sgetrf), triangular solves (strsx)
#   batched/   kernels that work on every front of a tree level in one launch
#   chordal/   the sparse factor on the GPU: hybrid factorization, solves, closure
#   multigpu.jl, api.jl

using CUDA
using SparseArrays: SparseMatrixCSC
import SparseArrays
import FileWatching
using CliqueTrees

using PATHPACK: CPU, GPU
using PATHPACK.CPU: AbstractSemiring, smuladd, splus, szero, sone

# the functions of the GPU-only API are declared in PATHPACK.GPU; this module adds their methods
import PATHPACK.GPU: sgemx_gpu!, rmul_gpu!, sssp_gpu!, sgetrf_gpu!, mlu_gpu, factorize!, closure_gpu!, closure_gpu,
    precompute_ops!, device_profile, measure!, closure_multigpu!, config, with_config, with_phases, apsp_gpu, apsp_gpu!, apsp_plan

export sgemx_gpu!, GPUSLU, rmul_gpu!, sssp_gpu!, SSSPPlan, sgetrf_gpu!, mlu_gpu, FactorPlan, factorize!, closure_gpu!, closure_gpu, precompute_ops!
export DeviceProfile, device_profile, measure!
export MultiGPUSLU, closure_multigpu!
export GPUConfig, config, with_config, with_phases
export apsp_gpu, apsp_gpu!, apsp_plan, APSPPlan

# ===== device overrides =====
#
# CPU.vmin(x, y) is ifelse(x < y, x, y) on x86 hosts (a host-side
# @static choice that the GPU compilation inherits), which ptxas compiles
# to FSETP + FSEL. On the GPU, llvm.minnum gives the same result whenever
# the accumulator is not NaN (in particular +∞ + -∞ = NaN is absorbed, as
# upstream intends) and is a single FMNMX instruction. This halves the cost
# of a min-plus multiply-add.
#
CUDA.@device_override @inline CPU.vmin(x::Float32, y::Float32) = ccall("llvm.minnum.f32", llvmcall, Float32, (Float32, Float32), x, y)
CUDA.@device_override @inline CPU.vmax(x::Float32, y::Float32) = ccall("llvm.maxnum.f32", llvmcall, Float32, (Float32, Float32), x, y)
CUDA.@device_override @inline CPU.vmin(x::Float64, y::Float64) = ccall("llvm.minnum.f64", llvmcall, Float64, (Float64, Float64), x, y)
CUDA.@device_override @inline CPU.vmax(x::Float64, y::Float64) = ccall("llvm.maxnum.f64", llvmcall, Float64, (Float64, Float64), x, y)

include("PATHPACKCUDAExt/runtime/device.jl")
include("PATHPACKCUDAExt/runtime/config.jl")

include("PATHPACKCUDAExt/dense/sgemx.jl")
include("PATHPACKCUDAExt/dense/sgemx_simt.jl")
include("PATHPACKCUDAExt/chordal/amalgamate.jl")
include("PATHPACKCUDAExt/chordal/sgetrs.jl")
include("PATHPACKCUDAExt/dense/strsx.jl")
include("PATHPACKCUDAExt/runtime/capture.jl")
include("PATHPACKCUDAExt/dense/strsx_diag.jl")
include("PATHPACKCUDAExt/batched/sweep.jl")
include("PATHPACKCUDAExt/chordal/toprows.jl")
include("PATHPACKCUDAExt/dense/sgemx_rows.jl")
include("PATHPACKCUDAExt/chordal/layered.jl")
include("PATHPACKCUDAExt/dense/sgetrf.jl")
include("PATHPACKCUDAExt/chordal/sgetrf.jl")
include("PATHPACKCUDAExt/batched/sgetrf.jl")
include("PATHPACKCUDAExt/chordal/sgetrf_top.jl")
include("PATHPACKCUDAExt/multigpu.jl")
include("PATHPACKCUDAExt/api.jl")

# the CPU's dense names on device matrices
include("PATHPACKCUDAExt/dense/dense.jl")

# Runtime state never outlives a session: a precompiled image must not carry device buffers, device
# profiles or tuning tables from precompilation, and the settings come from this session's environment.
function reset_caches!()
    lock(() -> (empty!(GEMM_TABLE); GEMM_TABLE_LOADED[] = false), GEMM_TABLE_LOCK)
    lock(() -> empty!(PROFILES), PROFILES_LOCK)
    lock(() -> empty!(AMALGAMATIONS), AMALGAMATIONS_LOCK)
    lock(() -> empty!(TRSM_WS), TRSM_WS)
    lock(() -> empty!(GRAPH_SCRATCH), GRAPH_SCRATCH)
    # (dropped, not freed: in a session loaded from a precompiled image the pointer is not ours)
    settle_arena()
    lock(() -> (ARENA.mem = nothing; ARENA.size = 0; ARENA.busy = false; ARENA.off = false), ARENA_LOCK)
    return
end

function __init__()
    init_config!()
    reset_caches!()
end

include("PATHPACKCUDAExt/runtime/precompile.jl")

end
