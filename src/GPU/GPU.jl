module GPU

#
# The GPU backend. Its code is the package extension PATHPACKCUDAExt (ext/), which loads with CUDA.jl:
#
#   using PATHPACK, CUDA
#   D = PATHPACK.GPU.apsp_gpu(A)                     the closure A* on the GPU
#
# This module declares the functions of the GPU-only API (plans, configuration, device profile,
# multi-GPU), and the extension adds their methods. The extension's types (GPUSLU, FactorPlan,
# APSPPlan, GPUConfig, ...) are reached through extension().
#
# PATHPACK.CPU's dense functions also take device matrices (CuMatrix), with the same arguments, where
# the GPU has the case: sgemx! (op N or T), sgetrf!, strsx! (side :R; side :L lower unit), strtri!,
# sgetrs! (side :R), sgetri!. Other cases throw an ArgumentError.
#

export apsp_gpu, apsp_gpu!, apsp_plan

"""
    extension()

The module of the GPU backend (`PATHPACKCUDAExt`), or `nothing` before CUDA.jl is loaded.
"""
function extension()
    return Base.get_extension(parentmodule(@__MODULE__), :PATHPACKCUDAExt)
end

"""
    apsp_gpu(A; semiring = MinPlus(), devices = [CUDA.device()], output = :device)
    apsp_gpu(A, sources; semiring = MinPlus(), output = :device)

The closure A* of the square sparse matrix `A` over `semiring`, on the GPU, in the vertex labels of
`A`. Needs CUDA.jl (`using CUDA`).
"""
function apsp_gpu end

"""
    apsp_gpu!(H, A; semiring = MinPlus(), alg)
    apsp_gpu!(H, plan, A)
    apsp_gpu!(D, A; semiring = MinPlus(), alg)

The closure of `A` into the host matrix `H`, or into the device matrix `D` (a `CuMatrix` the caller
allocated). Needs CUDA.jl.
"""
function apsp_gpu! end

"""
    apsp_plan(A; semiring = MinPlus())

A plan for repeated closures of matrices with the pattern of `A` (new weights, same graph). Needs
CUDA.jl.
"""
function apsp_plan end

"The closure of a factor on the GPU, as a new device matrix. Needs CUDA.jl."
function closure_gpu end

"The closure of a factor on the GPU, into a device matrix. Needs CUDA.jl."
function closure_gpu! end

"The closure on several GPUs, one block of rows per device. Needs CUDA.jl."
function closure_multigpu! end

"Single-source solves on the GPU, one source per row of the output. Needs CUDA.jl."
function sssp_gpu! end

"`B ← B A*` on the GPU, `B` with one right-hand side per row. Needs CUDA.jl."
function rmul_gpu! end

"`C ← C ⊕ A ⊗ B` on the GPU. Needs CUDA.jl."
function sgemx_gpu! end

"LU factorization of the closure on the GPU (dense, or hybrid sparse). Needs CUDA.jl."
function sgetrf_gpu! end

"The hybrid CPU + GPU factorization of a sparse matrix. Needs CUDA.jl."
function mlu_gpu end

"The numeric factorization of a factor plan (new weights, same structure). Needs CUDA.jl."
function factorize! end

"The dense operators of the large fronts of a GPU factor. Needs CUDA.jl."
function precompute_ops! end

"The measured profile of a GPU. Needs CUDA.jl."
function device_profile end

"Measure the profile of a GPU. Needs CUDA.jl."
function measure! end

"The GPU solver's settings in this scope. Needs CUDA.jl."
function config end

"Run `f` with some settings of the GPU solver changed. Needs CUDA.jl."
function with_config end

"""
    r, t = with_phases(f)

`f()` and the time of each phase of the GPU solver's calls in it (symbolic, numeric CPU / GPU part,
transfer, inverse U*, solve L*, relabel, ...). Needs CUDA.jl.
"""
function with_phases end

const API = (apsp_gpu, apsp_gpu!, apsp_plan, closure_gpu, closure_gpu!, closure_multigpu!, sssp_gpu!, rmul_gpu!,
    sgemx_gpu!, sgetrf_gpu!, mlu_gpu, factorize!, precompute_ops!, device_profile, measure!, config, with_config, with_phases)

function __init__()
    # a call before CUDA.jl is loaded has no methods: say why
    Base.Experimental.register_error_hint(MethodError) do io, e, argtypes, kwargs
        if any(f -> e.f === f, API) && isnothing(extension())
            print(io, "\nThe GPU backend of PATHPACK loads with CUDA.jl: run `using CUDA` first.")
        end
    end
end

end
