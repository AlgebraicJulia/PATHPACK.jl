# ===== strsx_gpu! =====

const STRSX_GPU_NB = 64

#
# Dense right triangular solve X ← X A*, blocked over 64-column diagonal
# blocks:
#
#   uplo = :U   forward,  X[:, j] ← (X[:, j] ⊕ Σ_{k<j} X[:, k] A[k, j]) A[j, j]*
#   uplo = :L   backward, X[:, j] ←  X[:, j] ⊕ Σ_{k>j} X[:, k] A[k, j]      (unit)
#
# A diagonal block is applied by inversion (diagonal-block inversion, as in
# GPU supernodal solvers): its closure T = A[J, J]* is formed by solving
# with the identity (one small kernel, b threads), and X[:, J] ← X[:, J] T is
# a semiring GEMM. The trailing update is a GEMM as well. Over exact
# arithmetic (tropical with integer values, bottleneck, Boolean) this is
# identical to substitution; with floating-point values it re-associates the
# products, so results may differ in the last bits.
# When X has few rows, substitution with one thread per row is used instead.
#
function strsx_gpu!(s::AbstractSemiring, trans::Val, scale::Val, uplo::Val{UPLO}, X::AbstractMatrix{T}, A::AbstractMatrix; nb::Int = STRSX_GPU_NB) where {UPLO, T}
    n = size(A, 1)
    m = size(X, 1)
    inv = m > STRSX_INV_MIN

    blocks = UPLO === :U ? [(j0, min(j0 + nb - 1, n)) for j0 in 1:nb:n] : [(max(j1 - nb + 1, 1), j1) for j1 in n:-nb:1]

    for (j0, j1) in blocks
        J = j0:j1
        b = j1 - j0 + 1

        if inv
            Tb, W = trsm_workspace(T, m * b)
            Tj = view(Tb, 1:b, 1:b)
            identity_gpu!(s, Tj)
            diag_solve!(s, trans, scale, uplo, Tj, view(A, J, J))
            Wj = reshape(view(W, 1:(m * b)), m, b)
            fill!(Wj, szero(s, T, Val(:N)))
            sgemx_gpu!(s, Wj, view(X, :, J), Tj; inplace = false)
            copy_gpu!(view(X, :, J), Wj)
        else
            diag_solve!(s, trans, scale, uplo, view(X, :, J), view(A, J, J))
        end

        if UPLO === :U && j1 < n
            # (disjoint columns of X: as mightalias finds for strided views, which it cannot check for index views)
            sgemx_gpu!(s, view(X, :, (j1 + 1):n), view(X, :, J), view(A, J, (j1 + 1):n); inplace = false)
        elseif UPLO === :L && j0 > 1
            sgemx_gpu!(s, view(X, :, 1:(j0 - 1)), view(X, :, J), view(A, J, 1:(j0 - 1)); inplace = false)
        end
    end

    return X
end

const STRSX_INV_MIN = 64

# Scratch for diagonal-block inversion, per stream (so that concurrent
# streams never share it) and element type. It only grows outside stream
# capture (plans warm up before capturing).
#
# Each table belongs to the owner of the streams that use it: the plan that
# runs work on its own streams (SCRATCH_OWNER, set by FactorPlan and
# precompute_ops!), else the current task, whose task-local stream it is.
# Owners are held weakly, so the scratch is freed with its plan or task. They
# must hash by identity (a Task, a plan, a Ref token), not by contents.
# (A table keyed weakly by the stream itself would never free anything:
# a CUDA.jl array keeps a reference to the last stream that used it. And
# a table keyed by stream handle, as before, never freed anything and could
# hand a reused handle a stale buffer, possibly from another device.)
const TRSM_WS = WeakKeyDict{Any, Dict{Tuple{CuStream, DataType}, Tuple{CuMatrix, CuVector}}}()

const SCRATCH_OWNER = ScopedValue{Any}(nothing)

function trsm_workspace(::Type{T}, len::Int) where {T}
    owner = something(SCRATCH_OWNER[], current_task())
    key = (CUDA.stream(), T)
    Tb, W = lock(TRSM_WS) do
        get(get!(Dict{Tuple{CuStream, DataType}, Tuple{CuMatrix, CuVector}}, TRSM_WS, owner), key, (nothing, nothing))
    end

    if isnothing(Tb) || length(W) < len
        @assert !CUDA.is_capturing() "trsm workspace must grow before graph capture"
        Tb = CuMatrix{T}(undef, STRSX_GPU_NB, STRSX_GPU_NB)
        W = CuVector{T}(undef, max(len, isnothing(W) ? 0 : 2 * length(W)))
        CUDA.enable_synchronization!(Tb, false)
        CUDA.enable_synchronization!(W, false)
        lock(() -> (TRSM_WS[owner][key] = (Tb, W)), TRSM_WS)
    end

    keep = CAPTURED[]
    isnothing(keep) || push!(keep, Tb, W)
    return Tb::CuMatrix{T}, W::CuVector{T}
end

# X ← semiring identity
function identity_gpu!(s::AbstractSemiring, X::AbstractMatrix{T}) where {T}
    function kernel(s, X)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(X, 1)
            @inbounds while j <= size(X, 2)
                X[i, j] = i == j ? sone(s, T, Val(:N)) : szero(s, T, Val(:N))
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(X, 1), size(X, 2), s, X)
    return X
end

# X ← Y (both may be strided views)
function copy_gpu!(X::AbstractMatrix, Y::AbstractMatrix)
    function kernel(X, Y)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(X, 1)
            @inbounds while j <= size(X, 2)
                X[i, j] = Y[i, j]
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(X, 1), size(X, 2), X, Y)
    return X
end

#
# Launch kernel(args...) over an m × ncols index space: rows on the x axis,
# columns on the y axis (strided past the 65535 limit), so that kernels
# need no (emulated, 64-bit) integer division to recover (i, j).
#
function launch2d(kernel, m::Integer, ncols::Integer, args...)
    if m > 0 && ncols > 0
        tb = min(256, 32 * cld(m, 32))
        @cuda threads = tb blocks = (cld(m, tb), min(ncols, 65535)) kernel(args...)
    end

    return
end

function strsx_diag_kernel!(s::AbstractSemiring, trans::Val, ::Val{SCALE}, ::Val{UPLO}, X::AbstractMatrix, A::AbstractMatrix) where {SCALE, UPLO}
    t = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(A, 1)

    if t > size(X, 1)
        return
    end

    @inbounds if UPLO === :U
        for j in 1:n
            acc = X[t, j]

            for k in 1:(j - 1)
                acc = smuladd(s, X[t, k], A[k, j], acc, Val(:N), trans)
            end

            if SCALE
                acc = sprod(s, acc, sstar(s, A[j, j]), Val(:N), trans)
            end

            X[t, j] = acc
        end
    else
        for j in n:-1:1
            acc = X[t, j]

            for k in (j + 1):n
                acc = smuladd(s, X[t, k], A[k, j], acc, Val(:N), trans)
            end

            X[t, j] = acc
        end
    end

    return
end

