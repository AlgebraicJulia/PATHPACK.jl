# GPU factorization of the closure, A* = U* L*.
#
#   sgetrf_gpu!(s, A)        dense LU of a CuMatrix, in place (as CPU.sgetrf!)
#   mlu_gpu(s, A)            hybrid sparse factorization: the many small fronts
#                            at the bottom of the elimination tree on the CPU,
#                            the large fronts near the root (and all their
#                            ancestors) on the GPU. Returns (F, G): the CPU
#                            factor and a GPUSLU whose factor lives on the GPU.
#
# Every kernel performs the same semiring operations in the same per-entry
# order as the CPU code, so idempotent semirings (tropical, bottleneck)
# reproduce the CPU factor bit for bit.

using PATHPACK.CPU: AbstractSemiring, ChordalSLU, splus, sprod, sstar, szero, isintegral, spool_mt,
    sgetrf_loop!, sgetrf_loop_1!

# ===== dense LU =====

const SGETRF_GPU_NB = 64
const LU_DIAG_THREADS = 1024             # 256 left most of the b³/3 updates of a 64-block serial per thread


#
# A ← L + U with A* = U* L*: right-looking and blocked, as CPU.sgetrf_mt!
#
#   Akk ← LU of Akk                 one thread block, Akk in shared memory
#   Akn ← Lkk* Akn                  left lower unit TRSM
#   Ank ← Ank Ukk*                  right upper TRSM
#   Ann ← Ann ⊕ Ank Akn             semiring GEMM
#
function sgetrf_gpu!(s::AbstractSemiring, A::AbstractMatrix{T}; nb::Int = SGETRF_GPU_NB) where {T}
    @assert size(A, 1) == size(A, 2)
    @assert nb <= SGETRF_GPU_NB

    n = size(A, 1)
    scale = Val(!isintegral(s))

    for k in 1:nb:n
        b = min(nb, n - k + 1)
        J = k:(k + b - 1)
        Akk = view(A, J, J)
        @phase FTIMER[] :lu_diag @cuda threads = LU_DIAG_THREADS sgetrf_diag_kernel!(s, scale, Akk)

        if k + b <= n
            R = (k + b):n
            @phase FTIMER[] :lu_trsm strsx_left_gpu!(s, Val(:N), Akk, view(A, J, R))
            @phase FTIMER[] :lu_trsm strsx_gpu!(s, Val(:N), scale, Val(:U), view(A, R, J), Akk)
            @phase FTIMER[] :lu_gemm sgemx_gpu!(s, view(A, R, R), view(A, R, J), view(A, J, R))
        end
    end

    return A
end

#
# Unblocked LU of a b × b block (b ≤ 64), as CPU.sgetrf2!:
#
#   for p = 1, …, b:
#       A[k, p] ← A[k, p] A[p, p]*              k > p
#       A[k, j] ← A[k, j] ⊕ A[k, p] A[p, j]     k, j > p
#
function sgetrf_diag_kernel!(s::AbstractSemiring, ::Val{SCALE}, A::AbstractMatrix{T}) where {SCALE, T}
    S = CuStaticSharedArray(T, (SGETRF_GPU_NB, SGETRF_GPU_NB))
    b = size(A, 1)
    t = threadIdx().x
    nt = blockDim().x

    @inbounds begin
        e = t

        while e <= b * b
            i = (e - 1) % b + 1
            j = (e - 1) ÷ b + 1
            S[i, j] = A[i, j]
            e += nt
        end

        sync_threads()

        for p in 1:b
            if SCALE
                sp = sstar(s, S[p, p])
                k = p + t

                while k <= b
                    S[k, p] = sprod(s, S[k, p], sp, Val(:N), Val(:N))
                    k += nt
                end

                sync_threads()
            end

            m = b - p
            e = t

            while e <= m * m
                k = p + (e - 1) % m + 1
                j = p + (e - 1) ÷ m + 1
                S[k, j] = smuladd(s, S[k, p], S[p, j], S[k, j], Val(:N), Val(:N))
                e += nt
            end

            sync_threads()
        end

        e = t

        while e <= b * b
            i = (e - 1) % b + 1
            j = (e - 1) ÷ b + 1
            A[i, j] = S[i, j]
            e += nt
        end
    end

    return
end

#
# Left lower unit triangular solve X ← A* X, blocked over rows:
#
#   X[i, :] ← X[i, :] ⊕ Σ_{k<i} A[i, k] X[k, :]
#
# with diagonal-block inversion as in strsx_gpu!: T = A[J, J]* (solve with
# the identity), X[J, :] ← T X[J, :] as a GEMM, then the trailing GEMM.
#
function strsx_left_gpu!(s::AbstractSemiring, trans::Val, A::AbstractMatrix, X::AbstractMatrix{T}; nb::Int = STRSX_GPU_NB) where {T}
    n = size(A, 1)
    m = size(X, 2)
    inv = m > STRSX_INV_MIN

    for i0 in 1:nb:n
        i1 = min(i0 + nb - 1, n)
        J = i0:i1
        b = i1 - i0 + 1

        if inv
            Tb, W = trsm_workspace(T, b * m)
            Tj = view(Tb, 1:b, 1:b)
            identity_gpu!(s, Tj)
            diag_solve_left!(s, trans, Tj, view(A, J, J))
            Wj = reshape(view(W, 1:(b * m)), b, m)
            fill!(Wj, szero(s, T, Val(:N)))
            sgemx_gpu!(s, Wj, Tj, view(X, J, :))
            copy_gpu!(view(X, J, :), Wj)
        else
            diag_solve_left!(s, trans, view(X, J, :), view(A, J, J))
        end

        if i1 < n
            sgemx_gpu!(s, view(X, (i1 + 1):n, :), view(A, (i1 + 1):n, J), view(X, J, :))
        end
    end

    return X
end

function strsx_left_diag_kernel!(s::AbstractSemiring, trans::Val, X::AbstractMatrix, A::AbstractMatrix)
    c = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    n = size(A, 1)

    if c <= size(X, 2)
        @inbounds for i in 1:n
            acc = X[i, c]

            for k in 1:(i - 1)
                acc = smuladd(s, A[i, k], X[k, c], acc, Val(:N), trans)
            end

            X[i, c] = acc
        end
    end

    return
end

