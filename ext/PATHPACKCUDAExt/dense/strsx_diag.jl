# ===== diagonal-block triangular solves in shared memory =====
#
# strsx_diag_kernel! gives each row of X one thread that walks the b ≤ 64 columns of the block in
# order, reading X and A from global memory: a chain of ~b²/2 dependent multiply-adds per thread, with
# only b threads (64 for the inversion of a diagonal block), ~40 µs per call. Here the block lives in
# shared memory (padded against bank conflicts), each row gets DIAG_KS threads that split the inner
# sum, and the partial sums are combined with warp shuffles: one barrier per column, ~b steps of a few
# cycles. The products are the same; only the order of the ⊕ reduction differs, so the result is
# bit-identical for idempotent ⊕ (min, max) and equal up to rounding for (+, ×).
#
const DIAG_KS = 16                       # threads per row (a power of 2 dividing 32)
const DIAG_NB = 64

# (row, column) of entry e (from 0) of a column-major matrix with d > 0 rows, by unsigned 32-bit division,
# which the GPU does inline: Int64 division calls a software routine, and checked division adds
# exception branches (both were CALLs in the diagonal-block kernels' copy loops)
@inline function cm_index(e::Integer, d::Integer)
    a = e % UInt32; b = d % UInt32
    return Int(Core.Intrinsics.urem_int(a, b)) + 1, Int(Core.Intrinsics.udiv_int(a, b)) + 1
end

@inline function ks_reduce(s::AbstractSemiring, x::T) where {T}
    o = DIAG_KS ÷ 2
    while o >= 1
        x = splus(s, x, shfl_xor_sync(0xffffffff, x, o), Val(:N))
        o ÷= 2
    end
    return x
end

# X ← X A* (UPLO = :U, upper A, columns left to right) or X ← X A (strictly lower part of A, unit
# diagonal, columns right to left; as strsx_diag_kernel!), for size(X, 1) ≤ 64 and size(A, 1) ≤ 64
function strsx_diag_shared_kernel!(s::AbstractSemiring, trans::Val, ::Val{SCALE}, ::Val{UPLO}, X::AbstractMatrix{T}, A::AbstractMatrix) where {SCALE, UPLO, T}
    Xs = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    As = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    m = size(X, 1); n = size(A, 1)
    tid = threadIdx().x - 1; nt = blockDim().x

    @inbounds begin
        e = tid
        while e < n * n
            ci, cj = cm_index(e, n)
            As[ci, cj] = A[ci, cj]
            e += nt
        end
        e = tid
        while e < m * n
            ci, cj = cm_index(e, m)
            Xs[ci, cj] = X[ci, cj]
            e += nt
        end
        sync_threads()

        q = tid % DIAG_KS
        r = tid ÷ DIAG_KS + 1                  # row of X
        z = szero(s, T, trans)

        for jj in 1:n
            j = UPLO === :U ? jj : n - jj + 1
            part = z

            if r <= m
                if UPLO === :U
                    k = 1 + q
                    while k < j
                        part = smuladd(s, Xs[r, k], As[k, j], part, Val(:N), trans)
                        k += DIAG_KS
                    end
                else
                    k = j + 1 + q
                    while k <= n
                        part = smuladd(s, Xs[r, k], As[k, j], part, Val(:N), trans)
                        k += DIAG_KS
                    end
                end
            end

            part = ks_reduce(s, part)

            if r <= m && q == 0
                acc = splus(s, Xs[r, j], part, Val(:N))
                SCALE && UPLO === :U && (acc = sprod(s, acc, sstar(s, As[j, j]), Val(:N), trans))
                Xs[r, j] = acc
            end

            sync_threads()
        end

        e = tid
        while e < m * n
            ci, cj = cm_index(e, m)
            X[ci, cj] = Xs[ci, cj]
            e += nt
        end
    end

    return
end

# X ← A* X for a lower unit triangular A (as strsx_left_diag_kernel!), for size(X, 2) ≤ 64, size(A, 1) ≤ 64
function strsx_left_diag_shared_kernel!(s::AbstractSemiring, trans::Val, X::AbstractMatrix{T}, A::AbstractMatrix) where {T}
    Xs = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    As = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    n = size(A, 1); m = size(X, 2)
    tid = threadIdx().x - 1; nt = blockDim().x

    @inbounds begin
        e = tid
        while e < n * n
            ci, cj = cm_index(e, n)
            As[ci, cj] = A[ci, cj]
            e += nt
        end
        e = tid
        while e < n * m
            ci, cj = cm_index(e, n)
            Xs[ci, cj] = X[ci, cj]
            e += nt
        end
        sync_threads()

        q = tid % DIAG_KS
        c = tid ÷ DIAG_KS + 1                  # column of X
        z = szero(s, T, Val(:N))

        for i in 1:n
            part = z

            if c <= m
                k = 1 + q
                while k < i
                    part = smuladd(s, As[i, k], Xs[k, c], part, Val(:N), trans)
                    k += DIAG_KS
                end
            end

            part = ks_reduce(s, part)
            (c <= m && q == 0) && (Xs[i, c] = splus(s, Xs[i, c], part, Val(:N)))
            sync_threads()
        end

        e = tid
        while e < n * m
            ci, cj = cm_index(e, n)
            X[ci, cj] = Xs[ci, cj]
            e += nt
        end
    end

    return
end

diag_threads(rows::Integer) = DIAG_KS * 32 * cld(max(rows, 1), 32)    # whole warps of whole row groups

# the diagonal solve of strsx_gpu! / strsx_left_gpu! on one ≤ 64 × 64 block
function diag_solve!(s::AbstractSemiring, trans::Val, scale::Val, uplo::Val, X::AbstractMatrix{T}, A::AbstractMatrix) where {T}
    if size(X, 1) <= DIAG_NB && size(A, 1) <= DIAG_NB && sizeof(T) <= 4      # 2 padded 64² blocks: 33 KB in 32 bits
        @cuda threads = diag_threads(size(X, 1)) strsx_diag_shared_kernel!(s, trans, scale, uplo, X, A)
    else
        tb = min(128, 32 * cld(size(X, 1), 32))
        @cuda threads = tb blocks = cld(size(X, 1), tb) strsx_diag_kernel!(s, trans, scale, uplo, X, A)
    end
    return X
end

function diag_solve_left!(s::AbstractSemiring, trans::Val, X::AbstractMatrix{T}, A::AbstractMatrix) where {T}
    if size(X, 2) <= DIAG_NB && size(A, 1) <= DIAG_NB && sizeof(T) <= 4
        @cuda threads = diag_threads(size(X, 2)) strsx_left_diag_shared_kernel!(s, trans, X, A)
    else
        tb = min(128, 32 * cld(size(X, 2), 32))
        @cuda threads = tb blocks = cld(size(X, 2), tb) strsx_left_diag_kernel!(s, trans, X, A)
    end
    return X
end

# C[:, Stgt[Sp + r - 1]] ← C[:, Stgt[Sp + r - 1]] ⊕ M[:, r]
function scatteradd_gpu!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, M::AbstractMatrix, Stgt::CuVector, Sp)
    function kernel(s, trans, C, M, Stgt, Sp)
        t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
        r = blockIdx().x

        if t <= size(M, 1)
            @inbounds begin
                j = Stgt[Sp + r - 1]
                C[t, j] = splus(s, C[t, j], M[t, r], trans)
            end
        end

        return
    end

    tb = min(256, 32 * cld(size(M, 1), 32))
    @cuda threads = tb blocks = (size(M, 2), cld(size(M, 1), tb)) kernel(s, trans, C, M, Stgt, Sp)
    return C
end

# M[:, r] ← C[:, Stgt[Sp + r - 1]]
function gather_gpu!(M::AbstractMatrix, C::AbstractMatrix, Stgt::CuVector, Sp)
    function kernel(M, C, Stgt, Sp)
        t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
        r = blockIdx().x

        if t <= size(M, 1)
            @inbounds M[t, r] = C[t, Stgt[Sp + r - 1]]
        end

        return
    end

    tb = min(256, 32 * cld(size(M, 1), 32))
    @cuda threads = tb blocks = (size(M, 2), cld(size(M, 1), tb)) kernel(M, C, Stgt, Sp)
    return M
end

# dst[:, perm[j]] = src[:, j], as permutecols! on the CPU
function permutecols_gpu!(dst::CuMatrix, src::CuMatrix, perm::CuVector)
    function kernel(dst, src, perm)
        t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
        j = blockIdx().x

        if t <= size(src, 1)
            @inbounds dst[t, perm[j]] = src[t, j]
        end

        return
    end

    tb = min(256, 32 * cld(size(src, 1), 32))
    @cuda threads = tb blocks = (size(src, 2), cld(size(src, 1), tb)) kernel(dst, src, perm)
    return dst
end

