# ===== the CPU's dense names on device matrices =====
#
# PATHPACK.CPU's dense kernels, with the same arguments, for matrices on the GPU:
#
#   sgemx!(s, tA, tB, C, A, B)          C ← C ⊕ op(A) ⊗ op(B)                 sgemx_gpu!
#   sgetrf!(s, A)                       A ← L + U with A* = U* L*             sgetrf_gpu!
#   strsx!(s, side, trans, uplo, diag, A, B)                                  strsx_gpu!, strsx_left_gpu!
#   strtri!(s, uplo, diag, A)           A ← A* (the triangle of A)            strsx_gpu! on the identity
#   sgetrs!(s, side, trans, A, B)       B ← B A* (side :R)                    two strsx!
#   sgetri!(s, C, A)                    C ← A* = U* L*                        strtri! + strsx!
#
# The GPU has the cases the chordal code needs (side :R, and the left lower unit solve of the LU), in
# the semiring's own order (trans = Val(:N)); the others throw an ArgumentError. Triangular solves
# invert 64 × 64 diagonal blocks and use GEMMs (strsx_gpu!): over exact values (integer weights in the
# tropical semirings, or the bottleneck ones) the results are the CPU's bit for bit, with floating-point
# weights they agree up to rounding.
#

const DenseCuMatrix{T} = Union{CuMatrix{T}, SubArray{T, 2, <:CuArray{T}}}

unsupported(what) = throw(ArgumentError("$what is not supported on the GPU; use a host matrix (PATHPACK.CPU)"))

function CPU.sgemx!(s::AbstractSemiring, tA::Val{TA}, tB::Val{TB}, C::DenseCuMatrix{T}, A::DenseCuMatrix{T}, B::DenseCuMatrix{T};
        nt::Integer = 1) where {T, TA, TB}
    (TA in (:N, :T) && TB in (:N, :T)) || unsupported("sgemx! with op = $TA, $TB")
    # (a transposed operand is copied: the kernels read A and B column-major)
    sgemx_gpu!(s, C, TA === :N ? A : permutedims(A), TB === :N ? B : permutedims(B))
    return C
end

function CPU.sgetrf!(s::AbstractSemiring, A::DenseCuMatrix; nt::Integer = 1)
    return sgetrf_gpu!(s, A)
end

function CPU.strsx!(s::AbstractSemiring, side::Val{SIDE}, trans::Val{TRANS}, uplo::Val{UPLO}, diag::Val{DIAG}, A::DenseCuMatrix,
        B::DenseCuMatrix; nt::Integer = 1) where {SIDE, TRANS, UPLO, DIAG}
    TRANS === :N || unsupported("strsx! with trans = $TRANS")

    if SIDE === :R && UPLO === :U
        #
        #   B ← B U*   (diag :N; :U for a unit diagonal)
        #
        strsx_gpu!(s, trans, Val(DIAG === :N && !isintegral(s)), uplo, B, A)
    elseif SIDE === :R && UPLO === :L && DIAG === :U
        #
        #   B ← B L*   (unit diagonal)
        #
        strsx_gpu!(s, trans, Val(false), uplo, B, A)
    elseif SIDE === :L && UPLO === :L && DIAG === :U
        #
        #   B ← L* B   (unit diagonal)
        #
        strsx_left_gpu!(s, trans, A, B)
    else
        unsupported("strsx! with side = $SIDE, uplo = $UPLO, diag = $DIAG")
    end

    return B
end

#
#   A ← A*, on the triangle of A: X ← I A* (strsx! on the identity), then the triangle of X into A (the
#   diagonal too when it is not unit). The other triangle of A is left as it is, as on the CPU.
#
function CPU.strtri!(s::AbstractSemiring, uplo::Val{UPLO}, diag::Val{DIAG}, A::DenseCuMatrix{T}; nt::Integer = 1) where {T, UPLO, DIAG}
    @assert size(A, 1) == size(A, 2)
    (UPLO === :U || DIAG === :U) || unsupported("strtri! with uplo = $UPLO, diag = $DIAG")
    n = size(A, 1)
    X = CuMatrix{T}(undef, n, n)
    identity_gpu!(s, X)
    CPU.strsx!(s, Val(:R), Val(:N), uplo, diag, A, X)
    copytri_gpu!(A, X, uplo, Val(DIAG === :N))
    CUDA.unsafe_free!(X)
    return A
end

function CPU.sgetrs!(s::AbstractSemiring, side::Val{SIDE}, trans::Val{TRANS}, A::DenseCuMatrix, B::DenseCuMatrix; nt::Integer = 1) where {SIDE, TRANS}
    if MF.isforward(:L, TRANS, SIDE)
        CPU.strsx!(s, side, trans, Val(:L), Val(:U), A, B)
        CPU.strsx!(s, side, trans, Val(:U), Val(:N), A, B)
    else
        CPU.strsx!(s, side, trans, Val(:U), Val(:N), A, B)
        CPU.strsx!(s, side, trans, Val(:L), Val(:U), A, B)
    end

    return B
end

function CPU.sgetri!(s::AbstractSemiring, C::DenseCuMatrix{T}, A::DenseCuMatrix{T}; nt::Integer = 1) where {T}
    @assert size(A, 1) == size(A, 2)
    @assert size(C) == size(A)
    #
    #   C ← U
    #
    fill!(C, szero(s, T, Val(:N)))
    copytri_gpu!(C, A, Val(:U), Val(true))
    #
    #   C ← U*        C ← C L* = U* L*
    #
    CPU.strtri!(s, Val(:U), Val(:N), C)
    CPU.strsx!(s, Val(:R), Val(:N), Val(:L), Val(:U), A, C)
    return C
end

# A ← X on the upper (uplo :U) or lower triangle, with the diagonal when diag is true
function copytri_gpu!(A::AbstractMatrix, X::AbstractMatrix, ::Val{UPLO}, ::Val{DIAG}) where {UPLO, DIAG}
    function kernel(A, X)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(A, 1)
            @inbounds while j <= size(A, 2)
                if UPLO === :U ? (DIAG ? i <= j : i < j) : (DIAG ? i >= j : i > j)
                    A[i, j] = X[i, j]
                end

                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(A, 1), size(A, 2), A, X)
    return A
end
