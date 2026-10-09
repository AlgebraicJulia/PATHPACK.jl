struct ChordalSLU{
        Sem <: AbstractSemiring,
        T,
        I,
        LDvl <: AbstractVector{T},
        LLvl <: AbstractVector{T},
        UDvl <: AbstractVector{T},
        ULvl <: AbstractVector{T},
        NVal <: AbstractVector{T},
        RPrm <: AbstractVector{I},
        RIvp <: AbstractVector{I},
        CPrm <: AbstractVector{I},
        CIvp <: AbstractVector{I},
    } <: AbstractSLU{T}
    s::Sem
    S::ChordalSSymbolic{I}
    LDval::LDvl
    LLval::LLvl
    UDval::UDvl
    ULval::ULvl
    Nval::NVal
    rperm::RPrm
    rinvp::RIvp
    cperm::CPrm
    cinvp::CIvp
end

const FChordalSLU{Sem, T, I} = ChordalSLU{
    Sem,
    T,
    I,
    FVector{T},
    FVector{T},
    FVector{T},
    FVector{T},
    FVector{T},
    FVector{I},
    FVector{I},
    FVector{I},
    FVector{I},
}

const DChordalSLU{Sem, T, I} = ChordalSLU{
    Sem,
    T,
    I,
    Vector{T},
    Vector{T},
    Vector{T},
    Vector{T},
    Vector{T},
    Vector{I},
    Vector{I},
    Vector{I},
    Vector{I},
}

function ChordalSLU(s::AbstractSemiring, A::SparseMatrixCSC{T}; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM) where {T}
    P, S = ssymbolic(A; alg)
    return ChordalSLU(s, T, S, P.perm, P.invp, P.perm, P.invp)
end

function ChordalSLU(s::AbstractSemiring, ::Type{T}, S::ChordalSSymbolic{I}, rperm, rinvp, cperm, cinvp) where {T, I}
    L = FChordalTriangular{:N, :L, T, I}(S.S)
    U = FChordalTriangular{:N, :U, T, I}(S.S)
    Nval = FVector{T}(undef, ne(S.N))
    return ChordalSLU(s, S, L.Dval, L.Lval, U.Dval, U.Lval, Nval, rperm, rinvp, cperm, cinvp)
end

function ChordalSLU{Sem}(F::ChordalSLU) where {Sem}
    return ChordalSLU(Sem(), F.S, F.LDval, F.LLval, F.UDval, F.ULval, F.Nval, F.rperm, F.rinvp, F.cperm, F.cinvp)
end

function ncc(F::ChordalSLU)
    return ncc(F.S)
end

function components(F::ChordalSLU)
    return components(F.S)
end

function lowertriangular(F::ChordalSLU)
    return ChordalTriangular{:N, :L}(F.S.S, F.LDval, F.LLval)
end

function uppertriangular(F::ChordalSLU)
    return ChordalTriangular{:N, :U}(F.S.S, F.UDval, F.ULval)
end

function Base.size(F::ChordalSLU)
    return size(F.S)
end

function Base.size(F::ChordalSLU, d::Integer)
    return size(F.S, d)
end

function Base.getproperty(F::ChordalSLU, name::Symbol)
    if name === :L
        return lowertriangular(F)
    elseif name === :U
        return uppertriangular(F)
    elseif name === :P
        return Permutation(getfield(F, :rperm), getfield(F, :rinvp))
    elseif name === :Q
        return Permutation(getfield(F, :cperm), getfield(F, :cinvp))
    else
        return getfield(F, name)
    end
end

#
# Each entry A[r, c] goes straight to its slot. With i = rinvp[r], j = cinvp[c] and f the front of j,
# g the front of i:
#
#   i before the component of j       -> the coupling block N (column j)
#   i, j in one front                 -> the diagonal blocks of L and of U (both hold all of it)
#   i > j, other fronts               -> the off-diagonal block of L at f (i in sep(f))
#   i < j, other fronts               -> the off-diagonal block of U at g (j in sep(g))
#
# Every entry has its own slot, so the columns of A are independent and threads take disjoint ranges.
# This is copyto_reference! (permute, then scatter front by front) without the permuted copy of A.
#
function Base.copyto!(F::ChordalSLU{<:Any, T}, A::SparseMatrixCSC) where {T}
    z = szero(F.s, T, Val(:N))
    pfill!(F.LDval, z); pfill!(F.LLval, z)
    pfill!(F.UDval, z); pfill!(F.ULval, z)
    fill!(F.Nval, z)

    n = size(A, 2)
    nchunk = getcolptr(A)[n + 1] - 1 < 2^15 ? 1 : 8 * nthreads()

    if isone(nchunk)
        scatter_columns!(F, A, 1, n)
    else
        @threads for k in 1:nchunk
            scatter_columns!(F, A, cld((k - 1) * n, nchunk) + 1, cld(k * n, nchunk))
        end
    end

    return F
end

function scatter_columns!(F::ChordalSLU{<:Any, T, I}, A::SparseMatrixCSC, c0::Integer, c1::Integer) where {T, I}
    S = F.S.S
    rptr = pointers(S.res); sptr = pointers(S.sep); stgt = targets(S.sep)
    idx = S.idx; Dptr = S.Dptr; Lptr = S.Lptr
    Bptr = F.S.Bptr; nBptr = F.S.nBptr; fcc = F.S.fcc
    Nptr = pointers(F.S.N); Ntgt = targets(F.S.N)
    LD = F.LDval; LL = F.LLval; UD = F.UDval; UL = F.ULval; Nval = F.Nval
    rinvp = F.rinvp; cinvp = F.cinvp
    Aptr = getcolptr(A); Arow = rowvals(A); Aval = nonzeros(A)

    @inbounds for c in c0:c1
        j = convert(I, cinvp[c]); f = idx[j]
        jlo = rptr[f]; nn = rptr[f + one(I)] - jlo
        na = sptr[f + one(I)] - sptr[f]
        jstrt = Bptr[fcc[f]]

        for p in Aptr[c]:(Aptr[c + 1] - 1)
            i = convert(I, rinvp[Arow[p]]); x = Aval[p]

            if i < jstrt
                r = sortedindex(Ntgt, Nptr[j], Nptr[j + one(I)] - one(I), i)
                iszero(r) || (Nval[r] = x)
            elseif jlo <= i < jlo + nn
                o = Dptr[f] + (i - jlo) + (j - jlo) * nn
                LD[o] = x; UD[o] = x
            elseif i > j
                k = sortedindex(stgt, sptr[f], sptr[f + one(I)] - one(I), i)
                iszero(k) || (LL[Lptr[f] + (k - sptr[f]) + (j - jlo) * na] = x)
            else
                g = idx[i]; ilo = rptr[g]; mm = rptr[g + one(I)] - ilo
                k = sortedindex(stgt, sptr[g], sptr[g + one(I)] - one(I), j)
                iszero(k) || (UL[Lptr[g] + (i - ilo) + (k - sptr[g]) * mm] = x)
            end
        end
    end

    return
end

# the index of v in the sorted range tgt[lo:hi], or 0
function sortedindex(tgt::AbstractVector{I}, lo::I, hi::I, v::I) where {I}
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1; t = tgt[mid]
        t == v && return mid
        t < v ? (lo = mid + one(I)) : (hi = mid - one(I))
    end

    return zero(I)
end

# fill! on threads, for the large factor arrays
function pfill!(x::AbstractVector, v)
    n = length(x)
    nchunk = n < 2^18 ? 1 : 4 * nthreads()

    if isone(nchunk)
        fill!(x, v)
    else
        @threads for k in 1:nchunk
            fill!(view(x, (cld((k - 1) * n, nchunk) + 1):cld(k * n, nchunk)), v)
        end
    end

    return x
end

# copyto! through a permuted copy of A (the reference for the direct scatter above)
function copyto_reference!(F::ChordalSLU, A::SparseMatrixCSC)
    A = permute(A, F.rperm, F.cperm)
    scopyto_offd!(F, A)
    scopyto!(F.s, F.L, A)
    scopyto!(F.s, F.U, A)
    return F
end

function scopyto_offd!(F::ChordalSLU{<:Any, T, I}, A::SparseMatrixCSC) where {T, I}
    n = convert(I, size(A, 2))

    Bptr = F.S.Bptr
    nBptr = F.S.nBptr
    Nptr = pointers(F.S.N)
    Ntgt = targets(F.S.N)
    Nval = F.Nval

    Aptr = getcolptr(A)
    Atgt = rowvals(A)
    Aval = nonzeros(A)

    fill!(Nval, szero(F.s, T, Val(:N)))

    q = one(I)

    @inbounds for c in oneto(nBptr)
        jstrt = Bptr[c]
        jstop = Bptr[c + one(I)] - one(I)

        for j in jstrt:jstop
            pstrt = Aptr[j]
            pstop = Aptr[j + one(I)] - one(I)
            Aptr[j] = q

            r = Nptr[j]

            for p in pstrt:pstop
                i = Atgt[p]

                if i < jstrt
                    while Ntgt[r] < i
                        r += one(I)
                    end

                    Nval[r] = Aval[p]
                else
                    Atgt[q] = i
                    Aval[q] = Aval[p]
                    q += one(I)
                end
            end
        end
    end

    Aptr[n + one(I)] = q
    resize!(Atgt, q - one(I))
    resize!(Aval, q - one(I))
    return A
end

function scopyto!(s::AbstractSemiring, A::ChordalTriangular{<:Any, <:Any, T}, B::SparseMatrixCSC) where {T}
    fill!(A, szero(s, T, Val(:N)))
    return copy_scatter!(A, B)
end

# ===== sgetrf! =====

function sgetrf!(F::ChordalSLU; nt::Integer = nthreads())
    sgetrf!(F.s, F.L, F.U; nt)
    return F
end

function sgetrp!(F::ChordalSLU; nt::Integer = nthreads())
    sgetrp!(F.s, lowertriangular(F), uppertriangular(F); nt)
    return F
end

# ===== sgetrs! =====

function sgetrs!(F::ChordalSLU{<:Any, T, I}, side::Val{SIDE}, trans::Val{TRANS}, B::AbstractVecOrMat; nt::Integer = nthreads()) where {T, I, SIDE, TRANS}
    if SIDE === :L
        m = size(B, 1)
        n = size(B, 2)
    else
        m = size(B, 2)
        n = size(B, 1)
    end

    if isforward(:U, TRANS, SIDE)
        invp = F.cinvp
        perm = F.rperm
    else
        invp = F.rinvp
        perm = F.cperm
    end

    work = FVector{T}(undef, m * min(8, n))

    if SIDE === :L
        permuterows!(B, work, invp)
    else
        permutecols!(B, work, invp)
    end

    if B isa AbstractVector
        pool = nothing
    else
        pool = spool_mt(F.s, T, nt)
    end

    sgetrs_mt!(F.s, side, trans, F.L, F.U, F.S.Bptr, F.S.Fptr, F.S.nBptr, pointers(F.S.N), targets(F.S.N), F.Nval, B, pool, nt)

    if SIDE === :L
        permuterows!(B, work, perm)
    else
        permutecols!(B, work, perm)
    end

    return B
end

# ===== subtree-parallel solve =====

function TDSymbolic(F::ChordalSLU; nt::Integer = nthreads())
    return TDSymbolic(F.S.S, F.S.Fptr, F.S.nBptr; nt)
end

function TDWorkspace(F::ChordalSLU{<:Any, T, I}, nrhs::Integer; nt::Integer = nthreads()) where {T, I}
    D = TDSymbolic(F; nt)
    return TDWorkspace{T}(F.S.S, D, nrhs; nt)
end

function sgetrs!(F::ChordalSLU{<:Any, T, I}, side::Val{SIDE}, trans::Val{TRANS}, B::AbstractVecOrMat, D::TDSymbolic{I}; nt::Integer = nthreads()) where {T, I, SIDE, TRANS}
    if B isa AbstractVector
        nrhs = 1
    elseif SIDE === :L
        nrhs = size(B, 2)
    else
        nrhs = size(B, 1)
    end

    W = TDWorkspace{T}(F.S.S, D, nrhs; nt)
    return sgetrs!(F, side, trans, B, W; nt)
end

function sgetrs!(F::ChordalSLU{<:Any, T, I}, side::Val{SIDE}, trans::Val{TRANS}, B::AbstractVecOrMat, W::TDWorkspace{T, I}; nt::Integer = nthreads()) where {T, I, SIDE, TRANS}
    if SIDE === :L
        m = size(B, 1)
        n = size(B, 2)
    else
        m = size(B, 2)
        n = size(B, 1)
    end

    if isforward(:U, TRANS, SIDE)
        invp = F.cinvp
        perm = F.rperm
    else
        invp = F.rinvp
        perm = F.cperm
    end

    work = FVector{T}(undef, m * min(8, n))

    if SIDE === :L
        permuterows!(B, work, invp)
    else
        permutecols!(B, work, invp)
    end

    if B isa AbstractVector
        pool = nothing
    else
        pool = spool_mt(F.s, T, nt)
    end

    sgetrs_mt!(F.s, side, trans, F.L, F.U, F.S.Bptr, F.S.Fptr, F.S.nBptr, pointers(F.S.N), targets(F.S.N), F.Nval, B, W, pool, nt)

    if SIDE === :L
        permuterows!(B, work, perm)
    else
        permutecols!(B, work, perm)
    end

    return B
end

# ===== sgetre! =====

#
# Solve against an elementary vector eₖ, overwriting b:
#
#   sgetre!(F, Val(:L), Val(:N), b, k)   b ← A* eₖ    (column k of A*)
#   sgetre!(F, Val(:L), Val(:T), b, k)   b ← A*ᵀ eₖ   (row k of A*)
#
# Side :R treats b as a row vector.
#
function sgetre!(F::ChordalSLU{<:Any, T, I}, side::Val{SIDE}, trans::Val{TRANS}, b::AbstractVector, k::Integer; nt::Integer = nthreads()) where {T, I, SIDE, TRANS}
    D = TDSymbolic(F; nt)
    return sgetre!(F, side, trans, b, k, D; nt)
end

function sgetre!(F::ChordalSLU{<:Any, T, I}, side::Val{SIDE}, trans::Val{TRANS}, b::AbstractVector, k::Integer, D::TDSymbolic{I}; nt::Integer = nthreads()) where {T, I, SIDE, TRANS}
    W = TDWorkspace{T}(F.S.S, D, one(I); nt)
    return sgetre!(F, side, trans, b, k, W; nt)
end

function sgetre!(F::ChordalSLU{<:Any, T, I}, side::Val{SIDE}, trans::Val{TRANS}, b::AbstractVector, k::Integer, W::TDWorkspace{T, I}; nt::Integer = nthreads()) where {T, I, SIDE, TRANS}
    n = convert(I, size(F, 1))

    @assert length(b) == n
    @assert one(I) <= k <= n

    if isforward(:U, TRANS, SIDE)
        invp = F.cinvp
        perm = F.rperm
    else
        invp = F.rinvp
        perm = F.cperm
    end
    #
    #   b ← P eₖ
    #
    j = convert(I, invp[k])
    fill!(b, szero(F.s, T, trans))
    @inbounds b[j] = sone(F.s, T, trans)
    #
    # c is the component containing j
    #
    #     c = fcc(idx(j))
    #
    c = F.S.fcc[F.S.S.idx[j]]
    #
    #   b ← U* L* b
    #
    sgetre!(F.s, side, trans, F.L, F.U, F.S.Bptr, F.S.Fptr, F.S.nBptr, pointers(F.S.N), targets(F.S.N), F.Nval, b, W, nothing, j, nt, c)
    #
    #   b ← P⁻¹ b
    #
    work = FVector{T}(undef, n)
    permuterows!(b, work, perm)

    return b
end

function sgetri!(F::ChordalSLU{<:Any, T}, C::AbstractMatrix; nt::Integer = nthreads()) where {T}
    @assert size(F, 1) == size(C, 1) == size(C, 2)
    #
    #   C ← U* L*
    #
    sgetri!(F.s, F.L, F.U, F.S.Bptr, F.S.Fptr, F.S.fcc, F.S.nBptr, pointers(F.S.N), targets(F.S.N), F.Nval, C; nt)
    #
    #   C ← P⁻¹ C Q⁻¹
    #
    permuterowscols!(C, F.rperm, F.cperm; nt)

    return C
end
