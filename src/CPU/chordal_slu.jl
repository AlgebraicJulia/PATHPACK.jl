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

# ===== copyto! =====

const SCOPYTO_LEAF = 2^15

function Base.copyto!(F::ChordalSLU, A::SparseMatrixCSC; nt::Integer = nthreads())
    return scopyto_mt!(F, A, nt)
end

function scopyto_st!(F::ChordalSLU{<:Any, T, I}, A::SparseMatrixCSC) where {T, I}
    n = nov(F.S.S.res)
    h = nv(F.S.S.res)

    ind = FVector{I}(undef, n)
    cur = FVector{I}(undef, h)

    fill!(ind, zero(I))

    @inbounds for f in oneto(h)
        cur[f] = pointers(F.S.S.sep)[f]
    end

    szerorec_st!(F.s, F.LDval, Val(:N))
    szerorec_st!(F.s, F.LLval, Val(:N))
    szerorec_st!(F.s, F.UDval, Val(:N))
    szerorec_st!(F.s, F.ULval, Val(:N))
    szerorec_st!(F.s, F.Nval,  Val(:N))

    scopyto_loop!(F, A, ind, cur, one(I), h)
    return F
end

function scopyto_mt!(F::ChordalSLU{<:Any, T, I}, A::SparseMatrixCSC, nt::Integer) where {T, I}
    if nt <= 1 || nnz(A) < SCOPYTO_LEAF
        scopyto_st!(F, A)
    else
        res = F.S.S.res

        h = nv(res)
        m = 4nt

        work = FVector{I}(undef, h + one(I))
        fbnd = FVector{I}(undef, m + 1)

        szerorec_mt!(F.s, F.LDval, Val(:N), nt)
        szerorec_mt!(F.s, F.LLval, Val(:N), nt)
        szerorec_mt!(F.s, F.UDval, Val(:N), nt)
        szerorec_mt!(F.s, F.ULval, Val(:N), nt)
        szerorec_mt!(F.s, F.Nval,  Val(:N), nt)

        work[begin] = zero(I)

        @inbounds for f in oneto(h)
            w = work[f]

            for j in neighbors(res, f)
                c = F.cperm[j]
                w += A.colptr[c + one(I)] - A.colptr[c]
            end

            work[f + one(I)] = w
        end

        fbnd[begin] = f = one(I)

        @inbounds for k in 1:m - 1
            w = fld(widemul(work[h + one(I)], k), m)

            while f <= h && work[f + one(I)] < w
                f += one(I)
            end

            fbnd[k + 1] = f
        end

        fbnd[m + 1] = h + one(I)

        @threads for t in 1:nt
            scopyto_task!(F, A, fbnd, m, t, nt)
        end
    end

    return F
end

function scopyto_task!(F::ChordalSLU{<:Any, T, I}, A::SparseMatrixCSC, fbnd::AbstractVector{I}, m::Integer, t::Integer, nt::Integer) where {T, I}
    n = nov(F.S.S.res)
    h = nv(F.S.S.res)

    ind = FVector{I}(undef, n)
    cur = FVector{I}(undef, h)

    fill!(ind, zero(I))

    @inbounds for f in oneto(h)
        cur[f] = pointers(F.S.S.sep)[f]
    end

    for k in t:nt:m
        fstrt = fbnd[k]
        fstop = fbnd[k + 1] - one(I)

        if fstrt <= fstop
            scopyto_loop!(F, A, ind, cur, fstrt, fstop)
        end
    end

    return
end

function scopyto_loop!(
        F::ChordalSLU{<:Any, T, I},
        A::SparseMatrixCSC,
        ind::AbstractVector{I},
        cur::AbstractVector{I},
        fstrt::I,
        fstop::I,
    ) where {T, I}
    S = F.S.S

    res = S.res
    sep = S.sep
    idx = S.idx

    Dptr = S.Dptr
    Lptr = S.Lptr

    Bptr = F.S.Bptr
    fcc = F.S.fcc

    rinvp = F.rinvp
    cperm = F.cperm

    @inbounds for f in fstrt:fstop
        nn = eltypedegree(res, f)
        na = eltypedegree(sep, f)

        rlo = pointers(res)[f]
        rhi = pointers(res)[f + one(I)] - one(I)

        sstrt = pointers(sep)[f]
        sstop = pointers(sep)[f + one(I)] - one(I)

        Dp = Dptr[f]
        Lp = Lptr[f]

        jstrt = Bptr[fcc[f]]

        for q in sstrt:sstop
            ind[targets(sep)[q]] = q - sstrt + one(I)
        end

        for j in rlo:rhi
            c = cperm[j]

            for r in pointers(F.S.N)[j]:pointers(F.S.N)[j + one(I)] - one(I)
                ind[targets(F.S.N)[r]] = r
            end

            for p in A.colptr[c]:A.colptr[c + one(I)] - one(I)
                i = convert(I, rinvp[A.rowval[p]])

                if i < jstrt
                    r = ind[i]

                    if ispositive(r)
                        F.Nval[r] = A.nzval[p]
                    end
                elseif rlo <= i <= rhi
                    o = Dp + (i - rlo) + (j - rlo) * nn
                    F.LDval[o] = A.nzval[p]
                    F.UDval[o] = A.nzval[p]
                elseif i > j
                    k = ind[i]

                    if ispositive(k)
                        F.LLval[Lp + (k - one(I)) + (j - rlo) * na] = A.nzval[p]
                    end
                else
                    g = idx[i]
                    qstrt = pointers(sep)[g]
                    qstop = pointers(sep)[g + one(I)]

                    q = cur[g]

                    while q < qstop && targets(sep)[q] < j
                        q += one(I)
                    end

                    cur[g] = q

                    if q < qstop && targets(sep)[q] == j
                        mlo = pointers(res)[g]
                        mm = eltypedegree(res, g)
                        F.ULval[Lptr[g] + (i - mlo) + (q - qstrt) * mm] = A.nzval[p]
                    end
                end
            end

            for r in pointers(F.S.N)[j]:pointers(F.S.N)[j + one(I)] - one(I)
                ind[targets(F.S.N)[r]] = zero(I)
            end
        end

        for q in sstrt:sstop
            ind[targets(sep)[q]] = zero(I)
        end
    end

    return
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
    if SIDE === :L || B isa AbstractVector
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

    if SIDE === :L || B isa AbstractVector
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

    if SIDE === :L || B isa AbstractVector
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
    if SIDE === :L || B isa AbstractVector
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

    if SIDE === :L || B isa AbstractVector
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

    if SIDE === :L || B isa AbstractVector
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
#   sgetre!(F, Val(:L), Val(:N), b, k)   b ← A*  eₖ
#   sgetre!(F, Val(:L), Val(:T), b, k)   b ← A*ᵀ eₖ
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
