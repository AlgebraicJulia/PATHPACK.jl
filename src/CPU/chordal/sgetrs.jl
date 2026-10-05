# ===== sgetrs! =====

function sgetrs!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat;
        nt::Integer = nthreads(),
    ) where {T, I, SIDE, TRANS}
    S = L.S

    if B isa AbstractVector
        nrhs = one(I)
        pool = nothing
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
        pool = spool_mt(s, T, nt)
    else
        nrhs = convert(I, size(B, 1))
        pool = spool_mt(s, T, nt)
    end

    W = DivisionWorkspace{T}(S, nrhs)
    return sgetrs_mt!(s, side, trans, L, U, B, W, pool, nt, one(I), nv(S.res))
end

function sgetrs_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
        fstrt::I,
        fstop::I,
    ) where {T, I, SIDE, TRANS}
    if isforward(:L, TRANS, SIDE)
        strsx_mt!(s, side, trans, Val(:U), L, B, W.Mval, pool, nt, fstrt, fstop)
        strsx_mt!(s, side, trans, Val(:N), U, B, W.Mval, pool, nt, fstrt, fstop)
    else
        strsx_mt!(s, side, trans, Val(:N), U, B, W.Mval, pool, nt, fstrt, fstop)
        strsx_mt!(s, side, trans, Val(:U), L, B, W.Mval, pool, nt, fstrt, fstop)
    end

    return B
end

# ===== subtree-parallel solve =====

function sgetrs!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat,
        W::TDWorkspace{T, I};
        nt::Integer = nthreads(),
    ) where {SIDE, TRANS, T, I}
    if B isa AbstractVector
        pool = nothing
    else
        pool = spool_mt(s, T, nt)
    end

    return sgetrs_mt!(s, side, trans, L, U, B, W, pool, nt)
end

function sgetrs_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat,
        W::TDWorkspace{T, I},
        pool,
        nt::Integer,
    ) where {SIDE, TRANS, T, I}
    return sgetrs_mt!(s, side, trans, L, U, B, W, pool, nt, one(I), one(I), nv(L.S.res))
end

function sgetrs_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat,
        W::TDWorkspace{T, I},
        pool,
        nt::Integer,
        b::I,
        fstrt::I,
        fstop::I,
    ) where {SIDE, TRANS, T, I}
    if isforward(:L, TRANS, SIDE)
        strsx_mt!(s, side, trans, Val(:U), L, B, W, pool, nt, b, fstrt, fstop)
        strsx_mt!(s, side, trans, Val(:N), U, B, W, pool, nt, b, fstrt, fstop)
    else
        strsx_mt!(s, side, trans, Val(:N), U, B, W, pool, nt, b, fstrt, fstop)
        strsx_mt!(s, side, trans, Val(:U), L, B, W, pool, nt, b, fstrt, fstop)
    end

    return B
end
