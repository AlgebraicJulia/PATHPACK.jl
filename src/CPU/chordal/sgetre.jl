# ===== sgetre! =====

function sgetre!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        C::AbstractVector{T},
        W::TDWorkspace{T, I},
        pool,
        k::I,
        nt::Integer,
        b::I,
        fstrt::I,
        fstop::I,
    ) where {SIDE, TRANS, T, I}
    M = view(W.Mval, oneto(L.S.nFval))

    if isforward(:L, TRANS, SIDE)
        strse!(s, side, trans, Val(:U), L, C, M, k)
        strsx_mt!(s, side, trans, Val(:N), U, C, W, pool, nt, b, fstrt, fstop)
    else
        strse!(s, side, trans, Val(:N), U, C, M, k)
        strsx_mt!(s, side, trans, Val(:U), L, C, W, pool, nt, b, fstrt, fstop)
    end

    return C
end
