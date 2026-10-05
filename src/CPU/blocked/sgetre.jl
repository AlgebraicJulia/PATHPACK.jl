# ===== sgetre! =====

function sgetre!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        Bptr::AbstractVector{I},
        Fptr::AbstractVector{I},
        nBptr::I,
        Nptr::AbstractVector{I},
        Ntgt::AbstractVector{I},
        Nval::AbstractVector{T},
        C::AbstractVector{T},
        W::TDWorkspace{T, I},
        pool,
        k::I,
        nt::Integer,
        c::I,
    ) where {SIDE, TRANS, T, I}
    sgetre!(s, side, trans, L, U, C, W, pool, k, nt, c, Fptr[c], Fptr[c + one(I)] - one(I))

    if isforward(:U, TRANS, SIDE)
        for d in c + one(I):nBptr
            fstrt = Fptr[d]
            fstop = Fptr[d + one(I)] - one(I)

            jstrt = Bptr[d]
            jstop = Bptr[d + one(I)] - one(I)

            if SIDE === :L
                sgemx_sparse!(s, trans, Val(:N), C, Nptr, Ntgt, Nval, jstrt, jstop, C, 1)
            else
                sgemx_sparse!(s, Val(:N), trans, C, C, Nptr, Ntgt, Nval, jstrt, jstop, 1)
            end

            if !siszero(s, trans, C, jstrt, jstop)
                sgetrs_mt!(s, side, trans, L, U, C, W, pool, nt, d, fstrt, fstop)
            end
        end
    else
        for d in reverse(oneto(c))
            fstrt = Fptr[d]
            fstop = Fptr[d + one(I)] - one(I)

            jstrt = Bptr[d]
            jstop = Bptr[d + one(I)] - one(I)

            if d < c
                if siszero(s, trans, C, jstrt, jstop)
                    continue
                end

                sgetrs_mt!(s, side, trans, L, U, C, W, pool, nt, d, fstrt, fstop)
            end

            if SIDE === :L
                sgemx_sparse!(s, trans, Val(:N), C, Nptr, Ntgt, Nval, jstrt, jstop, C, 1)
            else
                sgemx_sparse!(s, Val(:N), trans, C, C, Nptr, Ntgt, Nval, jstrt, jstop, 1)
            end
        end
    end

    return C
end
