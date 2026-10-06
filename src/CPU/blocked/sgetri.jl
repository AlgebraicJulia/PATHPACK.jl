# ===== sgetri! =====

function sgetri!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        Bptr::AbstractVector{I},
        Fptr::AbstractVector{I},
        fcc::AbstractVector{I},
        nBptr::I,
        Nptr::AbstractVector{I},
        Ntgt::AbstractVector{I},
        Nval::AbstractVector{T},
        C::AbstractMatrix;
        nt::Integer = nthreads(),
    ) where {T, I}
    n = convert(I, size(C, 2))

    pool = spool_mt(s, T, nt)
    #
    # compute diagonal blocks
    #
    #        [ A₁₁*          ]
    #   C ←  [       ⋱       ]
    #        [          Aₖₖ* ]
    #
    sgetri_mt!(s, L, U, C, pool, nt)
    #
    # compute off-diagonal blocks
    #
    #        [ A₁₁*          ]
    #   C ←  [  ⋮    ⋱       ]
    #        [ Aₖ₁*  ⋯  Aₖₖ* ]
    #
    if ispositive(n) && Nptr[n + one(I)] > one(I)
        W = DivisionWorkspace{T}(L.S, n)
        sgetri_offd!(s, L, U, Bptr, Fptr, fcc, nBptr, Nptr, Ntgt, Nval, C, W, pool, nt)
    end

    return C
end

# ===== sgetri_offd! =====

function sgetri_offd!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        Bptr::AbstractVector{I},
        Fptr::AbstractVector{I},
        fcc::AbstractVector{I},
        nBptr::I,
        Nptr::AbstractVector{I},
        Ntgt::AbstractVector{I},
        Nval::AbstractVector{T},
        C::AbstractMatrix,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
    ) where {T, I}
    n = convert(I, size(C, 2))

    idx = L.S.idx

    anc = FVector{I}(undef, nBptr)
    mark = FVector{I}(undef, nBptr)
    stack = FVector{I}(undef, nBptr)
    fill!(mark, zero(I))

    for d in oneto(nBptr)
        dstrt = Bptr[d]
        dstop = Bptr[d + one(I)] - one(I)

        if Nptr[dstrt] < Nptr[dstop + one(I)]
            Cd = view(C, oneto(n), dstrt:dstop)
            #
            # compute the closed ancestors
            #
            #   anc(d)
            #
            tstrt = sgetri_reach!(anc, mark, stack, d, d, Bptr, Nptr, Ntgt, idx, fcc)

            for t in tstrt:nBptr
                c = anc[t]

                jstrt = Bptr[c]
                jstop = Bptr[c + one(I)] - one(I)
                #
                #   Ccd ← Acc* Ccd
                #
                if c != d
                    fstrt = Fptr[c]
                    fstop = Fptr[c + one(I)] - one(I)
                    sgetrs_mt!(s, Val(:L), Val(:N), L, U, Cd, W, pool, nt, fstrt, fstop)
                end
                #
                #   Cbd ← Cbd + Abc Ccd     for b → c
                #
                if Nptr[jstrt] < Nptr[jstop + one(I)]
                    sgemx_sparse!(s, Val(:N), Val(:N), Cd, Nptr, Ntgt, Nval, jstrt, jstop, Cd, nt)
                end
            end
        end
    end

    return C
end

# ===== sgetri_reach! =====

function sgetri_reach!(
        anc::AbstractVector{I},
        mark::AbstractVector{I},
        stack::AbstractVector{I},
        tag::I,
        d::I,
        Bptr::AbstractVector{I},
        Nptr::AbstractVector{I},
        Ntgt::AbstractVector{I},
        idx::AbstractVector{I},
        fcc::AbstractVector{I},
    ) where {I}
    top = convert(I, length(anc)) + one(I)
    head = one(I)

    @inbounds anc[head] = d
    @inbounds mark[d] = tag
    @inbounds stack[head] = Nptr[Bptr[d]]

    @inbounds while ispositive(head)
        c = anc[head]
        p = stack[head]
        pstop = Nptr[Bptr[c + one(I)]] - one(I)

        b = zero(I)

        while p <= pstop
            a = fcc[idx[Ntgt[p]]]
            p += one(I)

            if mark[a] != tag
                b = a
                break
            end
        end

        stack[head] = p

        if ispositive(b)
            mark[b] = tag
            head += one(I)
            anc[head] = b
            stack[head] = Nptr[Bptr[b]]
        else
            head -= one(I)
            top -= one(I)
            anc[top] = c
        end
    end

    return top
end
