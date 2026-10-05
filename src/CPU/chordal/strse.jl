# ===== strse! =====

function strse!(
        s::AbstractSemiring,
        side::Val,
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        C::AbstractVector{T},
        M::AbstractVector{T},
        k::I,
    ) where {UPLO, T, I}
    res  = A.S.res
    sep  = A.S.sep
    pnt  = A.S.pnt
    Dptr = A.S.Dptr
    Lptr = A.S.Lptr
    #
    # walk the path from the leaf idx(k) up to the root, scattering
    # each front's update into C
    #
    f = A.S.idx[k]
    r = f

    while ispositive(f)
        nn = eltypedegree(res, f)

        if isone(nn)
            strse_fwd_1!(s, C, A.Dval, A.Lval, Dptr, Lptr, res, sep, f, trans, A.uplo, diag, side)
        else
            strse_fwd!(s, C, M, A.Dval, A.Lval, Dptr, Lptr, res, sep, f, trans, A.uplo, diag, side)
        end

        r = f
        f = pnt[f]
    end

    return r
end

# ===== strse_fwd! =====

function strse_fwd!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        M::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        f::I,
        trans::Val,
        uplo::Val{UPLO},
        diag::Val,
        side::Val{SIDE},
    ) where {T, I, UPLO, SIDE}
    #
    # nn is the size of the residual at node f
    #
    #     nn = | res(f) |
    #
    nn = eltypedegree(res, f)
    #
    # na is the size of the separator at node f
    #
    #     na = | sep(f) |
    #
    na = eltypedegree(sep, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)
    Rp = pointers(res)[f]
    Dp = Dptr[f]
    Lp = Lptr[f]
    #
    #          res(f)
    #     L = [ D₁₁ ] res(f)
    #         [ L₂₁ ] sep(f)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    # C₁ is the residual of C at node f, holding the contributions
    # scattered up from the path so far
    #
    #     C₁ = C[res(f)]
    #
    C₁ = view(C, Rp:Rp + nn - one(I))
    #
    #   C₁ ← D₁₁* C₁
    #
    strsx!(s, side, trans, uplo, diag, D₁₁, C₁)

    if ispositive(na)
        M₂ = view(M, oneto(na))
        szerorec!(s, M₂, trans)
        #
        #   M₂ ← L₂₁ C₁
        #
        if SIDE === :L
            sgemx!(s, trans, Val(:N), M₂, L₂₁, C₁)
        else
            sgemx!(s, Val(:N), trans, M₂, C₁, L₂₁)
        end
        #
        #   C₂ ← C₂ + M₂
        #
        sscatteradd!(s, trans, C, M₂, fsep)
    end

    return
end

# ===== strse_fwd_1! =====

function strse_fwd_1!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val{DIAG},
        side::Val{SIDE},
    ) where {T, I, TRANS, UPLO, DIAG, SIDE}
    #
    # na is the size of the separator at node f
    #
    #     na = | sep(f) |
    #
    na = eltypedegree(sep, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)
    Rp = pointers(res)[f]
    Dp = Dptr[f]
    Lp = Lptr[f]
    #
    #          res(f)
    #     L = [ d₁₁ ] res(f)
    #         [ l₂₁ ] sep(f)
    #
    d₁₁ = Dval[Dp]
    l₂₁ = view(Lval, Lp:Lp + na - one(I))
    #
    #   c₁ ← d₁₁* c₁
    #
    if !isintegral(s) && DIAG === :N
        v = sstar(s, d₁₁)

        if SIDE === :L
            C[Rp] = sprod(s, v, C[Rp], trans, Val(:N))
        else
            C[Rp] = sprod(s, C[Rp], v, Val(:N), trans)
        end
    end
    #
    #   C₂ ← C₂ + l₂₁ c₁
    #
    if ispositive(na)
        v = C[Rp]

        @inbounds for i in oneto(na)
            if SIDE === :L
                C[fsep[i]] = smuladd(s, l₂₁[i], v, C[fsep[i]], trans, Val(:N))
            else
                C[fsep[i]] = smuladd(s, v, l₂₁[i], C[fsep[i]], Val(:N), trans)
            end
        end
    end

    return
end
