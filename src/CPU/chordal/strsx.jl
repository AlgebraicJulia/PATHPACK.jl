const STRSX_NB = 8

const STRSX_1_NB = 12

const TD_SPLIT = 4

# ===== strsx! =====

function strsx!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat;
        nt::Integer = nthreads(),
    ) where {SIDE, TRANS, UPLO, T, I}
    if B isa AbstractVector
        pool = nothing
    else
        pool = spool_mt(s, T, nt)
    end

    return strsx_mt!(s, side, trans, diag, A, B, pool, nt)
end

function strsx_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        pool,
        nt::Integer,
    ) where {SIDE, TRANS, UPLO, T, I}
    S = A.S

    if B isa AbstractVector
        nrhs = 1
    elseif SIDE === :L
        nrhs = size(B, 2)
    else
        nrhs = size(B, 1)
    end

    if B isa AbstractMatrix && nt > 1 && nrhs >= 4nt
        tsize = fld(nrhs, nt)

        @threads for t in 1:nt
            tstrt = (t - 1) * tsize + 1

            if t < nt
                tstop = t * tsize
            else
                tstop = nrhs
            end

            trhs = tstop - tstrt + 1

            if SIDE === :L
                Bt = view(B, :, tstrt:tstop)
            else
                Bt = view(B, tstrt:tstop, :)
            end

            Wt = DivisionWorkspace{T}(S, trhs)
            poolt = spool_mt(s, T, 1)
            strsx_mt!(s, side, trans, diag, A, Bt, Wt.Mval, poolt, 1, one(I), nv(A.S.res))
        end
    else
        W = DivisionWorkspace{T}(S, nrhs)
        strsx_mt!(s, side, trans, diag, A, B, W.Mval, pool, nt, one(I), nv(A.S.res))
    end

    return B
end

function strsx_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        Mval::AbstractVector{T},
        pool,
        nt::Integer,
        fstrt::I,
        fstop::I,
        bnd = nothing,
    ) where {SIDE, TRANS, UPLO, T, I}
    S = A.S

    if B isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    if isforward(UPLO, TRANS, SIDE)
        for f in fstrt:fstop
            nn = eltypedegree(S.res, f)

            if isone(nn)
                strsx_fwd_1!(s, B, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, f, trans, A.uplo, diag, side, bnd)
            else
                strsx_fwd!(s, B, Mval, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, f, trans, A.uplo, diag, side, bnd)
            end
        end
    else
        for f in reverse(fstrt:fstop)
            nn = eltypedegree(S.res, f)

            if isone(nn)
                strsx_bwd_1!(s, B, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, f, trans, A.uplo, diag, side)
            else
                strsx_bwd!(s, B, Mval, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, f, trans, A.uplo, diag, side)
            end
        end
    end

    return B
end

# ===== strsx_fwd! =====

function strsx_fwd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        pool,
        nt::Integer,
        nrhs::I,
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val,
        side::Val{SIDE},
        bnd,
    ) where {T, I, TRANS, UPLO, SIDE}
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

    if isnothing(bnd) && SIDE === :R && nn < STRSX_NB
        strsx_fwd_upd_small!(s, C, D₁₁, L₂₁, res, sep, nn, na, nrhs, f, diag, trans, uplo, side)
    else
        strsx_fwd_upd!(s, C, Mval, D₁₁, L₂₁, res, sep, na, nrhs, pool, nt, f, trans, uplo, diag, side, bnd)
    end

    return
end

function strsx_fwd_upd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        na::I,
        nrhs::I,
        pool,
        nt::Integer,
        f::I,
        trans::Val,
        uplo::Val,
        diag::Val,
        side::Val{SIDE},
        bnd,
    ) where {T, I, SIDE}
    #
    #   C = [ C₁ ] res(f)
    #       [ C₂ ] sep(f)
    #
    #
    # fres is the residual at node f
    #
    #     fres = res(f)
    #
    fres = neighbors(res, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    if C isa AbstractVector
        C₁ = view(C, fres)
    elseif SIDE === :L
        C₁ = view(C, fres, oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), fres)
    end
    #
    #   C₁ ← L₁₁* C₁
    #
    if C isa AbstractVector
        strsx!(s, side, trans, uplo, diag, D₁₁, C₁)
    else
        strsx_mt!(s, side, trans, uplo, diag, D₁₁, C₁, pool, nt)
    end

    if ispositive(na)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #   M₂ ← L₂₁ C₁
        #
        szerorec!(s, M₂, trans)

        if C isa AbstractVector
            if SIDE === :L
                sgemx_mt!(s, trans, Val(:N), M₂, L₂₁, C₁, nt)
            else
                sgemx_mt!(s, Val(:N), trans, M₂, C₁, L₂₁, nt)
            end
        elseif SIDE === :L
            sgemx_mt!(s, trans, Val(:N), M₂, L₂₁, C₁, pool, nt)
        else
            sgemx_mt!(s, Val(:N), trans, M₂, C₁, L₂₁, pool, nt)
        end
        #
        #   C₂ ← C₂ + M₂
        #
        if C isa AbstractVector
            sscatteradd!(s, trans, C, M₂, fsep, na, f, bnd)
        else
            sscatteradd!(s, trans, C, M₂, fsep, na, f, side, bnd)
        end
    end

    return
end

function strsx_fwd_upd_small!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::N_OR_R,
        ::Val{:U},
        ::Val{:R},
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    @inbounds for j in oneto(nn)
        c = Rp + j - one(I)

        for i in oneto(j - one(I))
            C[c] = smuladd(s, C[Rp + i - one(I)], D₁₁[i, j], C[c], Val(:N), trans)
        end

        if !isintegral(s) && diag === Val(:N)
            C[c] = sprod(s, C[c], sstar(s, D₁₁[j, j]), Val(:N), trans)
        end
    end

    if ispositive(na)
        @inbounds for j in oneto(na)
            c = fsep[j]

            for i in oneto(nn)
                C[c] = smuladd(s, C[Rp + i - one(I)], L₂₁[i, j], C[c], Val(:N), trans)
            end
        end
    end

    return
end

function strsx_fwd_upd_small!(
        s::AbstractSemiring,
        C::AbstractMatrix{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::N_OR_R,
        ::Val{:U},
        ::Val{:R},
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    Z = sizeof(T)
    sC = stride(C, 2)

    @preserve C begin
        pC = pointer(C)

        @inbounds for j in oneto(nn)
            c = Rp + j - one(I)
            pc = pC + (c - one(I)) * sC * Z

            for i in oneto(j - one(I))
                d = Rp + i - one(I)
                saxpy_kern!(s, Val(:N), trans, Val(:R), pc, pC + (d - one(I)) * sC * Z, D₁₁[i, j], nrhs)
            end

            if !isintegral(s) && diag === Val(:N)
                v = sstar(s, D₁₁[j, j])

                for k in oneto(nrhs)
                    C[k, c] = sprod(s, C[k, c], v, Val(:N), trans)
                end
            end
        end

        if ispositive(na)
            @inbounds for j in oneto(na)
                c = fsep[j]
                pc = pC + (c - one(I)) * sC * Z

                for i in oneto(nn)
                    d = Rp + i - one(I)
                    saxpy_kern!(s, Val(:N), trans, Val(:R), pc, pC + (d - one(I)) * sC * Z, L₂₁[i, j], nrhs)
                end
            end
        end
    end

    return
end

function strsx_fwd_upd_small!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::T_OR_C,
        ::Val{:L},
        ::Val{:R},
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    @inbounds for i in oneto(nn)
        ci = Rp + i - one(I)

        if !isintegral(s) && diag === Val(:N)
            C[ci] = sprod(s, C[ci], sstar(s, D₁₁[i, i]), Val(:N), trans)
        end

        for j in i + one(I):nn
            cj = Rp + j - one(I)
            C[cj] = smuladd(s, C[ci], D₁₁[j, i], C[cj], Val(:N), trans)
        end
    end

    if ispositive(na)
        @inbounds for i in oneto(na)
            c = fsep[i]

            for j in oneto(nn)
                d = Rp + j - one(I)
                C[c] = smuladd(s, C[d], L₂₁[i, j], C[c], Val(:N), trans)
            end
        end
    end

    return
end

function strsx_fwd_upd_small!(
        s::AbstractSemiring,
        C::AbstractMatrix{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::T_OR_C,
        ::Val{:L},
        ::Val{:R},
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    Z = sizeof(T)
    sC = stride(C, 2)

    @preserve C begin
        pC = pointer(C)

        @inbounds for i in oneto(nn)
            ci = Rp + i - one(I)
            pci = pC + (ci - one(I)) * sC * Z

            if !isintegral(s) && diag === Val(:N)
                v = sstar(s, D₁₁[i, i])

                for k in oneto(nrhs)
                    C[k, ci] = sprod(s, C[k, ci], v, Val(:N), trans)
                end
            end

            for j in i + one(I):nn
                cj = Rp + j - one(I)
                saxpy_kern!(s, Val(:N), trans, Val(:R), pC + (cj - one(I)) * sC * Z, pci, D₁₁[j, i], nrhs)
            end
        end

        if ispositive(na)
            @inbounds for i in oneto(na)
                c = fsep[i]
                pc = pC + (c - one(I)) * sC * Z

                for j in oneto(nn)
                    d = Rp + j - one(I)
                    saxpy_kern!(s, Val(:N), trans, Val(:R), pc, pC + (d - one(I)) * sC * Z, L₂₁[i, j], nrhs)
                end
            end
        end
    end

    return
end

# ===== strsx_fwd_1! =====

function strsx_fwd_1!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val{DIAG},
        side::Val{SIDE},
        bnd,
    ) where {T, I, TRANS, UPLO, SIDE, DIAG}
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
    Dp = Dptr[f]
    Lp = Lptr[f]
    Rp = pointers(res)[f]
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

        if C isa AbstractVector
            if SIDE === :L
                C[Rp] = sprod(s, v, C[Rp], trans, Val(:N))
            else
                C[Rp] = sprod(s, C[Rp], v, Val(:N), trans)
            end
        else
            @inbounds for k in oneto(nrhs)
                if SIDE === :L
                    C[Rp, k] = sprod(s, v, C[Rp, k], trans, Val(:N))
                else
                    C[k, Rp] = sprod(s, C[k, Rp], v, Val(:N), trans)
                end
            end
        end
    end

    if ispositive(na)
        #
        #   M₂ ← l₂₁ c₁       C₂ ← C₂ + M₂
        #
        strsx_fwd_upd_1!(s, C, fsep, l₂₁, Rp, na, nrhs, trans, side, f, bnd)
    end

    return
end

function strsx_fwd_upd_1!(s::AbstractSemiring, C::AbstractVecOrMat{T}, fsep::AbstractVector{I}, l₂₁::AbstractVector{T}, Rp::I, na::I, nrhs::I, trans::Val, ::Val{SIDE}) where {T, I, SIDE}
    if C isa AbstractVector
        v = C[Rp]

        @inbounds for i in oneto(na)
            if SIDE === :L
                C[fsep[i]] = smuladd(s, l₂₁[i], v, C[fsep[i]], trans, Val(:N))
            else
                C[fsep[i]] = smuladd(s, v, l₂₁[i], C[fsep[i]], Val(:N), trans)
            end
        end
    else
        if SIDE === :L
            @inbounds for k in oneto(nrhs)
                v = C[Rp, k]

                for i in oneto(na)
                    C[fsep[i], k] = smuladd(s, l₂₁[i], v, C[fsep[i], k], trans, Val(:N))
                end
            end
        else
            Z = sizeof(T)
            sC = stride(C, 2)

            @preserve C begin
                pC = pointer(C)
                pr = pC + (Rp - one(I)) * sC * Z

                @inbounds for i in oneto(na)
                    saxpy_kern!(s, Val(:N), trans, Val(:R), pC + (fsep[i] - one(I)) * sC * Z, pr, l₂₁[i], nrhs)
                end
            end
        end
    end

    return
end

# ===== strsx_bwd! =====

function strsx_bwd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        pool,
        nt::Integer,
        nrhs::I,
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val,
        side::Val{SIDE},
    ) where {T, I, TRANS, UPLO, SIDE}
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
    Dp = Dptr[f]
    Lp = Lptr[f]
    #
    #          res(f) sep(f)
    #     U = [ D₁₁    U₁₂ ] res(f)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :U
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    else
        U₁₂ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    end

    if SIDE === :R && nn < STRSX_NB
        strsx_bwd_upd_small!(s, C, D₁₁, U₁₂, res, sep, nn, na, nrhs, f, diag, trans, uplo, side)
    else
        strsx_bwd_upd!(s, C, Mval, D₁₁, U₁₂, res, sep, na, nrhs, pool, nt, f, trans, uplo, diag, side)
    end

    return
end

function strsx_bwd_upd!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        U₁₂::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        na::I,
        nrhs::I,
        pool,
        nt::Integer,
        f::I,
        trans::Val,
        uplo::Val,
        diag::Val,
        side::Val{SIDE},
    ) where {T, I, SIDE}
    #
    #   C = [ C₁ ] res(f)
    #       [ C₂ ] sep(f)
    #
    #
    # fres is the residual at node f
    #
    #     fres = res(f)
    #
    fres = neighbors(res, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    if C isa AbstractVector
        C₁ = view(C, fres)
    elseif SIDE === :L
        C₁ = view(C, fres, oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), fres)
    end

    if ispositive(na)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #   M₂ ← C₂
        #
        if C isa AbstractVector
            copygatherrec!(M₂, C, fsep)
        else
            copygatherrec!(M₂, C, fsep, side)
        end
        #
        #   C₁ ← U₁₂ M₂ + C₁
        #
        if C isa AbstractVector
            if SIDE === :L
                sgemx_mt!(s, trans, Val(:N), C₁, U₁₂, M₂, nt)
            else
                sgemx_mt!(s, Val(:N), trans, C₁, M₂, U₁₂, nt)
            end
        elseif SIDE === :L
            sgemx_mt!(s, trans, Val(:N), C₁, U₁₂, M₂, pool, nt)
        else
            sgemx_mt!(s, Val(:N), trans, C₁, M₂, U₁₂, pool, nt)
        end
    end
    #
    #   C₁ ← U₁₁* C₁
    #
    if C isa AbstractVector
        strsx!(s, side, trans, uplo, diag, D₁₁, C₁)
    else
        strsx_mt!(s, side, trans, uplo, diag, D₁₁, C₁, pool, nt)
    end

    return
end

function strsx_bwd_upd_small!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        U₁₂::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::N_OR_R,
        ::Val{:L},
        ::Val{:R},
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    @inbounds for j in oneto(nn)
        c = Rp + j - one(I)

        for i in oneto(na)
            d = fsep[i]
            C[c] = smuladd(s, C[d], U₁₂[i, j], C[c], Val(:N), trans)
        end
    end
    @inbounds for j in reverse(oneto(nn))
        c = Rp + j - one(I)

        for i in j + one(I):nn
            d = Rp + i - one(I)
            C[c] = smuladd(s, C[d], D₁₁[i, j], C[c], Val(:N), trans)
        end

        if !isintegral(s) && diag === Val(:N)
            C[c] = sprod(s, C[c], sstar(s, D₁₁[j, j]), Val(:N), trans)
        end
    end

    return
end

function strsx_bwd_upd_small!(
        s::AbstractSemiring,
        C::AbstractMatrix{T},
        D₁₁::AbstractMatrix{T},
        U₁₂::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::N_OR_R,
        ::Val{:L},
        ::Val{:R},
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    Z = sizeof(T)
    sC = stride(C, 2)

    @preserve C begin
        pC = pointer(C)

        @inbounds for j in oneto(nn)
            c = Rp + j - one(I)
            pc = pC + (c - one(I)) * sC * Z

            for i in oneto(na)
                d = fsep[i]
                saxpy_kern!(s, Val(:N), trans, Val(:R), pc, pC + (d - one(I)) * sC * Z, U₁₂[i, j], nrhs)
            end
        end

        @inbounds for j in reverse(oneto(nn))
            c = Rp + j - one(I)
            pc = pC + (c - one(I)) * sC * Z

            for i in j + one(I):nn
                d = Rp + i - one(I)
                saxpy_kern!(s, Val(:N), trans, Val(:R), pc, pC + (d - one(I)) * sC * Z, D₁₁[i, j], nrhs)
            end

            if !isintegral(s) && diag === Val(:N)
                v = sstar(s, D₁₁[j, j])

                for k in oneto(nrhs)
                    C[k, c] = sprod(s, C[k, c], v, Val(:N), trans)
                end
            end
        end
    end

    return
end

function strsx_bwd_upd_small!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        U₁₂::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::T_OR_C,
        ::Val{:U},
        ::Val{:R},
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    if ispositive(na)
        @inbounds for j in oneto(nn)
            cj = Rp + j - one(I)

            for i in oneto(na)
                d = fsep[i]
                C[cj] = smuladd(s, C[d], U₁₂[j, i], C[cj], Val(:N), trans)
            end
        end
    end

    @inbounds for i in reverse(oneto(nn))
        ci = Rp + i - one(I)

        if !isintegral(s) && diag === Val(:N)
            C[ci] = sprod(s, C[ci], sstar(s, D₁₁[i, i]), Val(:N), trans)
        end

        for j in oneto(i - one(I))
            cj = Rp + j - one(I)
            C[cj] = smuladd(s, C[ci], D₁₁[j, i], C[cj], Val(:N), trans)
        end
    end

    return
end

function strsx_bwd_upd_small!(
        s::AbstractSemiring,
        C::AbstractMatrix{T},
        D₁₁::AbstractMatrix{T},
        U₁₂::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::T_OR_C,
        ::Val{:U},
        ::Val{:R},
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    Z = sizeof(T)
    sC = stride(C, 2)

    @preserve C begin
        pC = pointer(C)

        if ispositive(na)
            @inbounds for j in oneto(nn)
                cj = Rp + j - one(I)
                pcj = pC + (cj - one(I)) * sC * Z

                for i in oneto(na)
                    d = fsep[i]
                    saxpy_kern!(s, Val(:N), trans, Val(:R), pcj, pC + (d - one(I)) * sC * Z, U₁₂[j, i], nrhs)
                end
            end
        end

        @inbounds for i in reverse(oneto(nn))
            ci = Rp + i - one(I)
            pci = pC + (ci - one(I)) * sC * Z

            if !isintegral(s) && diag === Val(:N)
                v = sstar(s, D₁₁[i, i])

                for k in oneto(nrhs)
                    C[k, ci] = sprod(s, C[k, ci], v, Val(:N), trans)
                end
            end

            for j in oneto(i - one(I))
                cj = Rp + j - one(I)
                saxpy_kern!(s, Val(:N), trans, Val(:R), pC + (cj - one(I)) * sC * Z, pci, D₁₁[j, i], nrhs)
            end
        end
    end

    return
end

# ===== strsx_bwd_1! =====

function strsx_bwd_1!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val{DIAG},
        side::Val{SIDE},
    ) where {T, I, TRANS, UPLO, SIDE, DIAG}
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
    Dp = Dptr[f]
    Lp = Lptr[f]
    Rp = pointers(res)[f]
    #
    #          res(f) sep(f)
    #     U = [ d₁₁    u₁₂ ] res(f)
    #
    d₁₁ = Dval[Dp]
    u₁₂ = view(Lval, Lp:Lp + na - one(I))

    if ispositive(na)
        #
        #   M₂ ← C₂       c₁ ← u₁₂ M₂ + c₁
        #
        strsx_bwd_upd_1!(s, C, fsep, u₁₂, Rp, na, nrhs, trans, side)
    end
    #
    #   c₁ ← d₁₁* c₁
    #
    if !isintegral(s) && DIAG === :N
        v = sstar(s, d₁₁)

        if C isa AbstractVector
            if SIDE === :L
                C[Rp] = sprod(s, v, C[Rp], trans, Val(:N))
            else
                C[Rp] = sprod(s, C[Rp], v, Val(:N), trans)
            end
        else
            @inbounds for k in oneto(nrhs)
                if SIDE === :L
                    C[Rp, k] = sprod(s, v, C[Rp, k], trans, Val(:N))
                else
                    C[k, Rp] = sprod(s, C[k, Rp], v, Val(:N), trans)
                end
            end
        end
    end

    return
end

function strsx_bwd_upd_1!(s::AbstractSemiring, C::AbstractVecOrMat{T}, fsep::AbstractVector{I}, u₁₂::AbstractVector{T}, Rp::I, na::I, nrhs::I, trans::Val, ::Val{SIDE}) where {T, I, SIDE}
    if C isa AbstractVector
        if SIDE === :L
            C[Rp] = strsx_vec_gather(s, trans, C, 0, fsep, u₁₂, C[Rp], na)
        else
            v = C[Rp]

            @inbounds for i in oneto(na)
                v = smuladd(s, C[fsep[i]], u₁₂[i], v, Val(:N), trans)
            end

            C[Rp] = v
        end
    else
        if SIDE === :L
            @inbounds for k in oneto(nrhs)
                C[Rp, k] = strsx_vec_gather(s, trans, C, (k - 1) * size(C, 1), fsep, u₁₂, C[Rp, k], na)
            end
        elseif nrhs <= STRSX_1_NB
            @inbounds for k in oneto(nrhs)
                v = C[k, Rp]

                for i in oneto(na)
                    v = smuladd(s, C[k, fsep[i]], u₁₂[i], v, Val(:N), trans)
                end

                C[k, Rp] = v
            end
        else
            Z = sizeof(T)
            sC = stride(C, 2)

            @preserve C begin
                pC = pointer(C)
                pr = pC + (Rp - one(I)) * sC * Z

                @inbounds for i in oneto(na)
                    saxpy_kern!(s, Val(:N), trans, Val(:R), pr, pC + (fsep[i] - one(I)) * sC * Z, u₁₂[i], nrhs)
                end
            end
        end
    end

    return
end

# ===== strsx_vec =====

@inline function strsx_vec_gather_step(s::AbstractSemiring, trans::Val, C::AbstractVecOrMat{T}, o::Integer, idx::AbstractVector, a::AbstractVector{T}, d::Vec{W, T}, i::Integer) where {T, W}
    function cf(l)
        return @inbounds(C[o + idx[i + l - 1]])
    end

    function af(l)
        return @inbounds(a[i + l - 1])
    end

    bv = Vec{W, T}(ntuple(cf, Val(W)))
    av = Vec{W, T}(ntuple(af, Val(W)))
    return smuladd(s, av, bv, d, trans, Val(:N))
end

@inline function strsx_vec_gather(s::AbstractSemiring, trans::Val, C::AbstractVecOrMat{T}, o::Integer, idx::AbstractVector, a::AbstractVector{T}, v::T, n::Integer) where {T}
    i = 1
    W = min(vecwidth(T), 8)
    op = compose(trans, Val(:N))

    if n >= W
        d = szero(s, Vec{W, T}, op)

        @inbounds while i + W - 1 <= n
            d = strsx_vec_gather_step(s, trans, C, o, idx, a, d, i)
            i += W
        end

        v = splus(s, v, sreduce(s, d, op), op)
    end

    @inbounds while i <= n
        v = smuladd(s, a[i], C[o + idx[i]], v, trans, Val(:N))
        i += 1
    end

    return v
end

# ===== subtree-parallel solve =====

struct TDSymbolic{I}
    nb::I
    ng::I
    fdsc::FVector{I}
    rts::FBipartiteGraph{I, I}
    bnd::FBipartiteGraph{I, I}
    col::FBipartiteGraph{I, I}
    cut::FVector{I}
    perm::FVector{I}
    gbeg::FVector{I}
    gend::FVector{I}
    cptr::FVector{I}
    gptr::FVector{I}
end

function TDSymbolic(S::ChordalSymbolic{I}; nt::Integer) where {I <: Integer}
    return TDSymbolic(S, convert(I, nt))
end

function TDSymbolic(S::ChordalSymbolic{I}, nt::I) where {I <: Integer}
    nf = nv(S.res)

    Fptr = FVector{I}(undef, two(I))
    Fptr[one(I)] = one(I)
    Fptr[two(I)] = nf + one(I)

    return TDSymbolic(S, Fptr, one(I), nt)
end

function TDSymbolic(S::ChordalSymbolic{I}, Fptr::AbstractVector{I}, nBptr::I; nt::Integer) where {I <: Integer}
    return TDSymbolic(S, Fptr, nBptr, convert(I, nt))
end

function TDSymbolic(S::ChordalSymbolic{I}, Fptr::AbstractVector{I}, nBptr::I, nt::I) where {I <: Integer}
    res  = S.res
    pnt  = S.pnt
    sep  = S.sep
    Dptr = S.Dptr
    Lptr = S.Lptr

    nf = nv(res)
    ns = ne(sep)
    nr = zero(I)
    nc = zero(I)
    ng = zero(I)
    up = one(I)

    fdsc = FVector{I}(undef, nf)
    rptr = FVector{I}(undef, nf + one(I))
    rtgt = FVector{I}(undef, nf)
    work = FVector{I}(undef, nf)
    cut  = FVector{I}(undef, nf)
    perm = FVector{I}(undef, nf)
    gbeg = FVector{I}(undef, nf)
    gend = FVector{I}(undef, nf)
    uptr = FVector{I}(undef, nf + one(I))
    uval = FVector{I}(undef, ns)
    cval = FVector{I}(undef, ns)
    cptr = FVector{I}(undef, nBptr + one(I))
    gptr = FVector{I}(undef, nBptr + one(I))
    #
    # initialize first descendants
    #
    @inbounds for f in vertices(res)
        fdsc[f] = f
    end
    #
    # find chunks and their separators, one subtree at a time
    #
    @inbounds for b in oneto(nBptr)
        #
        # fb0 ... fb1 is the front range of subtree b
        #
        fb0 = Fptr[b]
        fb1 = Fptr[b + one(I)] - one(I)

        cptr[b] = nc + one(I)
        gptr[b] = ng + one(I)
        #
        # nc0 is the chunk count before subtree b
        #
        nc0 = nc
        #
        # pf is the last front placed in a chunk
        #
        pf = fb0 - one(I)
        #
        # wn is the total weight of the subtree
        #
        wn = ((Dptr[fb1 + one(I)] - Dptr[fb0]) >> 1) + Lptr[fb1 + one(I)] - Lptr[fb0]
        #
        # wt is the target weight
        #
        wt = cld(wn, convert(I, TD_SPLIT * nt))

        for f in fb0:fb1
            #
            # df is the first descendant of front f
            #
            df = fdsc[f]
            #
            # if f is a root, then p is zero; otherwise, it
            # is the parent of f
            #
            p = pnt[f]

            if ispositive(p)
                #
                # dp is the first descendant of parent p
                #
                dp = fdsc[p] = min(fdsc[p], df)
                #
                # wp is the weight of parent p
                #
                wp = ((Dptr[p + one(I)] - Dptr[dp]) >> 1) + Lptr[p + one(I)] - Lptr[dp]
            else
                wp = typemax(I)
            end
            #
            # wf is the weight of front f
            #
            wf = ((Dptr[f + one(I)] - Dptr[df]) >> 1) + Lptr[f + one(I)] - Lptr[df]

            if wf <= wt < wp
                #
                # f is part of a chunk
                #
                if df > pf + one(I)
                    #
                    # f is preceded by a gap
                    #
                    ng += one(I)
                    gbeg[ng] = pf + one(I)
                    gend[ng] = df - one(I)
                end

                if nc > nc0 && rtgt[nr] + one(I) == df && work[nc] + wf <= wt
                    #
                    # extend the current chunk
                    #
                    work[nc] += wf
                else
                    #
                    # start a new chunk
                    #
                    nc += one(I)
                    work[nc] = wf
                    rptr[nc] = nr + one(I)
                    uptr[nc] = up
                end

                nr += one(I); rtgt[nr] = pf = f
                #
                # merge the separator of f into the separator of its chunk
                #
                cp = uptr[nc]
                sp = pointers(sep)[f]
                sq = pointers(sep)[f + one(I)] - one(I)
                ss = sq - sp + one(I)

                for q in up - one(I):-one(I):cp
                    uval[q + ss] = uval[q]
                end

                ap = cp + ss
                aq = up - one(I) + ss
                up = cp
                pv = zero(I)

                while ap <= aq || sp <= sq
                    if sp > sq || (ap <= aq && uval[ap] <= targets(sep)[sp])
                        v = uval[ap]; ap += one(I)
                    else
                        v = targets(sep)[sp]; sp += one(I)
                    end

                    if v != pv
                        uval[up] = pv = v; up += one(I)
                    end
                end
            end
        end

        if pf < fb1
            ng += one(I)
            gbeg[ng] = pf + one(I)
            gend[ng] = fb1
        end
    end

    cptr[nBptr + one(I)] = nc + one(I)
    gptr[nBptr + one(I)] = ng + one(I)

    rptr[nc + one(I)] = nr + one(I)
    uptr[nc + one(I)] = up
    #
    # sort chunks by weight (decreasing), within each subtree
    #
    @inbounds for b in oneto(nBptr)
        cstrt = cptr[b]
        cstop = cptr[b + one(I)] - one(I)

        sortperm!(view(perm, cstrt:cstop), view(work, cstrt:cstop); rev = true)

        for k in cstrt:cstop
            perm[k] += cstrt - one(I)
        end
    end
    #
    # compute relative indices
    #
    @inbounds for c in oneto(nc)
        fstrt = fdsc[rtgt[rptr[c]]]
        fstop = rtgt[rptr[c + one(I)] - one(I)]
        ustop = pointers(res)[fstop + one(I)] - one(I)
        gstrt = uptr[c]

        for f in fstrt:fstop
            g = gstrt
            k = zero(I)
            sstrt = pointers(sep)[f]
            sstop = pointers(sep)[f + one(I)] - one(I)

            for sp in sstrt:sstop
                u = targets(sep)[sp]

                if u <= ustop
                    k += one(I)
                else
                    while uval[g] < u
                        g += one(I)
                    end

                    cval[sp] = g - gstrt + one(I)
                end
            end

            cut[f] = k
        end
    end

    rts = FBipartiteGraph{I, I}(nf, nc, nr, rptr, rtgt)
    bnd = FBipartiteGraph{I, I}(pointers(res)[nf + one(I)] - one(I), nc, up - one(I), uptr, uval)
    col = FBipartiteGraph{I, I}(pointers(res)[nf + one(I)] - one(I), nf, ns, pointers(sep), cval)
    return TDSymbolic{I}(nBptr, ng, fdsc, rts, bnd, col, cut, perm, gbeg, gend, cptr, gptr)
end

struct TDWorkspace{T, I}
    D::TDSymbolic{I}
    next::Atomic{I}
    Pval::FVector{T}
    Mval::FVector{T}
end

function TDWorkspace{T}(S::ChordalSymbolic{I}, D::TDSymbolic{I}, nrhs::Integer; nt::Integer) where {T, I <: Integer}
    return TDWorkspace{T}(S, D, convert(I, nrhs), convert(I, nt))
end

function TDWorkspace{T}(S::ChordalSymbolic{I}, D::TDSymbolic{I}, nrhs::I, nt::I) where {T, I <: Integer}
    nw = min(nt, nv(D.rts))
    nf = S.nFval

    Pval = FVector{T}(undef, ne(D.bnd) * nrhs)
    Mval = FVector{T}(undef, max(nw, one(I)) * nf * nrhs)

    return TDWorkspace{T, I}(D, Atomic{I}(zero(I)), Pval, Mval)
end

struct TDBound{P, K, G}
    P::P
    cut::K
    col::G
end

function strsx!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        W::TDWorkspace{T, I};
        nt::Integer = nthreads(),
    ) where {SIDE, TRANS, UPLO, T, I}
    if B isa AbstractVector
        pool = nothing
    else
        pool = spool_mt(s, T, nt)
    end

    return strsx_mt!(s, side, trans, diag, A, B, W, pool, nt)
end

function strsx_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        W::TDWorkspace{T, I},
        pool,
        nt::Integer,
    ) where {SIDE, TRANS, UPLO, T, I}
    return strsx_mt!(s, side, trans, diag, A, B, W, pool, nt, one(I), one(I), nv(A.S.res))
end

function strsx_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        W::TDWorkspace{T, I},
        pool,
        nt::Integer,
        b::I,
        fstrt::I,
        fstop::I,
    ) where {SIDE, TRANS, UPLO, T, I}
    D = W.D

    if B isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    nFval = A.S.nFval

    nf = nv(A.S.res)
    nw = min(nt, D.cptr[b + one(I)] - D.cptr[b])

    @assert length(D.fdsc)                 >= nf
    @assert ne(D.bnd) * nrhs               <= length(W.Pval)
    @assert max(nw, one(I)) * nFval * nrhs <= length(W.Mval)

    if nw < 2
        strsx_mt!(s, side, trans, diag, A, B, W.Mval, pool, nt, fstrt, fstop)
    elseif isforward(UPLO, TRANS, SIDE)
        td_strsx_fwd!(s, side, trans, diag, A, B, W, pool, nt, nrhs, b)
    else
        td_strsx_bwd!(s, side, trans, diag, A, B, W, pool, nt, nrhs, b)
    end

    return B
end

function td_strsx_fwd!(
        s::AbstractSemiring,
        side::Val,
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractVector,
        W::TDWorkspace{T, I},
        pool::Nothing,
        nt::Integer,
        nrhs::I,
        b::I,
    ) where {T, I}
    D = W.D
    cstrt = D.cptr[b]
    cstop = D.cptr[b + one(I)] - one(I)
    nw = min(nt, cstop - cstrt + one(I))
    tasks = FVector{Task}(undef, nw - one(I))
    atomic_xchg!(W.next, cstrt - one(I))

    for w in two(I):nw
        tasks[w - 1] = @spawn td_strsx_fwd_task!(s, side, trans, diag, A, B, W, $w, nothing, nrhs, b)
    end

    td_strsx_fwd_task!(s, side, trans, diag, A, B, W, one(I), nothing, nrhs, b)

    for t in tasks
        wait(t)
    end

    for c in cstrt:cstop
        m = eltypedegree(D.bnd, c)
        o = (pointers(D.bnd)[c] - one(I)) * nrhs
        Pc = view(W.Pval, o + one(I):o + m)
        Uc = neighbors(D.bnd, c)
        sscatteradd!(s, trans, B, Pc, Uc)
    end

    for g in D.gptr[b]:D.gptr[b + one(I)] - one(I)
        strsx_mt!(s, side, trans, diag, A, B, W.Mval, pool, nt, D.gbeg[g], D.gend[g])
    end

    return B
end

function td_strsx_fwd!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractMatrix,
        W::TDWorkspace{T, I},
        pool::AbstractVector,
        nt::Integer,
        nrhs::I,
        b::I,
    ) where {SIDE, T, I}
    D = W.D
    cstrt = D.cptr[b]
    cstop = D.cptr[b + one(I)] - one(I)
    nw = min(nt, cstop - cstrt + one(I))
    tasks = FVector{Task}(undef, nw - one(I))
    atomic_xchg!(W.next, cstrt - one(I))

    for w in two(I):nw
        tasks[w - 1] = @spawn td_strsx_fwd_task!(s, side, trans, diag, A, B, W, $w, view(pool, w:w), nrhs, b)
    end

    td_strsx_fwd_task!(s, side, trans, diag, A, B, W, one(I), view(pool, one(I):one(I)), nrhs, b)

    for t in tasks
        wait(t)
    end

    for c in cstrt:cstop
        m = eltypedegree(D.bnd, c)
        o = (pointers(D.bnd)[c] - one(I)) * nrhs

        if SIDE === :L
            Pc = reshape(view(W.Pval, o + one(I):o + m * nrhs), m, nrhs)
        else
            Pc = reshape(view(W.Pval, o + one(I):o + m * nrhs), nrhs, m)
        end

        Uc = neighbors(D.bnd, c)
        sscatteradd!(s, trans, B, Pc, Uc, side)
    end

    for g in D.gptr[b]:D.gptr[b + one(I)] - one(I)
        strsx_mt!(s, side, trans, diag, A, B, W.Mval, pool, nt, D.gbeg[g], D.gend[g])
    end

    return B
end

function td_strsx_bwd!(
        s::AbstractSemiring,
        side::Val,
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractVector,
        W::TDWorkspace{T, I},
        pool::Nothing,
        nt::Integer,
        nrhs::I,
        b::I,
    ) where {T, I}
    D = W.D
    cstrt = D.cptr[b]
    cstop = D.cptr[b + one(I)] - one(I)
    nw = min(nt, cstop - cstrt + one(I))
    tasks = FVector{Task}(undef, nw - one(I))

    for g in D.gptr[b + one(I)] - one(I):-one(I):D.gptr[b]
        strsx_mt!(s, side, trans, diag, A, B, W.Mval, pool, nt, D.gbeg[g], D.gend[g])
    end

    atomic_xchg!(W.next, cstrt - one(I))

    for w in two(I):nw
        tasks[w - 1] = @spawn td_strsx_bwd_task!(s, side, trans, diag, A, B, W, $w, nothing, nrhs, b)
    end

    td_strsx_bwd_task!(s, side, trans, diag, A, B, W, one(I), nothing, nrhs, b)

    for t in tasks
        wait(t)
    end

    return B
end

function td_strsx_bwd!(
        s::AbstractSemiring,
        side::Val,
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractMatrix,
        W::TDWorkspace{T, I},
        pool::AbstractVector,
        nt::Integer,
        nrhs::I,
        b::I,
    ) where {T, I}
    D = W.D
    cstrt = D.cptr[b]
    cstop = D.cptr[b + one(I)] - one(I)
    nw = min(nt, cstop - cstrt + one(I))
    tasks = FVector{Task}(undef, nw - one(I))

    for g in D.gptr[b + one(I)] - one(I):-one(I):D.gptr[b]
        strsx_mt!(s, side, trans, diag, A, B, W.Mval, pool, nt, D.gbeg[g], D.gend[g])
    end

    atomic_xchg!(W.next, cstrt - one(I))

    for w in two(I):nw
        tasks[w - 1] = @spawn td_strsx_bwd_task!(s, side, trans, diag, A, B, W, $w, view(pool, w:w), nrhs, b)
    end

    td_strsx_bwd_task!(s, side, trans, diag, A, B, W, one(I), view(pool, one(I):one(I)), nrhs, b)

    for t in tasks
        wait(t)
    end

    return B
end

function td_strsx_fwd_task!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractVecOrMat,
        W::TDWorkspace{T, I},
        w::I,
        pool,
        nrhs::I,
        b::I,
    ) where {SIDE, T, I}
    D  = W.D
    nf = A.S.nFval
    M  = view(W.Mval, (w - one(I)) * nf * nrhs + one(I):w * nf * nrhs)
    kstop = D.cptr[b + one(I)] - one(I)
    k  = atomic_add!(W.next, one(I)) + one(I)

    @inbounds while k <= kstop
        c = D.perm[k]
        m = eltypedegree(D.bnd, c)
        o = (pointers(D.bnd)[c] - one(I)) * nrhs

        if B isa AbstractVector
            Pc = view(W.Pval, o + one(I):o + m)
        elseif SIDE === :L
            Pc = reshape(view(W.Pval, o + one(I):o + m * nrhs), m, nrhs)
        else
            Pc = reshape(view(W.Pval, o + one(I):o + m * nrhs), nrhs, m)
        end

        p = pointers(D.rts)[c]
        q = pointers(D.rts)[c + one(I)] - one(I)

        f = targets(D.rts)[p]
        g = targets(D.rts)[q]

        szerorec!(s, Pc, trans)
        bd = TDBound(Pc, D.cut, D.col)
        strsx_mt!(s, side, trans, diag, A, B, M, pool, 1, D.fdsc[f], g, bd)

        k = atomic_add!(W.next, one(I)) + one(I)
    end

    return
end

function td_strsx_bwd_task!(
        s::AbstractSemiring,
        side::Val,
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractVecOrMat,
        W::TDWorkspace{T, I},
        w::I,
        pool,
        nrhs::I,
        b::I,
    ) where {T, I}
    D = W.D
    m = A.S.nFval * nrhs
    M = view(W.Mval, (w - one(I)) * m + one(I):w * m)
    kstop = D.cptr[b + one(I)] - one(I)
    k = atomic_add!(W.next, one(I)) + one(I)

    @inbounds while k <= kstop
        c = D.perm[k]
        p = pointers(D.rts)[c]
        q = pointers(D.rts)[c + one(I)] - one(I)
        f = targets(D.rts)[p]
        g = targets(D.rts)[q]
        strsx_mt!(s, side, trans, diag, A, B, M, pool, 1, D.fdsc[f], g)
        k = atomic_add!(W.next, one(I)) + one(I)
    end

    return
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractVector{T}, M::AbstractVector{T}, fsep::AbstractVector{I}, na::I, f::I, ::Nothing) where {T, I}
    return sscatteradd!(s, trans, C, M, fsep)
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractVector{T}, M::AbstractVector{T}, fsep::AbstractVector{I}, na::I, f::I, bnd::TDBound) where {T, I}
    k = bnd.cut[f]
    sscatteradd!(s, trans, C, view(M, oneto(k)), view(fsep, oneto(k)))

    if k < na
        pc = neighbors(bnd.col, f)
        sscatteradd!(s, trans, bnd.P, view(M, k + one(I):na), view(pc, k + one(I):na))
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, M::AbstractMatrix{T}, fsep::AbstractVector{I}, na::I, f::I, side::Val, ::Nothing) where {T, I}
    return sscatteradd!(s, trans, C, M, fsep, side)
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, M::AbstractMatrix{T}, fsep::AbstractVector{I}, na::I, f::I, side::Val{SIDE}, bnd::TDBound) where {T, I, SIDE}
    k = bnd.cut[f]

    if SIDE === :L
        sscatteradd!(s, trans, C, view(M, oneto(k), :), view(fsep, oneto(k)), side)

        if k < na
            pc = neighbors(bnd.col, f)
            sscatteradd!(s, trans, bnd.P, view(M, k + one(I):na, :), view(pc, k + one(I):na), side)
        end
    else
        sscatteradd!(s, trans, C, view(M, :, oneto(k)), view(fsep, oneto(k)), side)

        if k < na
            pc = neighbors(bnd.col, f)
            sscatteradd!(s, trans, bnd.P, view(M, :, k + one(I):na), view(pc, k + one(I):na), side)
        end
    end

    return C
end

function strsx_fwd_upd_1!(s::AbstractSemiring, C::AbstractVecOrMat{T}, fsep::AbstractVector{I}, l₂₁::AbstractVector{T}, Rp::I, na::I, nrhs::I, trans::Val, side::Val, f::I, ::Nothing) where {T, I}
    return strsx_fwd_upd_1!(s, C, fsep, l₂₁, Rp, na, nrhs, trans, side)
end

function strsx_fwd_upd_1!(s::AbstractSemiring, C::AbstractVecOrMat{T}, fsep::AbstractVector{I}, l₂₁::AbstractVector{T}, Rp::I, na::I, nrhs::I, trans::Val, ::Val{SIDE}, f::I, bnd::TDBound) where {T, I, SIDE}
    k = bnd.cut[f]
    idx = view(neighbors(bnd.col, f), k + one(I):na)
    P = bnd.P

    if C isa AbstractVector
        v = C[Rp]

        @inbounds for i in oneto(k)
            if SIDE === :L
                C[fsep[i]] = smuladd(s, l₂₁[i], v, C[fsep[i]], trans, Val(:N))
            else
                C[fsep[i]] = smuladd(s, v, l₂₁[i], C[fsep[i]], Val(:N), trans)
            end
        end

        @inbounds for i in k + one(I):na
            p = idx[i - k]

            if SIDE === :L
                P[p] = smuladd(s, l₂₁[i], v, P[p], trans, Val(:N))
            else
                P[p] = smuladd(s, v, l₂₁[i], P[p], Val(:N), trans)
            end
        end
    elseif SIDE === :L
        @inbounds for c in oneto(nrhs)
            v = C[Rp, c]

            for i in oneto(k)
                C[fsep[i], c] = smuladd(s, l₂₁[i], v, C[fsep[i], c], trans, Val(:N))
            end

            for i in k + one(I):na
                p = idx[i - k]
                P[p, c] = smuladd(s, l₂₁[i], v, P[p, c], trans, Val(:N))
            end
        end
    else
        @inbounds for i in oneto(k), c in oneto(nrhs)
            C[c, fsep[i]] = smuladd(s, C[c, Rp], l₂₁[i], C[c, fsep[i]], Val(:N), trans)
        end

        @inbounds for i in k + one(I):na
            p = idx[i - k]

            for c in oneto(nrhs)
                P[c, p] = smuladd(s, C[c, Rp], l₂₁[i], P[c, p], Val(:N), trans)
            end
        end
    end

    return
end
