# ===== sccs =====

function sccs(A::SparseMatrixCSC)
    return sccs(BipartiteGraph(A))
end

function sccs(graph::BipartiteGraph{I}) where {I}
    n = nv(graph); np1 = n + one(I)

    low  = FVector{I}(undef, n)
    arc  = FVector{I}(undef, n)
    ptr  = FVector{I}(undef, np1)
    tgt  = FVector{I}(undef, n)

    return sccs!(low, arc, ptr, tgt, graph)
end

function sccs!(low::AbstractVector{I}, arc::AbstractVector{I}, ptr::AbstractVector{I}, tgt::AbstractVector{I}, graph::BipartiteGraph{I}) where {I}
    n = nv(graph); np1 = n + one(I)

    fill!(low, zero(I))

    r = np1; c = zero(I); i = one(I)

    @inbounds for u in vertices(graph)
        if iszero(low[u])
            r -= one(I); tgt[r] = u; l = low[u] = np1 - r
            d  = one(I); ptr[np1 - d] = r
            v  = u; p = pointers(graph)[v]; pstop = pointers(graph)[v + one(I)]

            while true
                if p < pstop
                    w = targets(graph)[p]; p += one(I); m = low[w]

                    if iszero(m)
                        low[v] = l; arc[d] = p
                        r -= one(I); tgt[r] = w; l = low[w] = np1 - r
                        d += one(I); ptr[np1 - d] = r
                        v  = w; p = pointers(graph)[v]; pstop = pointers(graph)[v + one(I)]
                    else
                        l = min(l, m)
                    end
                else
                    rstop = ptr[np1 - d]

                    if l + rstop == np1
                        c += one(I); ptr[c] = i

                        for j in r:rstop
                            w = tgt[i] = tgt[j]; low[w] = np1; i += one(I)
                        end

                        r = rstop + one(I)
                    else
                        low[v] = l
                    end

                    d -= one(I); iszero(d) && break

                    v  = tgt[ptr[np1 - d]]
                    l  = min(low[v], l)
                    p  = arc[d]; pstop = pointers(graph)[v + one(I)]
                end
            end
        end
    end

    ptr[c + one(I)] = i
    return BipartiteGraph{I, I}(n, c, n, ptr, tgt)
end

function subgraph(graph::BipartiteGraph{I}, strt::I, stop::I) where {I}
    @assert one(I) <= strt <= stop <= nv(graph)

    n = stop - strt + one(I)

    ptr = FVector{I}(undef, n + one(I))

    p = one(I)

    @inbounds for j in oneto(n)
        ptr[j] = p

        for i in neighbors(graph, strt + j - one(I))
            if strt <= i <= stop
                p += one(I)
            end
        end
    end

    ptr[n + one(I)] = p; m = p - one(I)

    tgt = FVector{I}(undef, m)

    @inbounds for j in oneto(n)
        p = ptr[j]

        for i in neighbors(graph, strt + j - one(I))
            if strt <= i <= stop
                tgt[p] = i - strt + one(I); p += one(I)
            end
        end
    end

    return BipartiteGraph{I, I}(n, n, m, ptr, tgt)
end

function szerorec!(s::AbstractSemiring, A::AbstractVecOrMat{T}, trans::Val) where {T}
    fill!(A, szero(s, T, trans))
    return A
end

function sscatteradd!(s::AbstractSemiring, C::AbstractMatrix, M::AbstractMatrix, ind::AbstractVector, ::Val{:L})
    @inbounds for j in axes(M, 2)
        for i in axes(M, 1)
            C[ind[i], j] = splus(s, C[ind[i], j], M[i, j], Val(:N))
        end
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, C::AbstractMatrix, M::AbstractMatrix, ind::AbstractVector, ::Val{:R})
    @inbounds for j in axes(M, 2)
        indj = ind[j]

        for i in axes(M, 1)
            C[i, indj] = splus(s, C[i, indj], M[i, j], Val(:N))
        end
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, C::AbstractVector, M::AbstractVector, ind::AbstractVector)
    @inbounds for i in axes(M, 1)
        C[ind[i]] = splus(s, C[ind[i]], M[i], Val(:N))
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, M::AbstractMatrix, ind::AbstractVector, ::Val{:L})
    @inbounds for j in axes(M, 2)
        for i in axes(M, 1)
            C[ind[i], j] = splus(s, C[ind[i], j], M[i, j], trans)
        end
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, M::AbstractMatrix, ind::AbstractVector, ::Val{:R})
    @inbounds for j in axes(M, 2)
        indj = ind[j]

        for i in axes(M, 1)
            C[i, indj] = splus(s, C[i, indj], M[i, j], trans)
        end
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractVector, M::AbstractVector, ind::AbstractVector)
    @inbounds for i in axes(M, 1)
        C[ind[i]] = splus(s, C[ind[i]], M[i], trans)
    end

    return C
end

function permuterows!(A::AbstractVecOrMat, work::AbstractVector, perm::AbstractVector)
    m = size(A, 1)
    n = size(A, 2)
    k = min(8, n)

    B = reshape(view(work, oneto(m * k)), m, k)

    @inbounds for jstrt in 1:k:n
        jsize = min(jstrt + k - 1, n) - jstrt + 1

        for j in 1:jsize
            for i in 1:m
                B[i, j] = A[i, jstrt + j - 1]
            end
        end

        for j in 1:jsize
            for i in 1:m
                A[perm[i], jstrt + j - 1] = B[i, j]
            end
        end
    end

    return A
end

function permutecols!(A::AbstractVecOrMat, work::AbstractVector, perm::AbstractVector)
    m = size(A, 1)
    n = size(A, 2)
    k = min(8, m)

    B = reshape(view(work, oneto(k * n)), k, n)

    @inbounds for istrt in 1:k:m
        isize = min(istrt + k - 1, m) - istrt + 1

        for j in 1:n
            for i in 1:isize
                B[i, j] = A[istrt + i - 1, j]
            end
        end

        for j in 1:n
            for i in 1:isize
                A[istrt + i - 1, perm[j]] = B[i, j]
            end
        end
    end

    return A
end

# ===== permuterowscols! =====

function permuterowscols!(
        A::AbstractMatrix{T},
        rperm::AbstractVector{I},
        cperm::AbstractVector{I};
        nt::Integer = nthreads(),
    ) where {T, I}
    if isone(nt)
        permuterowscols_st!(A, rperm, cperm)
    else
        permuterowscols_mt!(A, rperm, cperm, nt)
    end

    return A
end

function permuterowscols_st!(
        A::AbstractMatrix{T},
        rperm::AbstractVector{I},
        cperm::AbstractVector{I},
    ) where {T, I}
    m = size(A, 1)
    n = size(A, 2)

    cur = FVector{T}(undef, m)
    nxt = FVector{T}(undef, m)
    mark = FVector{Bool}(undef, n)
    fill!(mark, false)

    @inbounds for jstrt in oneto(n)
        if !mark[jstrt]
            for i in oneto(m)
                cur[i] = A[i, jstrt]
            end

            j = jstrt

            while true
                mark[j] = true
                k = cperm[j]

                if k != jstrt
                    for i in oneto(m)
                        nxt[i] = A[i, k]
                    end
                end

                for i in oneto(m)
                    A[rperm[i], k] = cur[i]
                end

                k == jstrt && break
                cur, nxt = nxt, cur
                j = k
            end
        end
    end

    return A
end

function permuterowscols_mt!(
        A::AbstractMatrix{T},
        rperm::AbstractVector{I},
        cperm::AbstractVector{I},
        nt::Integer,
    ) where {T, I}
    m = size(A, 1)
    n = size(A, 2)

    ctgt = FVector{I}(undef, n)
    cptr = FVector{I}(undef, n + 1)
    mark = FVector{Bool}(undef, n)
    fill!(mark, false)

    q = zero(I)
    ncyc = zero(I)

    @inbounds for j in oneto(n)
        if !mark[j]
            ncyc += one(I)
            cptr[ncyc] = q + one(I)
            k = convert(I, j)

            while !mark[k]
                mark[k] = true
                q += one(I)
                ctgt[q] = k
                k = cperm[k]
            end
        end
    end

    cptr[ncyc + one(I)] = n + one(I)
    w = max(1, cld(n, 4nt))

    tstrt = I[]
    tstop = I[]
    tpred = I[]

    a = one(I)

    @inbounds for c in oneto(ncyc)
        cs = cptr[c]
        ce = cptr[c + one(I)] - one(I)

        if ce - cs + 1 > w
            if a < cs
                push!(tstrt, a)
                push!(tstop, cs - one(I))
                push!(tpred, zero(I))
            end

            for s in cs:w:ce
                if s == cs
                    pred = ce
                else
                    pred = s - one(I)
                end

                push!(tstrt, s)
                push!(tstop, min(s + w - 1, ce))
                push!(tpred, pred)
            end

            a = ce + one(I)
        elseif ce - a + 1 >= w
            push!(tstrt, a)
            push!(tstop, ce)
            push!(tpred, zero(I))
            a = ce + one(I)
        end
    end

    if a <= n
        push!(tstrt, a)
        push!(tstop, n)
        push!(tpred, zero(I))
    end

    ntask = length(tstrt)
    save = FVector{FVector{T}}(undef, ntask)

    @threads for t in 1:ntask
        if ispositive(tpred[t])
            k = ctgt[tpred[t]]
            col = FVector{T}(undef, m)

            for i in oneto(m)
                col[i] = A[i, k]
            end

            save[t] = col
        end
    end

    @threads for t in 1:ntask
        if ispositive(tpred[t])
            permuterowscols_seg!(A, rperm, ctgt, tstrt[t], tstop[t], save[t])
        else
            permuterowscols_grp!(A, rperm, cperm, ctgt, tstrt[t], tstop[t])
        end
    end

    return A
end

function permuterowscols_seg!(
        A::AbstractMatrix{T},
        rperm::AbstractVector{I},
        ctgt::AbstractVector{I},
        strt::I,
        stop::I,
        cur::FVector{T},
    ) where {T, I}
    m = size(A, 1)
    nxt = FVector{T}(undef, m)

    @inbounds for p in strt:stop
        k = ctgt[p]

        if p < stop
            for i in oneto(m)
                nxt[i] = A[i, k]
            end
        end

        for i in oneto(m)
            A[rperm[i], k] = cur[i]
        end

        cur, nxt = nxt, cur
    end

    return A
end

function permuterowscols_grp!(
        A::AbstractMatrix{T},
        rperm::AbstractVector{I},
        cperm::AbstractVector{I},
        ctgt::AbstractVector{I},
        strt::I,
        stop::I,
    ) where {T, I}
    m = size(A, 1)
    cur = FVector{T}(undef, m)
    nxt = FVector{T}(undef, m)
    cs = strt

    @inbounds while cs <= stop
        ce = cs

        while cperm[ctgt[ce]] != ctgt[cs]
            ce += one(I)
        end

        for i in oneto(m)
            cur[i] = A[i, ctgt[cs]]
        end

        for p in cs:ce
            if p < ce
                k = ctgt[p + one(I)]
            else
                k = ctgt[cs]
            end

            if p < ce
                for i in oneto(m)
                    nxt[i] = A[i, k]
                end
            end

            for i in oneto(m)
                A[rperm[i], k] = cur[i]
            end

            cur, nxt = nxt, cur
        end

        cs = ce + one(I)
    end

    return A
end

function intriangle(::Val{:L}, i, j)
    return i >= j
end

function intriangle(::Val{:U}, i, j)
    return i <= j
end

# ===== siszero =====

function siszero(s::AbstractSemiring, trans::Val, C::AbstractVector{T}, jstrt::I, jstop::I) where {T, I}
    z = szero(s, T, trans)

    @inbounds for j in jstrt:jstop
        if !isequal(C[j], z)
            return false
        end
    end

    return true
end
