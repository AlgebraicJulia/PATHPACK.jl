struct ChordalSSymbolic{I}
    S::ChordalSymbolic{I}
    N::FBipartiteGraph{I, I}
    Bptr::FVector{I}
    Fptr::FVector{I}
    fcc::FVector{I}
    nBptr::I
end

function Base.size(S::ChordalSSymbolic)
    return size(S.S)
end

function Base.size(S::ChordalSSymbolic, d::Integer)
    return size(S.S, d)
end

function ncc(S::ChordalSSymbolic)
    return convert(Int, S.nBptr)
end

function components(S::ChordalSSymbolic)
    return oneto(ncc(S))
end

function ssymbolic(A::SparseMatrixCSC{<:Any, I}; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM) where {I}
    n = convert(I, size(A, 2))

    scc = sccs(A)

    nBptr = nv(scc)
    Bptr  = pointers(scc)
    invp  = targets(scc)
    index = FVector{I}(undef, n)

    @inbounds for a in oneto(n)
        index[invp[a]] = a
    end

    graph = graphpermute(BipartiteGraph(A), invp, index)
    perm, tree = scliquetree(graph, Bptr, nBptr; alg)
    S = ChordalSymbolic(tree)
    Fptr = sccfronts(S, Bptr, nBptr)
    fcc = sccmap(S, Fptr, nBptr)

    N = soffd(graph, perm, Bptr, nBptr)

    @inbounds for a in oneto(n)
        perm[a] = invp[perm[a]]
    end

    @inbounds for a in oneto(n)
        invp[perm[a]] = a
    end

    return Permutation(perm, invp), ChordalSSymbolic(S, N, Bptr, Fptr, fcc, nBptr)
end

function scliquetree(A::SparseMatrixCSC{<:Any, I}, Bptr::AbstractVector{I}, nBptr::I; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM) where {I}
    return scliquetree(BipartiteGraph(A), Bptr, nBptr; alg)
end

function scliquetree(graph::BipartiteGraph{I}, Bptr::AbstractVector{I}, nBptr::I; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM) where {I}
    n = nv(graph)

    if isone(nBptr)
        order, ctree = cliquetree(symmetric(graph, 'N'); alg)
        Rtgt = FVector{I}(order)
    else
        prnt = FVector{I}(undef, n)
        Rptr = FVector{I}(undef, n + one(I)); Rptr[one(I)] = one(I)
        Sptr = FVector{I}(undef, n + one(I)); Sptr[one(I)] = one(I)
        Stgt = I[]
        Rtgt = FVector{I}(undef, n)

        f = k = zero(I); vstrt = fstrt = one(I)

        @inbounds for c in oneto(nBptr)
            vstop = Bptr[c + one(I)]

            if vstop - vstrt == one(I)
                f = scliquetree_loop_1!(prnt, Rptr, Sptr, Rtgt, f, vstrt)
            elseif vstop - vstrt == two(I)
                f = scliquetree_loop_2!(prnt, Rptr, Sptr, Rtgt, f, vstrt)
            else
                f, k = scliquetree_loop!(prnt, Rptr, Sptr, Stgt, Rtgt, graph, f, k, vstrt, vstop, fstrt; alg)
            end

            vstrt = vstop; fstrt = f + one(I)
        end

        m = fstrt - one(I)

        res = BipartiteGraph{I, I}(n, m, n, Rptr, oneto(n))
        sep = BipartiteGraph{I, I}(n, m, k, Sptr, FVector{I}(Stgt))

        etree = Tree(m, prnt)
        stree = SupernodeTree(etree, res)
        ctree = CliqueTree(stree, sep)
    end

    return Rtgt, ctree
end

function scliquetree_loop!(
        prnt::AbstractVector{I},
        Rptr::AbstractVector{I},
        Sptr::AbstractVector{I},
        Stgt::AbstractVector{I},
        Rtgt::AbstractVector{I},
        graph::BipartiteGraph{I},
        f::I,
        k::I,
        vstrt::I,
        vstop::I,
        fstrt::I;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
    ) where {I}
    cperm, ctree = cliquetree(symmetric(subgraph(graph, vstrt, vstop - one(I)), 'N'); alg)

    cres = residuals(ctree)
    csep = separators(ctree)
    cpnt = ctree.tree.tree.tree.prnt

    @inbounds for cf in vertices(cres)
        f += one(I); cg = cpnt[cf]

        if ispositive(cg)
            prnt[f] = cg + fstrt - one(I)
        else
            prnt[f] = cg
        end

        Rptr[f + one(I)] = Rptr[f] + eltypedegree(cres, cf)
        Sptr[f + one(I)] = Sptr[f] + eltypedegree(csep, cf)

        for cv in neighbors(csep, cf)
            k += one(I); push!(Stgt, cv + vstrt - one(I))
        end
    end

    @inbounds for i in oneto(vstop - vstrt)
        Rtgt[vstrt - one(I) + i] = vstrt - one(I) + cperm[i]
    end

    return f, k
end

function scliquetree_loop_1!(
        prnt::AbstractVector{I},
        Rptr::AbstractVector{I},
        Sptr::AbstractVector{I},
        Rtgt::AbstractVector{I},
        f::I,
        vstrt::I,
    ) where {I}
    f += one(I)

    @inbounds prnt[f] = zero(I)
    @inbounds Rptr[f + one(I)] = Rptr[f] + one(I)
    @inbounds Sptr[f + one(I)] = Sptr[f]
    @inbounds Rtgt[vstrt] = vstrt

    return f
end

function scliquetree_loop_2!(
        prnt::AbstractVector{I},
        Rptr::AbstractVector{I},
        Sptr::AbstractVector{I},
        Rtgt::AbstractVector{I},
        f::I,
        vstrt::I,
    ) where {I}
    f += one(I)

    @inbounds prnt[f] = zero(I)
    @inbounds Rptr[f + one(I)] = Rptr[f] + two(I)
    @inbounds Sptr[f + one(I)] = Sptr[f]
    @inbounds Rtgt[vstrt] = vstrt
    @inbounds Rtgt[vstrt + one(I)] = vstrt + one(I)

    return f
end

function sccfronts(S::ChordalSymbolic{I}, Bptr::AbstractVector{I}, nBptr::I) where {I}
    Fptr = FVector{I}(undef, nBptr + one(I))

    @inbounds for c in oneto(nBptr)
        Fptr[c] = S.idx[Bptr[c]]
    end

    Fptr[nBptr + one(I)] = nv(S.res) + one(I)
    return Fptr
end

function sccmap(S::ChordalSymbolic{I}, Fptr::AbstractVector{I}, nBptr::I) where {I}
    fcc = FVector{I}(undef, nv(S.res))

    @inbounds for c in oneto(nBptr)
        fstrt = Fptr[c]
        fstop = Fptr[c + one(I)] - one(I)

        for f in fstrt:fstop
            fcc[f] = c
        end
    end

    return fcc
end

function soffd(graph::BipartiteGraph{I}, perm::AbstractVector{I}, Bptr::AbstractVector{I}, nBptr::I) where {I}
    n = nv(graph)

    index = FVector{I}(undef, n)
    ptr = FVector{I}(undef, n + one(I))

    @inbounds for j in oneto(n)
        index[perm[j]] = j
    end

    p = one(I)

    @inbounds for c in oneto(nBptr)
        jstrt = Bptr[c]
        jstop = Bptr[c + one(I)] - one(I)

        for j in jstrt:jstop
            ptr[j] = p

            for i in neighbors(graph, perm[j])
                if i < jstrt
                    p += one(I)
                end
            end
        end
    end

    ptr[n + one(I)] = p; m = p - one(I)

    tgt = FVector{I}(undef, m)

    @inbounds for c in oneto(nBptr)
        jstrt = Bptr[c]
        jstop = Bptr[c + one(I)] - one(I)

        for j in jstrt:jstop
            pstrt = p = ptr[j]

            for i in neighbors(graph, perm[j])
                if i < jstrt
                    tgt[p] = index[i]; p += one(I)
                end
            end

            for q in pstrt + one(I):p - one(I)
                i = tgt[q]; r = q - one(I)

                while r >= pstrt && tgt[r] > i
                    tgt[r + one(I)] = tgt[r]; r -= one(I)
                end

                tgt[r + one(I)] = i
            end
        end
    end

    return BipartiteGraph{I, I}(n, n, m, ptr, tgt)
end
