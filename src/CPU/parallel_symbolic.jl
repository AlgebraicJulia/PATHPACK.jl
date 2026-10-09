#
# The symbolic phase of ssymbolic on several threads. Each function here returns exactly what its serial
# counterpart in CliqueTrees returns (the same arrays, entry for entry); only the work is split:
#
#   psymmetric_pattern    symmetric_pattern (chordal_ssymbolic.jl)
#   relabelsorted!        the graph relabelled by the order, each list sorted: sympermute!_impl!
#                         (Forward) and reverse!_impl! read from it
#   relabelreverse!       sympermute!_impl!(…, Reverse) of the transpose
#   pcliquetree           cliquetree(graph; alg, snd = Maximal())
#   pchordalsymbolic      ChordalSymbolic(tree)
#
# Below PAR_MIN_ENTRIES entries each is its serial counterpart (pcliquetree: cliquetree's own steps).
#
# The relabelled graphs are built list by list (each list from one list of the input, so threads fill
# disjoint ranges and no thread needs a count per list), in the order in which the serial versions
# fill them. The separators of the clique tree are computed per front from the fronts' children:
# disjoint subtrees go to different threads, and the fronts above them are done last, in order. The
# relative indices of ChordalSymbolic are independent per front.
#
const CTR = CliqueTrees

# below this many entries the serial versions are used (waking the threads costs more)
const PAR_MIN_ENTRIES = Ref(1 << 17)

usethreads(m::Integer) = nthreads() > 1 && m >= PAR_MIN_ENTRIES[]

# the (lo, hi) ranges of `nchunk` near-equal parts of 1:n
@inline chunkrange(n::Integer, nchunk::Integer, t::Integer) = (cld((t - 1) * n, nchunk) + 1):cld(t * n, nchunk)

#
# chunks of 1:n holding near-equal numbers of entries of the lists of a graph (pointers ptr)
#
function chunkbounds(ptr::AbstractVector{E}, n::Integer, nchunk::Integer) where {E}
    bnd = Vector{Int}(undef, nchunk + 1); bnd[1] = 1; m = Int(ptr[n + 1]) - 1; j = 1

    @inbounds for t in 1:nchunk - 1
        goal = cld(t * m, nchunk) + 1

        while j <= n && ptr[j] < goal
            j += 1
        end

        bnd[t + 1] = j
    end

    bnd[nchunk + 1] = n + 1
    return bnd
end

#
# symmetric_pattern(graph) on several threads (chordal_ssymbolic.jl): every list strictly increasing,
# without its own vertex, and each entry i of list j has j in list i (binary search: the lists are
# sorted). The same answer as the serial check, whose one pointer per list checks the same two things.
#
function psymmetric_pattern(graph::AbstractGraph{I}) where {I}
    n = nv(graph); ptr = pointers(graph); tgt = targets(graph); m = Int(ptr[n + one(I)]) - 1
    usethreads(m) || return symmetric_pattern(graph)
    T = nthreads(); bnd = chunkbounds(ptr, n, T); ok = Threads.Atomic{Bool}(true)

    check! = function (lo, hi)
        @inbounds for j in lo:hi
            p0 = ptr[j]; p1 = ptr[j + one(I)] - one(I); prev = zero(I)

            for p in p0:p1
                i = tgt[p]

                if i == j || i <= prev || !inlist(tgt, ptr[i], ptr[i + one(I)] - one(I), I(j))
                    ok[] = false; return
                end

                prev = i
            end

            (j & 1023 == 0 && !ok[]) && return
        end
    end

    @threads for t in 1:T
        check!(bnd[t], bnd[t + 1] - 1)
    end

    return ok[]
end

# whether v is in the sorted tgt[lo:hi]
@inline function inlist(tgt::AbstractVector, lo::Integer, hi::Integer, v)
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1; w = tgt[mid]
        w == v && return true
        w < v ? (lo = mid + one(mid)) : (hi = mid - one(mid))
    end

    return false
end

# ===== the graph relabelled =====

#
# H = the graph relabelled by the order (H[index[v]] = index[N(v)]), each list sorted, and split[x] =
# the number of entries of H[x] below x. From it, the three graphs of cliquetree (CliqueTrees:
# supernode_trees.jl, clique_trees.jl) without scattering: upper = sympermute(graph, index, Forward)
# has the entries of H below the diagonal (in another order within a list, which neither the
# elimination tree nor the transpose depends on); lower = reverse(upper) is the entries above it, in
# increasing order; and sympermute(lower, elmindex, Reverse) is relabelreverse!'s. Each list is
# independent of the others, so threads fill disjoint ranges.
#
function relabelsorted!(Hptr::AbstractVector{E}, Htgt::AbstractVector{V}, split::AbstractVector{V}, graph::AbstractGraph{V}, order::AbstractVector, index::AbstractVector, par::Bool = true) where {V, E}
    n = nv(graph); gptr = pointers(graph); gtgt = targets(graph)
    @inbounds Hptr[1] = acc = one(E)

    @inbounds for x in oneto(n)
        v = order[x]
        Hptr[x + one(V)] = acc += gptr[v + one(V)] - gptr[v]
    end

    fillH! = function (lo, hi)
        dmax = 0

        @inbounds for x in lo:hi
            dmax = max(dmax, Int(Hptr[x + one(V)] - Hptr[x]))
        end

        scratch = dmax > LONGLIST ? Vector{V}(undef, dmax) : nothing

        @inbounds for x in lo:hi
            v = order[x]; o = Hptr[x]; d = Hptr[x + one(V)] - o; g = gptr[v] - o

            for q in o:(o + d - one(E))
                Htgt[q] = index[gtgt[q + g]]
            end

            if d > LONGLIST
                sort!(view(Htgt, o:(o + d - one(E))); scratch)
            else
                sortlist!(Htgt, o, o + d - one(E))
            end

            b = zero(V)

            for q in o:(o + d - one(E))
                Htgt[q] < x ? (b += one(V)) : break
            end

            split[x] = b
        end
    end

    m = Int(Hptr[n + one(V)]) - 1

    if par && nthreads() > 1
        T = nthreads(); bnd = chunkbounds(Hptr, n, T)

        @threads for t in 1:T
            fillH!(bnd[t], bnd[t + 1] - 1)
        end
    else
        fillH!(1, n)
    end

    return BipartiteGraph(n, n, m, Hptr, Htgt)
end

# sort tgt[lo:hi] in place (distinct values; no allocation): Shell sort with Ciura's gaps (extended
# by ×2.25) for long lists, ending (as short lists do) with an insertion sort
const SHELLGAPS = (1035711, 460316, 204585, 90927, 40412, 17961, 7983, 3548, 1577, 701, 301, 132, 57, 23, 10, 4)
const LONGLIST = 1024                               # (longer lists: Base's sort, with a scratch buffer)

@inline function sortlist!(tgt::AbstractVector, lo::Integer, hi::Integer)
    len = hi - lo + 1

    if len > 64
        for gap in SHELLGAPS
            gap < len || continue

            @inbounds for q in (lo + gap):hi
                v = tgt[q]; r = q - gap

                while r >= lo && tgt[r] > v
                    tgt[r + gap] = tgt[r]; r -= gap
                end

                tgt[r + gap] = v
            end
        end
    end

    @inbounds for q in (lo + 1):hi
        v = tgt[q]; r = q - 1

        while r >= lo && tgt[r] > v
            tgt[r + 1] = tgt[r]; r -= 1
        end

        tgt[r + 1] = v
    end

    return tgt
end

#
# lower = reverse(sympermute(graph, index, Forward)): the entries of each list of H above the diagonal,
# in increasing order, copied out (the lists are independent: threads copy disjoint ranges)
#
function suffixes!(pointer::AbstractVector{E}, target::AbstractVector{V}, H::BipartiteGraph{V}, split::AbstractVector{V}, par::Bool = true) where {V, E}
    n = nv(H); Hptr = pointers(H); Htgt = targets(H)
    @inbounds pointer[1] = acc = one(E)

    @inbounds for x in oneto(n)
        pointer[x + one(V)] = acc += (Hptr[x + one(V)] - Hptr[x]) - split[x]
    end

    m = Int(acc) - 1

    copyrange! = function (lo, hi)
        @inbounds for x in lo:hi
            o = pointer[x]; g = Hptr[x] + split[x] - o

            for q in o:(pointer[x + one(V)] - one(E))
                target[q] = Htgt[q + g]
            end
        end
    end

    if par && nthreads() > 1
        T = nthreads(); bnd = chunkbounds(pointer, n, T)

        @threads for t in 1:T
            copyrange!(bnd[t], bnd[t + 1] - 1)
        end
    else
        copyrange!(1, n)
    end

    return BipartiteGraph(n, n, m, pointer, target)
end

#
# etree_impl!(tree, ancestor, upper) (CliqueTrees: parent.jl) with upper = the entries of H below the
# diagonal (Liu's algorithm with path compression; the tree does not depend on the order in which a
# row's entries are visited)
#
function etree_prefix!(tree::Parent{V}, ancestor::AbstractVector{V}, H::BipartiteGraph{V}, split::AbstractVector{V}) where {V}
    n = nv(H); Hptr = pointers(H); Htgt = targets(H); parent = tree.prnt

    @inbounds for i in oneto(n)
        parent[i] = zero(V)
        ancestor[i] = zero(V)

        for q in Hptr[i]:(Hptr[i] + split[i] - one(V))
            r = Htgt[q]

            while !iszero(ancestor[r]) && ancestor[r] != i
                t = ancestor[r]
                ancestor[r] = i
                r = t
            end

            if iszero(ancestor[r])
                ancestor[r] = i
                parent[r] = i
            end
        end
    end

    return tree
end

#
# sympermute!_impl!(pointer, target, lower, elmindex, Reverse) with lower = the entries of H above the
# diagonal: list c (c = elmindex[x]) holds elmindex[y] for the neighbors y of x in H whose label is
# greater than c, in increasing order of y (the order in which the serial version, visiting the lists
# of lower in order, appends them). Each list of H is read once, list x giving list elmindex[x].
#
function relabelreverse!(pointer::AbstractVector{E}, target::AbstractVector{V}, H::BipartiteGraph{V}, elmindex::AbstractVector{V}, par::Bool = true) where {V, E}
    n = nv(H); Hptr = pointers(H); Htgt = targets(H)
    T = par ? nthreads() : 1
    bnd = isone(T) ? [1, n + 1] : chunkbounds(Hptr, n, T)   # (each list of H once: x in order, c = elmindex[x])

    count! = function (lo, hi)
        @inbounds for x in lo:hi
            c = elmindex[x]; k = zero(E)

            for q in Hptr[x]:(Hptr[x + one(V)] - one(E))
                elmindex[Htgt[q]] > c && (k += one(E))
            end

            pointer[c + one(V)] = k
        end
    end

    fillrev! = function (lo, hi)
        @inbounds for x in lo:hi
            c = elmindex[x]; p = pointer[c]

            for q in Hptr[x]:(Hptr[x + one(V)] - one(E))
                y = elmindex[Htgt[q]]
                y > c && (target[p] = y; p += one(E))
            end
        end
    end

    if isone(T)
        count!(1, n)
    else
        @threads for t in 1:T
            count!(bnd[t], bnd[t + 1] - 1)
        end
    end

    @inbounds pointer[1] = acc = one(E)            # (not p: the closures' p is their own)

    @inbounds for c in oneto(n)
        pointer[c + one(V)] = acc += pointer[c + one(V)]
    end

    if isone(T)
        fillrev!(1, n)
    else
        @threads for t in 1:T
            fillrev!(bnd[t], bnd[t + 1] - 1)
        end
    end

    return BipartiteGraph(n, n, Int(pointer[n + one(V)]) - 1, pointer, target)
end

# ===== subtrees for threads =====

#
# Split a postordered forest (parent pnt, 0 at roots; work[j] ≥ 0 per vertex) among T threads: returns
# (sub, own): disjoint subtrees, each a range lo:hi of the postorder given by its root hi and size, and
# own[t] the subtrees of thread t; every vertex outside them (the "top") is done afterwards in order.
# Subtrees are taken top down: a subtree with more than total / (4T) of the work is opened (its root
# goes to the top, its children become candidates); the rest are dealt to the least loaded thread,
# largest first.
#
function splitforest(pnt::AbstractVector{I}, work::AbstractVector{<:Real}, h::Integer, T::Integer) where {I}
    sub = Vector{Float64}(undef, h)                 # work of the subtree rooted at j
    sz = Vector{Int}(undef, h)                      # its number of vertices
    @inbounds for j in 1:h
        sub[j] = work[j]; sz[j] = 1
    end

    # children lists (first child / next sibling) and subtree sums (postorder: children first)
    head = zeros(Int, h + 1); next = Vector{Int}(undef, h)

    @inbounds for j in h:-1:1
        p = Int(pnt[j]); p = ispositive(p) ? p : h + 1
        next[j] = head[p]; head[p] = j
    end

    @inbounds for j in 1:h
        p = Int(pnt[j])

        if ispositive(p)
            sub[p] += sub[j]; sz[p] += sz[j]
        end
    end

    total = 0.0

    @inbounds for j in 1:h
        ispositive(Int(pnt[j])) || (total += sub[j])
    end

    limit = total / (4T)
    cand = Int[]; k = head[h + 1]

    while !iszero(k)
        push!(cand, k); k = next[k]
    end

    leaves = Int[]

    while !isempty(cand)
        j = pop!(cand)

        if sub[j] > limit && sz[j] > 1
            k = head[j]

            while !iszero(k)
                push!(cand, k); k = next[k]
            end
        else
            push!(leaves, j)
        end
    end

    sort!(leaves; by = j -> -sub[j])
    load = zeros(T); own = [Int[] for _ in 1:T]

    for j in leaves
        t = argmin(load); push!(own[t], j); load[t] += sub[j]
    end

    intop = trues(h)

    for j in leaves, v in (j - sz[j] + 1):j
        intop[v] = false
    end

    for t in 1:T
        sort!(own[t])
    end

    return own, sz, intop
end

# ===== the clique tree =====

#
# cliquetree(graph, alg, Maximal()) (CliqueTrees: clique_trees.jl, supernode_trees.jl) for a symmetric
# graph without loops, with the relabelled graphs and the separators as above: the same order and
# clique tree. (The elimination tree, the column counts, the supernodes and the postorder are
# CliqueTrees' own, serial.)
#
function pcliquetree(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {V}
    E = CTR.etype(graph); n = nv(graph); m = CTR.half(CTR.de(graph))
    nn = n + one(V); nnn = nn + one(V)
    weights = CTR.Ones{V}(n)

    target1 = FVector{V}(undef, max(m, nn))
    target3 = FVector{V}(undef, n)

    pointer1 = FVector{E}(undef, nn)
    pointer3 = FVector{V}(undef, nnn)

    colcount = FVector{V}(undef, n)
    elmorder = FVector{V}(undef, n)
    elmindex = FVector{V}(undef, n)
    sndptr = FVector{V}(undef, nn)
    sepptr = FVector{E}(undef, nn)
    new = FVector{V}(undef, n)
    parent = FVector{V}(undef, n)
    elmtree = Parent{V}(n)

    order, index = CTR.permutation(weights, graph, alg)

    if !usethreads(CTR.de(graph))
        # (small: the serial algorithm, as cliquetree)
        sndtree, upper, lower = CTR.supernodetree_impl!(target1, FVector{V}(undef, m),
            target3, pointer1, FVector{E}(undef, nn), pointer3, colcount,
            elmorder, elmindex, sndptr, sepptr, new, parent, elmtree, graph,
            order, index, CTR.Maximal())

        h = last(sndtree.tree)
        k = sepptr[h + one(V)] - one(E)
        separator = BipartiteGraph(n, h, k, sepptr, FVector{V}(undef, k))
        return order, CTR.cliquetree_impl!(elmindex, upper, lower, separator, sndtree)
    end

    # supernodetree_impl!: the elimination tree and the column counts read the graph relabelled (H,
    # each list sorted): the etree skips the entries above the diagonal, and the counts read the ones
    # below it (the transpose of upper; split[x] = the number of entries of H[x] below x)
    Hptr = FVector{E}(undef, nn); Htgt = FVector{V}(undef, CTR.twice(m)); split = FVector{V}(undef, n)
    H = relabelsorted!(Hptr, Htgt, split, graph, order, index)
    etree_prefix!(elmtree, elmorder, H, split)
    lower = suffixes!(FVector{E}(undef, nn), FVector{V}(undef, m), H, split)

    CTR.supcnt_impl!(colcount, new, parent, index, elmorder,
        elmindex, sndptr, CTR.UnionFind(n, target1, pointer3, target3),
        weights, lower, elmtree)

    ancestor = index

    tree = CTR.stree_impl!(new, parent, ancestor, elmorder,
        CTR.tree_impl!(pointer3, target3, elmtree),
        colcount, CTR.Maximal())

    CTR.postorder!_impl!(target1, sndptr, elmindex, elmorder, tree)

    @inbounds for i in tree
        elmorder[elmindex[i]] = i
    end

    @inbounds sndptr[begin] = one(V)
    @inbounds sepptr[begin] = one(E)

    @inbounds for i in tree
        ii = i + one(V); j = elmorder[i]
        u = new[j]
        p = elmindex[u] = sndptr[i]

        for v in CTR.ancestorindices(elmtree, u)
            v == ancestor[j] && break
            elmindex[v] = p += one(V)
        end

        sepptr[ii] = sepptr[i] + convert(E, sndptr[i] + colcount[u] - p) - one(E)
        sndptr[ii] = p + one(V)
    end

    @inbounds for v in oneto(n)
        elmorder[v] = order[v]
    end

    @inbounds for v in oneto(n)
        order[elmindex[v]] = elmorder[v]
    end

    h = last(tree)
    residual = BipartiteGraph(n, h, n, sndptr, oneto(n))
    sndtree = SupernodeTree(CTR.tree_impl!(pointer3, target3, tree), residual)

    # cliquetree_impl!
    k = sepptr[h + one(V)] - one(E)
    septgt = FVector{V}(undef, k)
    separator = BipartiteGraph(n, h, k, sepptr, septgt)
    lower2 = relabelreverse!(pointer1, target1, H, elmindex)
    pseparators!(separator, lower2, sndtree)
    return order, CliqueTree(sndtree, separator)
end

#
# The separators of cliquetree_impl!: separator j is the neighbors (in `lower`) of the first vertex of
# residual j beyond the residual, merged with the separators of j's children beyond it. Threads take
# disjoint subtrees; the fronts above them are done last, in order.
#
function pseparators!(separator::BipartiteGraph{V, E}, lower::BipartiteGraph{V, E}, sndtree::SupernodeTree{V}, par::Bool = true) where {V, E}
    h = last(sndtree.tree); n = nv(lower)
    residual = residuals(sndtree)
    sptr = pointers(separator)
    T = nthreads()

    maxsep = zero(E)

    @inbounds for j in oneto(h)
        maxsep = max(maxsep, sptr[j + one(V)] - sptr[j])
    end

    if !par || isone(T)
        cache = FVector{V}(undef, maxsep + one(E))

        for j in oneto(h)
            separator_front!(separator, lower, sndtree, residual, cache, j)
        end

        return separator
    end

    work = Vector{Float64}(undef, h)

    @inbounds for j in oneto(h)
        work[j] = 1.0 + (sptr[j + one(V)] - sptr[j])
    end

    pnt = sndtree.tree.tree.prnt
    own, sz, intop = splitforest(pnt, work, h, T)
    caches = [FVector{V}(undef, maxsep + one(E)) for _ in 1:T]

    # (each task its own cache: a name assigned outside the loop too would be shared by the tasks)
    @threads for t in 1:T
        local tcache = caches[t]

        for r in own[t], j in (r - sz[r] + 1):r
            separator_front!(separator, lower, sndtree, residual, tcache, j)
        end
    end

    topcache = caches[1]

    for j in oneto(h)
        intop[j] && separator_front!(separator, lower, sndtree, residual, topcache, j)
    end

    return separator
end

# one iteration of cliquetree_impl!'s loop (CliqueTrees: clique_trees.jl)
function separator_front!(separator, lower, sndtree, residual, cache, j::V) where {V}
    E = eltype(pointers(separator))

    @inbounds begin
        pstrt = pointers(separator)[j]
        pstop = pointers(separator)[j + one(V)]

        qstrt = pointers(residual)[j]
        qstop = pointers(residual)[j + one(V)]

        rstrt = pointers(lower)[qstrt]
        rstop = pointers(lower)[qstrt + one(V)]

        p = pstrt; r = rstrt

        while r < rstop && targets(lower)[r] < qstop
            r += one(E)
        end

        while r < rstop
            targets(separator)[p] = targets(lower)[r]; p += one(E)
            r += one(E)
        end

        p1 = p

        for i in CTR.childindices(sndtree, j)
            pstop1 = p1
            pstrt1 = pstrt

            pstop2 = pointers(separator)[i + one(V)]
            pstrt2 = pointers(separator)[i]

            p1 = pstrt1; p2 = pstrt2; t = one(V)

            while p2 < pstop2 && targets(separator)[p2] < qstop
                p2 += one(E)
            end

            while p1 < pstop1 && p2 < pstop2
                v1 = targets(separator)[p1]
                v2 = targets(separator)[p2]

                if v1 == v2
                    cache[t] = v1
                    p1 += one(E)
                    p2 += one(E)
                elseif v1 < v2
                    cache[t] = v1
                    p1 += one(E)
                else
                    cache[t] = v2
                    p2 += one(E)
                end

                t += one(V)
            end

            while p1 < pstop1
                cache[t] = targets(separator)[p1]
                p1 += one(E)
                t += one(V)
            end

            while p2 < pstop2
                cache[t] = targets(separator)[p2]
                p2 += one(E)
                t += one(V)
            end

            p1 = pstrt; tstop = t

            for t in oneto(tstop - one(V))
                targets(separator)[p1] = cache[t]
                p1 += one(E)
            end
        end
    end

    return
end

# ===== ChordalSymbolic =====

#
# ChordalSymbolic(tree) (Multifrontal: chordal_symbolic.jl) with the relative indices on several threads
#
function pchordalsymbolic(ctree::CliqueTree{I, I}) where {I}
    res = residuals(ctree); sep = separators(ctree)
    usethreads(ne(sep)) || return ChordalSymbolic(ctree)
    tree = Tree(ctree)
    chd = tree.graph
    pnt = tree.tree.prnt
    h = nv(res)

    reltgt = FVector{I}(undef, ne(sep))
    rel = BipartiteGraph(nov(sep), nv(sep), ne(sep), pointers(sep), reltgt)

    nMptr = jMptr = one(I)
    nMval = jMval = zero(I)
    nNval = jNval = zero(I)
    nFval = zero(I)
    Dp = zero(I)
    Lp = zero(I)

    idx = FVector{I}(undef, nov(res))
    Dptr = FVector{I}(undef, h + one(I))
    Lptr = FVector{I}(undef, h + one(I))

    # the counts (serial: O(fronts))
    @inbounds for j in oneto(h)
        Dptr[j] = Dp + one(I)
        Lptr[j] = Lp + one(I)

        nn = eltypedegree(res, j)
        na = eltypedegree(sep, j)
        nj = nn + na

        for i in neighbors(chd, j)
            ma = eltypedegree(sep, i)

            jMptr -= one(I)
            jMval -= ma * ma
            jNval -= ma
        end

        if ispositive(na)
            jMptr += one(I)
            jMval += na * na
            jNval += na

            nMptr = max(nMptr, jMptr)
            nMval = max(nMval, jMval)
            nNval = max(nNval, jNval)
        end

        nFval = max(nFval, nj)
        Dp += nn * nn
        Lp += nn * na
    end

    @inbounds Dptr[h + one(I)] = Dp + one(I)
    @inbounds Lptr[h + one(I)] = Lp + one(I)

    # the relative indices of each front's children, fronts split among threads by blocks of equal work:
    # the position in front j (its residual, a range of vertices, then its separator) of each vertex
    # of a child's separator, which is sorted, as j's separator is: a walk along j's separator, or a
    # binary search when it is much longer than the child's
    rel! = function (lo, hi)
        @inbounds for j in lo:hi
            r0 = pointers(res)[j]; r1 = pointers(res)[j + one(I)]; nn = r1 - r0
            s0 = pointers(sep)[j]; s1 = pointers(sep)[j + one(I)]

            for w in r0:(r1 - one(I))
                idx[w] = j
            end

            for i in neighbors(chd, j)
                p0 = pointers(sep)[i]; p1 = pointers(sep)[i + one(I)]
                q = s0; long = (s1 - s0) > 16 * (p1 - p0)

                for p in p0:(p1 - one(I))
                    v = targets(sep)[p]

                    if v < r1
                        reltgt[p] = v - r0 + one(I)
                    else
                        if long
                            q = searchsortedfirst(targets(sep), v, q, s1 - one(I), Base.Order.Forward)
                        else
                            while targets(sep)[q] < v
                                q += one(I)
                            end
                        end

                        reltgt[p] = nn + (q - s0) + one(I)
                    end
                end
            end
        end
    end

    T = nthreads(); bnd = chunkbounds(pointers(sep), h, T)

    @threads for t in 1:T
        rel!(bnd[t], bnd[t + 1] - 1)
    end

    return ChordalSymbolic(res, sep, rel, chd, pnt, idx, Dptr, Lptr, nMptr, nMval, nNval, nFval)
end
