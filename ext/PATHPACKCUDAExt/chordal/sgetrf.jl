# ===== hybrid sparse factorization =====

struct HybridTimes
    cpu::Float64        # bottom fronts on the CPU
    upload::Float64     # factor arrays and boundary updates to the GPU
    gpu::Float64        # top fronts on the GPU (wall time, synchronized)
    host::Float64       # host time spent issuing the GPU work (or launching the graph)
    ntop::Int
    topwork::Float64    # fraction of the multiply-adds done on the GPU
end

#
# Everything that depends only on the sparsity pattern: which fronts go to
# the GPU (the top: fronts with nn + na ≥ `large` and all their ancestors),
# the CPU and GPU workspaces, the persistent device buffers, and the list of
# GPU front tasks. With `graph = true` the GPU work is captured as a CUDA
# graph after the first factorization, and replayed by later ones (new
# weights, same pattern), which removes the host cost of the ~50 small
# launches per front.
#
#
# One CPU thread's share of the bottom forest: whole subtrees (contiguous
# postorder ranges), its own update stack, and where its boundary updates
# (left on its stack) go in the device buffer Mb.
#
struct BottomWorker{T, I}
    ranges::Vector{UnitRange{Int}}
    Mptr::MF.FVector{I}
    Mval::MF.FVector{T}
    Fval::MF.FVector{T}
    pool::Any
    nbndval::Int
    base::Int
end

mutable struct FactorPlan{Sem <: AbstractSemiring, T, I}
    F::ChordalSLU{Sem, T, I}
    istop::BitVector
    topwork::Float64
    nt::Int
    # CPU, bottom fronts: independent subtrees spread over threads
    workers::Vector{BottomWorker{T, I}}
    nbndval::Int                    # total length of the boundary updates
    # device buffers
    LD::CuVector{T}
    LL::CuVector{T}
    UD::CuVector{T}
    UL::CuVector{T}
    Mb::CuVector{T}
    Mg::CuVector{T}
    Fg::CuVector{T}
    relptr::CuVector{I}
    reltgt::CuVector{I}
    # GPU, top fronts in postorder: (n₁, n₂, Dp, Lp, out, kids), kids = [(from_gpu, offset, na, rel pointer)]
    tasks::Vector{Tuple{Int, Int, Int, Int, Int, Vector{Tuple{Bool, Int, Int, Int}}}}
    # concurrency: with nstreams > 1 every top front has its own update slot,
    # and the tasks of each level run on nstreams streams (one graph branch each)
    levels::Vector{Vector{Int}}
    streams::Vector{CuStream}
    Fbufs::Vector{CuVector{T}}
    graph::Bool
    exec::Union{Nothing, CuGraphExec}
    # merged chains of top fronts (config().factor_merge), or nothing: the merge and the merged factor arrays
    amal::Any
    merged::NTuple{4, CuVector{T}}
end

function FactorPlan(F::ChordalSLU{Sem, T, I}; large::Integer = 256, nt::Integer = Threads.nthreads(), graph::Bool = true,
        nstreams::Integer = 8, maxbytes::Integer = 2^31, merge::Integer = config().factor_merge, alpha::Real = 0.5) where {Sem, T, I}
    return FactorPlan(F, bottom_plan(F; large, nt); graph, nstreams, maxbytes, merge, alpha)
end

#
# The plan in two parts: bottom_plan, what the CPU factorization of the bottom forest needs (which fronts
# go to the GPU, the subtrees of each thread and their update stacks), and FactorPlan(F, B), the GPU's
# part (the merge of the top, its schedule and the device buffers). The bottom fronts can be factored
# (factor_bottom!(F, B.workers)) while the second part is made; `buffers` = the device factor arrays and
# boundary buffer (LD, LL, UD, UL, Mb) when they were allocated beforehand (device_factor).
#
struct BottomPlan{T, I}
    large::Int
    nt::Int
    istop::BitVector
    topwork::Float64
    nlarge::Int
    workers::Vector{BottomWorker{T, I}}
    nbndval::Int
    bndoff::Vector{Int}
end

function bottom_plan(F::ChordalSLU{Sem, T, I}; large::Integer = 256, nt::Integer = Threads.nthreads()) where {Sem, T, I}
    s = F.s
    check_semiring(s, T)
    S = F.S.S
    nf = Int(MF.nv(S.res))
    pnt = S.pnt
    Rptr = MF.pointers(S.res)
    Sptr = MF.pointers(S.sep)
    relptr = MF.pointers(S.rel)
    chdptr = MF.pointers(S.chd)
    chdtgt = MF.targets(S.chd)

    nn(f) = Int(Rptr[f + 1] - Rptr[f])
    na(f) = Int(Sptr[f + 1] - Sptr[f])
    children(f) = (Int(chdtgt[p]) for p in chdptr[f]:(chdptr[f + 1] - 1))

    @step "top selection" begin
        istop = falses(nf)

        for f in 1:nf
            if nn(f) + na(f) >= large
                g = f

                while !iszero(g) && !istop[g]
                    istop[g] = true
                    g = pnt[g]
                end
            end
        end

        work(f) = nn(f)^3 / 3 + nn(f)^2 * na(f) + nn(f) * na(f)^2
        nlarge = count(istop)               # the top of the large fronts (and their ancestors)
        #
        # Balance (config().factor_balance): the CPU factors the bottom forest one subtree per thread, so a
        # subtree with much of the work keeps one thread busy while the others idle. (When no front
        # reaches `large` there is no top, and the bottom forest is the whole tree: one subtree, on one
        # thread.) For b = 1, 2, 4, … ≤ factor_balance, the subtrees with more than 1/(b nt) of the work
        # join the top; the b kept is the one whose CPU saving (the heaviest bottom subtree, or the even
        # share, before and after) exceeds the GPU top levels it adds by the most, a level counted as
        # config().factor_level_work, or none. Work here counts a front as at least 256 multiply-adds
        # (its fixed cost on the CPU, ~0.1 µs, against ~0.3 ns per multiply-add). Subtree work grows
        # toward the roots, so the fronts marked are closed under ancestors, as the top must be.
        #
        cf = config()

        if cf.factor_balance > 0 && nt > 1
            wsub = [work(f) + 256 for f in 1:nf]

            for f in 1:nf
                iszero(pnt[f]) || (wsub[pnt[f]] += wsub[f])
            end

            total = sum(f -> iszero(pnt[f]) ? wsub[f] : 0.0, 1:nf; init = 0.0)
            theight = zeros(Int, nf)
            #
            # (the CPU's critical path in work, the levels of the top) with the top `top`
            #
            function cpu_levels(top)
                fill!(theight, 0); levels = 0; heavy = 0.0; rest = 0.0

                for f in 1:nf                       # postorder: children first
                    p = pnt[f]

                    if top[f]
                        levels = max(levels, theight[f] + 1)
                        iszero(p) || (theight[p] = max(theight[p], theight[f] + 1))
                    elseif iszero(p) || top[p]      # a bottom subtree
                        heavy = max(heavy, wsub[f]); rest += wsub[f]
                    end
                end

                return max(heavy, rest / nt), levels
            end

            c0, l0 = cpu_levels(istop)
            best = 0.0; keep = nothing; b = 1

            while b <= cf.factor_balance
                top = istop .| (wsub .> total / (b * nt))
                c, l = cpu_levels(top)
                gain = (c0 - c) - cf.factor_level_work * max(l - l0, 1)
                gain > best && (best = gain; keep = top)
                b *= 2
            end

            isnothing(keep) || copyto!(istop, keep)
        end

        topwork = sum(work(f) for f in 1:nf if istop[f]; init = 0.0) / max(sum(work, 1:nf), 1.0)
    end
    #
    # the bottom forest: subtrees rooted at the bottom fronts whose parent is
    # in the top (or that are roots), balanced over nt threads by work
    #
    @step "bottom forest" begin
        sz = ones(Int, nf)

        for f in 1:nf
            iszero(pnt[f]) || (sz[pnt[f]] += sz[f])
        end

        roots = [f for f in 1:nf if !istop[f] && (iszero(pnt[f]) || istop[pnt[f]])]
        weight = [sum(g -> work(g) + 64, (f - sz[f] + 1):f) for f in roots]
        nw = max(1, min(nt, length(roots)))
        lists = [Int[] for _ in 1:nw]; load = zeros(nw)

        for r in sortperm(weight; rev = true)
            w = argmin(load)
            push!(lists[w], roots[r]); load[w] += weight[r]
        end
    end
    #
    # simulate each worker's update stack: the updates of its subtree roots
    # whose parent is in the top (the boundary) stay on it, at offsets that
    # the simulation reproduces exactly
    #
    @step "worker stacks" begin
        bndoff = zeros(Int, nf)
        workers = BottomWorker{T, I}[]
        base = 1

        for list in lists
            sort!(list)
            ranges = [(f - sz[f] + 1):f for f in list]
            cstack = Tuple{Int, Int}[]; cpeak = 0; cdepth = 0; nFcpu = 2      # (offset, length)

            for r in ranges, f in r
                for _ in children(f)
                    pop!(cstack)
                end

                if ispositive(na(f))
                    off = isempty(cstack) ? 1 : sum(cstack[end])
                    push!(cstack, (off, na(f)^2))
                    cpeak = max(cpeak, off + na(f)^2 - 1)
                    cdepth = max(cdepth, length(cstack))

                    if !iszero(pnt[f]) && istop[pnt[f]]
                        bndoff[f] = base + off - 1
                    end
                end

                nFcpu = max(nFcpu, (nn(f) + na(f))^2)
            end

            nb = isempty(cstack) ? 0 : sum(cstack[end]) - 1
            push!(workers, BottomWorker{T, I}(ranges, MF.FVector{I}(undef, cdepth + 1), MF.FVector{T}(undef, max(cpeak, 1)),
                MF.FVector{T}(undef, nFcpu), spool_mt(s, T, 1), nb, base))
            base += nb
        end

        nbndval = base - 1
    end

    return BottomPlan{T, I}(large, nt, istop, topwork, nlarge, workers, nbndval, bndoff)
end

# the device factor arrays and the boundary buffer of a plan (LD, LL, UD, UL, Mb), allocated ahead
device_factor(F::ChordalSLU{<:Any, T}, nbndval::Integer) where {T} =
    (CuVector{T}(undef, length(F.LDval)), CuVector{T}(undef, length(F.LLval)), CuVector{T}(undef, length(F.UDval)),
     CuVector{T}(undef, length(F.ULval)), CuVector{T}(undef, max(nbndval, 1)))

function FactorPlan(F::ChordalSLU{Sem, T, I}, B::BottomPlan{T, I}; graph::Bool = true, nstreams::Integer = 8, maxbytes::Integer = 2^31,
        merge::Integer = config().factor_merge, alpha::Real = 0.5, buffers = nothing) where {Sem, T, I}
    S = F.S.S
    nf = Int(MF.nv(S.res))
    pnt = S.pnt
    Rptr = MF.pointers(S.res)
    Sptr = MF.pointers(S.sep)
    relptr = MF.pointers(S.rel)
    chdptr = MF.pointers(S.chd)
    chdtgt = MF.targets(S.chd)
    nn(f) = Int(Rptr[f + 1] - Rptr[f])
    na(f) = Int(Sptr[f + 1] - Sptr[f])
    children(f) = (Int(chdtgt[p]) for p in chdptr[f]:(chdptr[f + 1] - 1))
    istop = B.istop; topwork = B.topwork; nlarge = B.nlarge; nt = B.nt
    workers = B.workers; nbndval = B.nbndval; bndoff = B.bndoff
    cf = config()
    dbuf = isnothing(buffers) ? device_factor(F, B.nbndval) : buffers
    #
    # merge chains of top fronts (see config().factor_merge): the GPU then factors merged fronts, in a merged
    # copy of the factor (gathered before and scattered back after the top), and every child's update
    # is assembled at its positions in the merged front of its parent
    #
    @step "top merge" begin
        amal = nothing

        if merge > 1 && any(istop) && nlarge >= cf.factor_merge_min
            Stgt = MF.targets(S.sep)
            # (views of the symbolic arrays: amalgamate_fronts only reads them, and copies of the separators
            # were tens of MB of garbage per call)
            amal = @step "amalgamation" amalgamation(amalgamation_key(S.Dptr), I, merge, alpha, view(Rptr, 1:(nf + 1)), view(Sptr, 1:(nf + 1)),
                view(Stgt, 1:(Sptr[nf + 1] - 1)), view(S.Dptr, 1:(nf + 1)), view(S.Lptr, 1:(nf + 1)),
                view(pnt, 1:nf), view(S.idx, 1:size(F, 1)); allowed = istop, compact = true)     # cached per symbolic factorization
            isnothing(amal) || amal.nf == nf && (amal = nothing)        # nothing merged
        end
    end
    if !isnothing(amal)
        return @step "merged plan" merged_plan(F, amal, istop, topwork, nt, workers, nbndval, bndoff, nstreams, maxbytes, graph, dbuf)
    end
    #
    # the top fronts, in postorder, with a stack of update matrices
    #
    @step "top schedule" begin
        gstack = Tuple{Int, Int}[]; gpeak = 0
        nFgpu = 1
        tasks = Tuple{Int, Int, Int, Int, Int, Vector{Tuple{Bool, Int, Int, Int}}}[]

        for f in 1:nf
            if istop[f]
                kids = Tuple{Bool, Int, Int, Int}[]

                for c in Iterators.reverse(collect(children(f)))
                    if istop[c]
                        off, _ = pop!(gstack)
                        push!(kids, (true, off, na(c), Int(relptr[c])))
                    else
                        push!(kids, (false, bndoff[c], na(c), Int(relptr[c])))
                    end
                end

                out = isempty(gstack) ? 1 : sum(gstack[end])
                push!(tasks, (nn(f), na(f), Int(S.Dptr[f]), Int(S.Lptr[f]), out, kids))

                if ispositive(na(f))
                    push!(gstack, (out, na(f)^2))
                    gpeak = max(gpeak, out + na(f)^2 - 1)
                end

                nFgpu = max(nFgpu, (nn(f) + na(f))^2)
            end
        end
        #
        # concurrent schedule: static update slots, levels by height in the top
        #
        topsize = sum(f -> istop[f] ? na(f)^2 : 0, 1:nf; init = 0)
        nstreams = (nstreams > 1 && topsize * sizeof(T) <= maxbytes) ? Int(nstreams) : 1
        levels = Vector{Int}[]

        if nstreams > 1
            slot = zeros(Int, nf); theight = zeros(Int, nf); q = 1
            tasks = empty(tasks)

            for f in 1:nf
                istop[f] || continue
                slot[f] = q; q += na(f)^2
                theight[f] = 1 + maximum((theight[c] for c in children(f) if istop[c]); init = 0)
                kids = [istop[c] ? (true, slot[c], na(c), Int(relptr[c])) : (false, bndoff[c], na(c), Int(relptr[c])) for c in Iterators.reverse(collect(children(f)))]
                push!(tasks, (nn(f), na(f), Int(S.Dptr[f]), Int(S.Lptr[f]), slot[f], kids))
                length(levels) < theight[f] && push!(levels, Int[])
                push!(levels[theight[f]], length(tasks))
            end

            gpeak = topsize
        end
    end
    @step "device buffers" begin
        P = FactorPlan{Sem, T, I}(
            F, istop, topwork, nt, workers, nbndval,
            dbuf..., CuVector{T}(undef, max(gpeak, 1)), CuVector{T}(undef, 1),
            upload(relptr), upload(MF.targets(S.rel)),
            tasks, levels, [CuStream() for _ in 1:nstreams],
            front_buffers(T, tasks, levels, nstreams), graph, nothing,
            nothing, ntuple(_ -> CuVector{T}(undef, 0), 4),
        )
        #
        # the plan orders every cross-stream access itself (fork event, level
        # barriers, join), so CUDA.jl's implicit synchronization on stream
        # switches is disabled for its buffers; it would also break capture
        #
        for x in (P.LD, P.LL, P.UD, P.UL, P.Mb, P.Mg, P.relptr, P.reltgt, P.Fbufs...)
            CUDA.enable_synchronization!(x, false)
        end
    end
    return P
end

# config().factor_merge: merge chains of top fronts in the GPU factorization up to this width (1 = off).
# The top of the elimination tree is nearly a chain (about 40 levels of 1 or 2 fronts on a 3D grid),
# and most top fronts have 1–8 pivots, so the factorization is a sequence of ~10 tiny launches per
# front. Merged fronts are factored as one dense front (the same eliminations; the padding is the
# semiring zero).

function merged_plan(F::ChordalSLU{Sem, T, I}, A::Amalgamation, istop, topwork, nt, workers, nbndval, bndoff, nstreams, maxbytes, graph,
        dbuf = device_factor(F, nbndval)) where {Sem, T, I}
    S = F.S.S
    nf = Int(MF.nv(S.res))
    Sptr = MF.pointers(S.sep); Stgt = MF.targets(S.sep)
    chdptr = MF.pointers(S.chd); chdtgt = MF.targets(S.chd)
    na(f) = Int(Sptr[f + 1] - Sptr[f])
    children(f) = (Int(chdtgt[p]) for p in chdptr[f]:(chdptr[f + 1] - 1))
    ng = A.nf; grp = A.group
    ufirst = zeros(Int, ng); ulast = zeros(Int, ng)

    for f in 1:nf
        q = grp[f]
        iszero(ufirst[q]) && (ufirst[q] = f)
        ulast[q] = f
    end

    uw(q) = Int(A.Rptr[q + 1] - A.Rptr[q])
    una(q) = Int(A.Sptr[q + 1] - A.Sptr[q])
    utop(q) = istop[ufirst[q]]
    #
    # the relative indices of the children's updates in their merged parents (newrel), child after child in
    # the order of the tasks: the loops below give each child its range (jobs: (q, c, first)), and
    # merged_positions! fills them afterwards, on several threads
    #
    jobs = Tuple{Int, Int, Int}[]; nrel = 0

    externals(q) = sort!([c for m in ufirst[q]:ulast[q] for c in children(m) if grp[c] != q])
    tasks = Tuple{Int, Int, Int, Int, Int, Vector{Tuple{Bool, Int, Int, Int}}}[]
    levels = Vector{Int}[]
    nFgpu = 1
    topsize = sum(q -> utop(q) ? una(q)^2 : 0, 1:ng; init = 0)
    nstreams = (nstreams > 1 && topsize * sizeof(T) <= maxbytes) ? Int(nstreams) : 1

    if nstreams > 1
        slot = zeros(Int, ng); theight = zeros(Int, ng); qq = 1

        for q in 1:ng
            utop(q) || continue
            slot[q] = qq; qq += una(q)^2
            ks = externals(q)
            theight[q] = 1 + maximum((theight[grp[c]] for c in ks if istop[c]); init = 0)
            kids = Tuple{Bool, Int, Int, Int}[]

            for c in Iterators.reverse(ks)
                rp = nrel + 1; nrel += na(c); push!(jobs, (q, c, rp))
                push!(kids, istop[c] ? (true, slot[grp[c]], na(c), rp) : (false, bndoff[c], na(c), rp))
            end

            push!(tasks, (uw(q), una(q), Int(A.Dptr[q]), Int(A.Lptr[q]), slot[q], kids))
            length(levels) < theight[q] && push!(levels, Int[])
            push!(levels[theight[q]], length(tasks))
            nFgpu = max(nFgpu, (uw(q) + una(q))^2)
        end

        gpeak = topsize
    else
        gstack = Tuple{Int, Int}[]; gpeak = 0

        for q in 1:ng
            utop(q) || continue
            ks = externals(q)
            offs = Dict{Int, Int}()

            for c in Iterators.reverse(ks)
                istop[c] && (offs[c] = first(pop!(gstack)))
            end

            kids = Tuple{Bool, Int, Int, Int}[]

            for c in Iterators.reverse(ks)
                rp = nrel + 1; nrel += na(c); push!(jobs, (q, c, rp))
                push!(kids, istop[c] ? (true, offs[c], na(c), rp) : (false, bndoff[c], na(c), rp))
            end

            out = isempty(gstack) ? 1 : sum(gstack[end])
            push!(tasks, (uw(q), una(q), Int(A.Dptr[q]), Int(A.Lptr[q]), out, kids))

            if ispositive(una(q))
                push!(gstack, (out, una(q)^2))
                gpeak = max(gpeak, out + una(q)^2 - 1)
            end

            nFgpu = max(nFgpu, (uw(q) + una(q))^2)
        end
    end

    newrel = @step "positions" merged_positions(A, Sptr, Stgt, jobs, nrel)
    @step "device buffers" begin
    merged = (CuVector{T}(undef, length(A.mLD)), CuVector{T}(undef, length(A.mLL)), CuVector{T}(undef, length(A.mUD)), CuVector{T}(undef, length(A.mUL)))
    P = FactorPlan{Sem, T, I}(
        F, istop, topwork, nt, workers, nbndval,
        dbuf..., CuVector{T}(undef, max(gpeak, 1)), CuVector{T}(undef, 1),
        upload(MF.pointers(S.rel)), upload(newrel),
        tasks, levels, [CuStream() for _ in 1:nstreams],
        front_buffers(T, tasks, levels, nstreams), graph, nothing,
        A, merged,
    )

    for x in (P.LD, P.LL, P.UD, P.UL, P.Mb, P.Mg, P.relptr, P.reltgt, P.Fbufs..., merged...)
        CUDA.enable_synchronization!(x, false)
    end
    end

    return P
end

#
# newrel[first:first + na(c) - 1] for each job (q, c, first): the position of each vertex u of sep(c) in
# merged front q, its residual (a range of vertices) first, then its separator sep(q) (A.Stgt); both
# sorted, so each search starts where the last one ended. Jobs write disjoint ranges: threads take runs
# of jobs with near-equal numbers of entries.
#
function merged_positions(A::Amalgamation{I}, Sptr, Stgt, jobs::Vector{Tuple{Int, Int, Int}}, nrel::Integer) where {I}
    newrel = Vector{I}(undef, nrel)

    fillrange! = function (j0, j1)
        @inbounds for j in j0:j1
            q, c, o = jobs[j]
            r0 = Int(A.Rptr[q]); w = Int(A.Rptr[q + 1]) - r0
            s0 = Int(A.Sptr[q]); s1 = Int(A.Sptr[q + 1]) - 1; k = s0

            for p in Int(Sptr[c]):(Int(Sptr[c + 1]) - 1)
                u = Stgt[p]

                if r0 <= u < r0 + w
                    newrel[o] = I(u - r0 + 1)
                else
                    k = searchsortedfirst(A.Stgt, u, k, s1, Base.Order.Forward)
                    k <= s1 && A.Stgt[k] == u || error("merged factorization: separator vertex outside the parent's front")
                    newrel[o] = I(w + k - s0 + 1)
                end

                o += 1
            end
        end
    end

    nj = length(jobs); T = Threads.nthreads()

    if T > 1 && nrel >= 2^16
        starts = [jobs[j][3] for j in 1:nj]
        bnd = [searchsortedfirst(starts, cld(t * nrel, T) + 1) for t in 0:T]
        bnd[1] = 1; bnd[end] = nj + 1

        Threads.@threads for t in 1:T
            fillrange!(bnd[t], bnd[t + 1] - 1)
        end
    else
        fillrange!(1, nj)
    end

    return newrel
end

#
# The front buffer of each stream (factor_top_streams!: the i-th task of a level runs on stream mod1(i, ns)),
# as large as the largest front run on that stream: with one dense front at the top of the tree, one stream
# needs its size and the others much less (they were all that size: 8 × 590 MB on EB_16384_64, each a new
# allocation on a cold call). One stream (no levels): one buffer for every task.
#
function front_buffers(::Type{T}, tasks, levels, nstreams::Integer) where {T}
    front(t) = (tasks[t][1] + tasks[t][2])^2

    if nstreams == 1 || isempty(levels)
        return [CuVector{T}(undef, max(1, maximum(front, eachindex(tasks); init = 1)))]
    end

    sizes = ones(Int, nstreams)

    for level in levels, (i, t) in enumerate(level)
        k = mod1(i, nstreams)
        sizes[k] = max(sizes[k], front(t))
    end

    return [CuVector{T}(undef, sz) for sz in sizes]
end

# the factor arrays the top fronts are factored in: (LD, UD, LL, UL)
top_arrays(P::FactorPlan) = isnothing(P.amal) ? (P.LD, P.UD, P.LL, P.UL) : (P.merged[1], P.merged[3], P.merged[2], P.merged[4])

ntop(P::FactorPlan) = count(P.istop)

# CPU: the bottom fronts, with CPU.sgetrf_loop!, one task per worker
# (the subtrees are disjoint, so the workers write disjoint parts of F)
factor_bottom!(P::FactorPlan) = (factor_bottom!(P.F, P.workers); P)

# each(W): run by worker W's task when its fronts are done (e.g. upload_worker!)
function factor_bottom!(F::ChordalSLU{Sem, T, I}, workers::Vector{BottomWorker{T, I}}; each = nothing) where {Sem, T, I}
    s = F.s; S = F.S.S
    Rptr = MF.pointers(S.res)

    @sync for W in workers
        Threads.@spawn begin
            Mptr = W.Mptr; Mval = W.Mval
            ns = zero(I); Mptr[one(I)] = one(I)

            for r in W.ranges, f in r
                if isone(Rptr[f + 1] - Rptr[f])
                    ns = sgetrf_loop_1!(s, F.LDval, F.UDval, F.LLval, F.ULval, S.Dptr, S.Lptr, Mptr, Mval, W.Fval, S.res, S.rel, S.chd, 1, ns, I(f))
                else
                    ns = sgetrf_loop!(s, F.LDval, F.UDval, F.LLval, F.ULval, S.Dptr, S.Lptr, Mptr, Mval, W.Fval, S.res, S.rel, S.chd, W.pool, 1, ns, I(f))
                end
            end

            @assert Mptr[ns + one(I)] - one(I) == W.nbndval
            isnothing(each) || each(W)
        end
    end

    return F
end

#
# The host factor to the device in two parts: upload_factor! copies it while the bottom is factored (the
# top fronts' entries, which the bottom does not change), and upload_bottom! the bottom fronts'
# part when they are done. A bottom subtree is a contiguous range of fronts in postorder, so of the factor
# arrays; there may be thousands (social graphs), so neighbouring ones are merged into one span when the
# gap between them (top fronts' entries, unchanged: copied again with them) is at most `gap` elements
# (bottom_spans). Every copy is complete when these return.
#
function bottom_spans(F::ChordalSLU, workers::Vector{<:BottomWorker}; gap::Integer = 2^15)
    S = F.S.S; Dptr = S.Dptr; Lptr = S.Lptr
    ranges = sort!([(first(r), last(r)) for W in workers for r in W.ranges])
    spans = Tuple{Int, Int}[]                   # fronts lo:hi

    for (lo, hi) in ranges
        if !isempty(spans) && (Int(Dptr[lo]) - Int(Dptr[spans[end][2] + 1])) + (Int(Lptr[lo]) - Int(Lptr[spans[end][2] + 1])) <= gap
            spans[end] = (spans[end][1], hi)
        else
            push!(spans, (lo, hi))
        end
    end

    return spans
end

# d[a:b-1] ← h[a:b-1] for the spans of fronts (D: through Dptr, else Lptr), in pieces
function upload_spans!(d, h, ptr, spans; piece::Integer = typemax(Int))
    for (lo, hi) in spans
        a = Int(ptr[lo]); b = Int(ptr[hi + 1])

        while b > a
            m = min(b - a, piece)
            GC.@preserve h unsafe_copyto!(pointer(d, a), pointer(h, a), m)
            a += m
        end
    end

    return
end

function upload_factor!(dbuf, F::ChordalSLU, spans)
    S = F.S.S
    LD, LL, UD, UL, _ = dbuf
    upload_spans!(LD, F.LDval, S.Dptr, spans); upload_spans!(UD, F.UDval, S.Dptr, spans)
    upload_spans!(LL, F.LLval, S.Lptr, spans); upload_spans!(UL, F.ULval, S.Lptr, spans)
    return
end

#
# The entries of A into the factor in two parts, copyto!(F, A) split by front:
#
#   host_entries!     the fronts of the bottom spans (the bottom subtrees, which the CPU factors, and the
#                     top fronts between them that upload_bottom! copies along) into the host factor F,
#                     as copyto! writes them;
#   device_entries!   the other fronts (the GPU's top) straight into the device factor: filled with the
#                     semiring zero, then each entry of A to its slot, found on the GPU (the same search
#                     as copyto!'s, one warp per column of A).
#
# The same factor as copyto! followed by the upload of all of it, but the device receives A and the
# symbolic arrays it needs instead of the factor (1.2 GB on EB_16384_64, the transfer that held up the
# plan on a cold call), the host's top part is never written (malloc'd memory that is never touched costs
# nothing, where copyto! faulted in all of it), and the host searches only the bottom's slots. The two
# parts are independent: they run at the same time.
#
function front_onhost(F::ChordalSLU, spans)
    onhost = zeros(Bool, Int(MF.nv(F.S.S.res)))

    for (lo, hi) in spans
        onhost[lo:hi] .= true
    end

    return onhost
end

function host_entries!(F::ChordalSLU{<:Any, T}, A::SparseMatrixCSC, onhost::Vector{Bool}, spans) where {T}
    S = F.S.S; z = szero(F.s, T, Val(:N))
    Dptr = S.Dptr; Lptr = S.Lptr

    for (lo, hi) in spans
        CPU.pfill!(view(F.LDval, Int(Dptr[lo]):(Int(Dptr[hi + 1]) - 1)), z); CPU.pfill!(view(F.UDval, Int(Dptr[lo]):(Int(Dptr[hi + 1]) - 1)), z)
        CPU.pfill!(view(F.LLval, Int(Lptr[lo]):(Int(Lptr[hi + 1]) - 1)), z); CPU.pfill!(view(F.ULval, Int(Lptr[lo]):(Int(Lptr[hi + 1]) - 1)), z)
    end

    fill!(F.Nval, z)
    n = size(A, 2)
    nchunk = SparseArrays.getcolptr(A)[n + 1] - 1 < 2^15 ? 1 : 8 * Threads.nthreads()

    if isone(nchunk)
        host_columns!(F, A, onhost, 1, n)
    else
        Threads.@threads for k in 1:nchunk
            host_columns!(F, A, onhost, cld((k - 1) * n, nchunk) + 1, cld(k * n, nchunk))
        end
    end

    return F
end

# CPU.scatter_columns! for columns c0:c1, only the entries whose slot is in a front on the host
function host_columns!(F::ChordalSLU{<:Any, T, I}, A::SparseMatrixCSC, onhost::Vector{Bool}, c0::Integer, c1::Integer) where {T, I}
    S = F.S.S
    rptr = MF.pointers(S.res); sptr = MF.pointers(S.sep); stgt = MF.targets(S.sep)
    idx = S.idx; Dptr = S.Dptr; Lptr = S.Lptr
    Bptr = F.S.Bptr; fcc = F.S.fcc
    Nptr = MF.pointers(F.S.N); Ntgt = MF.targets(F.S.N)
    LD = F.LDval; LL = F.LLval; UD = F.UDval; UL = F.ULval; Nval = F.Nval
    rinvp = F.rinvp; cinvp = F.cinvp
    Aptr = SparseArrays.getcolptr(A); Arow = SparseArrays.rowvals(A); Aval = SparseArrays.nonzeros(A)

    @inbounds for c in c0:c1
        j = convert(I, cinvp[c]); f = idx[j]
        jlo = rptr[f]; nn = rptr[f + one(I)] - jlo
        na = sptr[f + one(I)] - sptr[f]
        jstrt = Bptr[fcc[f]]
        hf = onhost[f]

        for p in Aptr[c]:(Aptr[c + 1] - 1)
            i = convert(I, rinvp[Arow[p]])

            if i < jstrt
                r = CPU.sortedindex(Ntgt, Nptr[j], Nptr[j + one(I)] - one(I), i)
                iszero(r) || (Nval[r] = Aval[p])
            elseif jlo <= i < jlo + nn
                hf || continue
                o = Dptr[f] + (i - jlo) + (j - jlo) * nn
                LD[o] = Aval[p]; UD[o] = Aval[p]
            elseif i > j
                hf || continue
                k = CPU.sortedindex(stgt, sptr[f], sptr[f + one(I)] - one(I), i)
                iszero(k) || (LL[Lptr[f] + (k - sptr[f]) + (j - jlo) * na] = Aval[p])
            else
                g = idx[i]
                onhost[g] || continue
                ilo = rptr[g]; mm = rptr[g + one(I)] - ilo
                k = CPU.sortedindex(stgt, sptr[g], sptr[g + one(I)] - one(I), j)
                iszero(k) || (UL[Lptr[g] + (i - ilo) + (k - sptr[g]) * mm] = Aval[p])
            end
        end
    end

    return
end

# the index of v in the sorted tgt[lo:hi], or 0 (CPU.sortedindex)
@inline function sortedindex_gpu(tgt, lo::Int, hi::Int, v::Int32)
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1; t = tgt[mid]
        t == v && return mid
        t < v ? (lo = mid + 1) : (hi = mid - 1)
    end

    return 0
end

# host_columns! for the fronts not on the host, one warp per column of A
function device_entries_kernel!(LD, UD, LL, UL, Aptr, Arow, Aval, rinvp, cinvp, idx, rptr, sptr, stgt, Dptr, Lptr, jstart, onhost)
    lane = (threadIdx().x - 1) % 32
    warp = ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) ÷ 32
    nwarps = (gridDim().x * blockDim().x) ÷ 32
    n = length(cinvp)
    c = warp + 1

    @inbounds while c <= n
        j = Int(cinvp[c]); f = Int(idx[j])
        jlo = Int(rptr[f]); nn = Int(rptr[f + 1]) - jlo
        na = Int(sptr[f + 1]) - Int(sptr[f])
        jstrt = Int(jstart[f]); hf = onhost[f]
        p = Int(Aptr[c]) + lane

        while p < Int(Aptr[c + 1])
            i = Int(rinvp[Int(Arow[p])])

            if i < jstrt
                # (coupling between components: no such entries on the GPU, check_coupling)
            elseif jlo <= i < jlo + nn
                if !hf
                    o = Int(Dptr[f]) + (i - jlo) + (j - jlo) * nn
                    LD[o] = Aval[p]; UD[o] = Aval[p]
                end
            elseif i > j
                if !hf
                    k = sortedindex_gpu(stgt, Int(sptr[f]), Int(sptr[f + 1]) - 1, i % Int32)
                    iszero(k) || (LL[Int(Lptr[f]) + (k - Int(sptr[f])) + (j - jlo) * na] = Aval[p])
                end
            else
                g = Int(idx[i])

                if !onhost[g]
                    ilo = Int(rptr[g]); mm = Int(rptr[g + 1]) - ilo
                    k = sortedindex_gpu(stgt, Int(sptr[g]), Int(sptr[g + 1]) - 1, j % Int32)
                    iszero(k) || (UL[Int(Lptr[g]) + (i - ilo) + (k - Int(sptr[g])) * mm] = Aval[p])
                end
            end

            p += 32
        end

        c += nwarps
    end

    return
end

# x as Int32 (vertex labels and column pointers below 2³¹), converted on threads when long
function int32_copy(x::AbstractVector)
    y = Vector{Int32}(undef, length(x)); m = length(x)

    if m >= 2^18 && Threads.nthreads() > 1
        T = Threads.nthreads()

        Threads.@threads for t in 1:T
            @inbounds for q in (cld((t - 1) * m, T) + 1):cld(t * m, T)
                y[q] = x[q] % Int32
            end
        end
    else
        @inbounds for q in 1:m
            y[q] = x[q] % Int32
        end
    end

    return y
end

# the device factor (dbuf) before the bottom is uploaded: the semiring zero, with the entries of A of the fronts not on the host
function device_entries!(dbuf, F::ChordalSLU{<:Any, T, I}, A::SparseMatrixCSC, onhost::Vector{Bool}) where {T, I}
    LD, LL, UD, UL, _ = dbuf
    z = szero(F.s, T, Val(:N))
    foreach(x -> isempty(x) || fill!(x, z), (LD, LL, UD, UL))
    n = size(A, 2); m = SparseArrays.nnz(A)
    (iszero(n) || iszero(m) || all(onhost)) && (CUDA.synchronize(); return)
    @assert n < typemax(Int32) && m < typemax(Int32)
    S = F.S.S; nf = Int(MF.nv(S.res)); nsep = Int(MF.pointers(S.sep)[nf + 1]) - 1
    jstart = [Int(F.S.Bptr[F.S.fcc[f]]) for f in 1:nf]
    vals = SparseArrays.nonzeros(A)
    dev = map(upload, (int32_copy(view(SparseArrays.getcolptr(A), 1:(n + 1))), int32_copy(view(SparseArrays.rowvals(A), 1:m)),
        eltype(A) === T ? view(vals, 1:m) : T.(view(vals, 1:m)), int32_copy(view(F.rinvp, 1:n)), int32_copy(view(F.cinvp, 1:n)),
        int32_copy(view(S.idx, 1:n)), Vector{Int}(view(MF.pointers(S.res), 1:(nf + 1))), Vector{Int}(view(MF.pointers(S.sep), 1:(nf + 1))),
        int32_copy(view(MF.targets(S.sep), 1:nsep)), Vector{Int}(view(S.Dptr, 1:(nf + 1))), Vector{Int}(view(S.Lptr, 1:(nf + 1))),
        jstart, onhost))
    nb = min(cld(n, 8), 2^16)
    @cuda threads = 256 blocks = nb device_entries_kernel!(LD, UD, LL, UL, dev...)
    CUDA.synchronize()
    foreach(CUDA.unsafe_free!, dev)
    return
end

function upload_bottom!(dbuf, F::ChordalSLU, workers::Vector{<:BottomWorker}, spans)
    S = F.S.S
    LD, LL, UD, UL, Mb = dbuf
    dev = CUDA.device()

    tasks = [Threads.@spawn begin
        CUDA.device!(dev)
        upload_spans!(d, h, ptr, spans)
    end for (d, h, ptr) in ((LD, F.LDval, S.Dptr), (UD, F.UDval, S.Dptr), (LL, F.LLval, S.Lptr), (UL, F.ULval, S.Lptr))]

    for W in workers
        if ispositive(W.nbndval)
            GC.@preserve W unsafe_copyto!(pointer(Mb, W.base), pointer(W.Mval), W.nbndval)
        end
    end

    foreach(wait, tasks)
    return
end

function upload!(P::FactorPlan)
    F = P.F

    for (d, h) in ((P.LD, F.LDval), (P.LL, F.LLval), (P.UD, F.UDval), (P.UL, F.ULval))
        GC.@preserve h unsafe_copyto!(pointer(d), pointer(h), length(h))
    end

    for W in P.workers
        if ispositive(W.nbndval)
            GC.@preserve W unsafe_copyto!(pointer(P.Mb, W.base), pointer(W.Mval), W.nbndval)
        end
    end

    CUDA.synchronize()
    return P
end

# GPU: issue the top fronts, ordered after the current stream. The GEMMs here are issued on several
# streams, so they use the heuristic kernel choice (no clean timing for the autotuner).
function factor_top!(P::FactorPlan)
    with(TUNING => false, SCRATCH_OWNER => P) do              # concurrent streams: no autotuning; P owns their scratch
        if !isnothing(P.amal)
            A = P.amal; z = szero(P.F.s, eltype(P.LD), Val(:N))
            LDm, LLm, UDm, ULm = P.merged
            @step "merge gather" begin
                amalgamate_gather!(LDm, P.LD, P.LL, A.mLD, z); amalgamate_gather!(LLm, P.LL, P.LL, A.mLL, z)
                amalgamate_gather!(UDm, P.UD, P.UL, A.mUD, z); amalgamate_gather!(ULm, P.UL, P.UL, A.mUL, z)
            end
            @step "fronts" factor_top_streams!(P)
            @step "merge scatter" begin
                amalgamate_scatter!(P.LD, P.LL, LDm, A.mLD); amalgamate_scatter!(P.LL, P.LL, LLm, A.mLL)
                amalgamate_scatter!(P.UD, P.UL, UDm, A.mUD); amalgamate_scatter!(P.UL, P.UL, ULm, A.mUL)
            end
            return P
        end

        return @step "fronts" factor_top_streams!(P)
    end
end

function factor_top_streams!(P::FactorPlan{Sem, T}) where {Sem, T}
    use_batched_top(P) && return factor_top_batched!(P)
    s = P.F.s
    scale = Val(!isintegral(s))

    LD, UD, LL, UL = top_arrays(P)

    run(t, Fbuf) = let (n₁, n₂, Dp, Lp, out, kids) = P.tasks[t]
        factor_front_gpu!(s, scale, Fbuf, LD, UD, LL, UL, P.Mg, out, n₁, n₂, Dp, Lp,
            ((k[1] ? P.Mg : P.Mb, k[2], k[3], k[4]) for k in kids), P.relptr, P.reltgt; direct = !isempty(P.levels))
    end

    ns = length(P.streams)

    if ns == 1
        fork = CuEvent(CUDA.EVENT_DISABLE_TIMING)
        record(fork, CUDA.stream())
        CUDA.wait(fork, P.streams[1])

        CUDA.stream!(P.streams[1]) do
            for t in eachindex(P.tasks)
                run(t, P.Fbufs[1])
            end
        end
        join_streams!(P.streams[1:1])
        return P
    end
    #
    # fork: every stream starts after the current one
    #
    fork = CuEvent(CUDA.EVENT_DISABLE_TIMING)
    record(fork, CUDA.stream())

    for st in P.streams
        CUDA.wait(fork, st)
    end

    for level in P.levels
        used = min(ns, length(level))

        for (i, t) in enumerate(level)
            k = mod1(i, ns)
            CUDA.stream!(() -> run(t, P.Fbufs[k]), P.streams[k])
        end
        #
        # barrier: the next level starts when this one is done on every stream
        #
        evs = [CuEvent(CUDA.EVENT_DISABLE_TIMING) for _ in 1:used]

        for k in 1:used
            record(evs[k], P.streams[k])
        end

        for j in 1:ns, k in 1:used
            j == k || CUDA.wait(evs[k], P.streams[j])
        end
    end

    join_streams!(P.streams)
    return P
end

# the current stream waits for all of `streams`
function join_streams!(streams)
    for st in streams
        ev = CuEvent(CUDA.EVENT_DISABLE_TIMING)
        record(ev, st)
        CUDA.wait(ev, CUDA.stream())
    end
end

#
# Numeric factorization with the plan; F must hold the entries of A
# (copyto!(P.F, A)). The first call issues the GPU work directly (and
# compiles the kernels); with P.graph it then records the graph that later
# calls replay.
#
function factorize!(P::FactorPlan; download::Bool = false)
    tcpu = @elapsed @step "bottom (CPU)" factor_bottom!(P)
    tup = @elapsed @step "upload" upload!(P)
    thost = 0.0

    tgpu = @elapsed @step "top (GPU)" begin
        thost = @elapsed if isnothing(P.exec)
            factor_top!(P)
        else
            CUDA.launch(P.exec)
        end

        CUDA.synchronize()
    end

    if P.graph && isnothing(P.exec)
        P.exec = capture_graph(() -> factor_top!(P))
    end

    if download
        F = P.F

        for (h, d) in ((F.LDval, P.LD), (F.LLval, P.LL), (F.UDval, P.UD), (F.ULval, P.UL))
            GC.@preserve h unsafe_copyto!(pointer(h), pointer(d), length(h))
        end

        CUDA.synchronize()
    end

    return HybridTimes(tcpu, tup, tgpu, thost, ntop(P), P.topwork)
end

#
# CPU-only reference with the same subtree parallelism: the bottom subtrees
# in parallel (factor_bottom!), then the top fronts in postorder on the CPU
# with CPU.sgetrf_loop! (multithreaded dense kernels). The boundary
# updates are pushed onto the top stack in postorder, so the stack holds
# every top front's children in the order sgetrf_loop! pops them.
#
function factorize_cpu!(P::FactorPlan{Sem, T, I}) where {Sem, T, I}
    F = P.F; s = F.s; S = F.S.S
    nf = Int(MF.nv(S.res))
    Rptr = MF.pointers(S.res); Sptr = MF.pointers(S.sep); pnt = S.pnt
    tbottom = @elapsed factor_bottom!(P)

    ttop = @elapsed begin
        # where each boundary update lives
        owner = Dict{Int, Tuple{BottomWorker{T, I}, Int}}()

        for W in P.workers
            # replay the worker's stack to find its boundary offsets
            stack = Tuple{Int, Int}[]

            for r in W.ranges, f in r
                for p in MF.pointers(S.chd)[f]:(MF.pointers(S.chd)[f + 1] - 1)
                    pop!(stack)
                end

                na = Int(Sptr[f + 1] - Sptr[f])

                if ispositive(na)
                    off = isempty(stack) ? 1 : sum(stack[end])
                    push!(stack, (off, na^2))
                    (!iszero(pnt[f]) && P.istop[pnt[f]]) && (owner[f] = (W, off))
                end
            end
        end

        size = 0; peak = 0; depth = 0; maxdepth = 0; nF = 2
        sizes = Int[]

        for f in 1:nf
            if P.istop[f]
                for _ in MF.pointers(S.chd)[f]:(MF.pointers(S.chd)[f + 1] - 1)
                    size -= pop!(sizes); depth -= 1
                end
            elseif !haskey(owner, f)
                continue
            end

            na = Int(Sptr[f + 1] - Sptr[f])

            if ispositive(na)
                push!(sizes, na^2); size += na^2; depth += 1
                peak = max(peak, size); maxdepth = max(maxdepth, depth)
            end

            P.istop[f] && (nF = max(nF, (Int(Rptr[f + 1] - Rptr[f]) + na)^2))
        end

        Mptr = MF.FVector{I}(undef, maxdepth + 1); Mval = MF.FVector{T}(undef, max(peak, 1)); Fval = MF.FVector{T}(undef, nF)
        pool = spool_mt(s, T, P.nt)
        ns = zero(I); Mptr[one(I)] = one(I)

        for f in 1:nf
            if P.istop[f]
                if isone(Rptr[f + 1] - Rptr[f])
                    ns = sgetrf_loop_1!(s, F.LDval, F.UDval, F.LLval, F.ULval, S.Dptr, S.Lptr, Mptr, Mval, Fval, S.res, S.rel, S.chd, P.nt, ns, I(f))
                else
                    ns = sgetrf_loop!(s, F.LDval, F.UDval, F.LLval, F.ULval, S.Dptr, S.Lptr, Mptr, Mval, Fval, S.res, S.rel, S.chd, pool, P.nt, ns, I(f))
                end
            elseif haskey(owner, f)
                W, off = owner[f]
                len = Int(Sptr[f + 1] - Sptr[f])^2
                ns += one(I)
                Mptr[ns + one(I)] = Mptr[ns] + len
                copyto!(Mval, Int(Mptr[ns]), W.Mval, off, len)
            end
        end
    end

    return tbottom, ttop
end

#
# The whole factorization on the CPU (factorize_cpu!: the bottom subtrees on their threads, then the top
# fronts with multithreaded dense kernels), then the factor to the device arrays of the plan, which the
# solve reads (config().factor_gpu = false). The same factor as factorize!(P) up to the rounding of
# non-idempotent semirings.
#
function factorize_host!(P::FactorPlan)
    F = P.F
    tbottom, ttop = factorize_cpu!(P)
    step_add!("bottom (CPU)", tbottom)                  # (timed inside factorize_cpu!)
    step_add!("top (CPU)", ttop)

    @step "upload" begin
        for (d, h) in ((P.LD, F.LDval), (P.LL, F.LLval), (P.UD, F.UDval), (P.UL, F.ULval))
            isempty(h) || GC.@preserve h unsafe_copyto!(pointer(d), pointer(h), length(h))
        end

        CUDA.synchronize()
    end

    return P
end

GPUSLU(P::FactorPlan; large::Integer = 2048, structure = nothing) = GPUSLU(P.F; large, factor = (P.LD, P.LL, P.UD, P.UL), structure)

#
# One-shot hybrid factorization: F ← factor of A (F must hold the entries of
# A, as after copyto!(F, A)). Returns a GPUSLU with the factor on the GPU and
# the phase times. With `download = true` the GPU part is copied back into F.
#
function sgetrf_gpu!(F::ChordalSLU; large::Integer = 256, nt::Integer = Threads.nthreads(), download::Bool = false, solve_large::Integer = 2048, nstreams::Integer = 1)
    P = FactorPlan(F; large, nt, graph = false, nstreams)
    times = factorize!(P; download)
    return GPUSLU(P; large = solve_large), times
end

function mlu_gpu(s::AbstractSemiring, A::SparseMatrixCSC; alg = MF.DEFAULT_ELIMINATION_ALGORITHM, kw...)
    F = ChordalSLU(s, A; alg)
    copyto!(F, A)
    G, times = sgetrf_gpu!(F; kw...)
    return F, G, times
end

#
# Host → device copy of a dense vector (or contiguous view). CuVector(x) on
# the FixedSizeArrays used by CliqueTrees takes a slow generic path (~40×
# slower than the raw copy).
#
function upload(v::AbstractVector{T}) where {T}
    d = CuVector{T}(undef, length(v))

    if !isempty(v)
        GC.@preserve v unsafe_copyto!(pointer(d), pointer(v), length(v))
    end

    return d
end

#
# upload(v) for a host array made while the current stream may have GPU work queued (the factorization, the
# solve's setup): from pageable memory, a copy on the current stream waits for all of that work (the driver
# stages it in stream order), and the host idles meanwhile. On a stream of its own (one per device) the copy
# waits only for itself; it is complete when this returns, so work issued afterwards on any stream sees it.
# The array is allocated on that stream too (pool memory freed by queued work is not handed to it early).
# SIDE_UPLOADS[] = false: upload(v) (A/B switch).
#
const SIDE_UPLOADS = Ref(true)
const SIDE_STREAMS = Dict{Int, CuStream}()
const SIDE_LOCK = ReentrantLock()

side_stream() = lock(() -> get!(() -> CuStream(; flags = (isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA).STREAM_NON_BLOCKING), SIDE_STREAMS, CUDA.deviceid(CUDA.device())), SIDE_LOCK)

function upload_side(v::AbstractVector{T}) where {T}
    SIDE_UPLOADS[] && !CUDA.is_capturing() || return upload(v)
    st = side_stream()

    return CUDA.stream!(st) do
        d = CuVector{T}(undef, length(v))
        isempty(v) || GC.@preserve v unsafe_copyto!(pointer(d), pointer(v), length(v); stream = st, async = false)
        d
    end
end

#
# Factorize one front on the current stream:
#
#   F ← 0;  F ← F ⊕ inj M_c injᵀ for each child c (kids: (buffer, offset, na, rel pointer))
#   L₁₁ ← LU of (F₁₁ ⊕ original entries);  U₁₁ ← upper part of L₁₁
#   L₂₁ ← (L₂₁ ⊕ F₂₁) U₁₁*;  U₁₂ ← L₁₁* (U₁₂ ⊕ F₁₂)
#   M ← F₂₂ ⊕ L₂₁ U₁₂, written to Mg[out:out + na² - 1]
#
function factor_front_gpu!(s::AbstractSemiring, scale::Val, Fbuf::CuVector{T}, LD, UD, LL, UL, Mg::CuVector{T}, out::Int,
        n₁::Int, n₂::Int, Dp::Int, Lp::Int, kids, relptr_g, reltgt_g; direct::Bool = false) where {T}
    nj = n₁ + n₂
    L₁₁ = reshape(view(LD, Dp:(Dp + n₁ * n₁ - 1)), n₁, n₁)
    U₁₁ = reshape(view(UD, Dp:(Dp + n₁ * n₁ - 1)), n₁, n₁)

    # (only with static update slots: on the single-stream stack the parent's update may overlap its children's)
    if direct && config().direct_assembly && config().fused_front && ispositive(n₂) && n₁ <= DIAG_NB && sizeof(T) <= 4
        #
        # no front matrix: the original entries are already in L₁₁/U₁₁, L₂₁ and U₁₂; the update matrix
        # starts empty, and every child's update is added straight into the block it belongs to
        #
        L₂₁ = reshape(view(LL, Lp:(Lp + n₁ * n₂ - 1)), n₂, n₁)
        U₁₂ = reshape(view(UL, Lp:(Lp + n₁ * n₂ - 1)), n₁, n₂)
        M = reshape(view(Mg, out:(out + n₂ * n₂ - 1)), n₂, n₂)
        @phase FTIMER[] :assemble merge_diag_gpu!(L₁₁, U₁₁)
        @phase FTIMER[] :assemble fill!(M, szero(s, T, Val(:N)))

        for (buf, off, nac, rp) in kids
            @phase FTIMER[] :assemble extendadd_direct_gpu!(s, L₁₁, L₂₁, U₁₂, M, buf, off, nac, reltgt_g, rp)
        end

        Tb, W = trsm_workspace(T, n₁ * n₁ + 2 * n₁ * n₂)
        TU = view(Tb, 1:n₁, 1:n₁)
        TL = reshape(view(W, 1:(n₁ * n₁)), n₁, n₁)
        @phase FTIMER[] :lu_diag @cuda threads = diag_threads(n₁) front_diag_kernel!(s, scale, L₁₁, TL, TU)
        @phase FTIMER[] :assemble copyupper_gpu!(U₁₁, L₁₁)
        front_panels!(s, L₂₁, U₁₂, TL, TU, W, n₁, n₂)
        @phase FTIMER[] :schur_gemm sgemx_gpu!(s, M, L₂₁, U₁₂)
        return
    end

    Fj = reshape(view(Fbuf, 1:(nj * nj)), nj, nj)
    @phase FTIMER[] :assemble fill!(Fj, szero(s, T, Val(:N)))

    for (buf, off, nac, rp) in kids
        @phase FTIMER[] :assemble extendadd_gpu!(s, Fj, buf, off, nac, relptr_g, reltgt_g, rp)
    end

    @phase FTIMER[] :assemble combine_gpu!(s, L₁₁, U₁₁, Fj)

    if config().fused_front && ispositive(n₂) && n₁ <= DIAG_NB && sizeof(T) <= 4
        #
        # LU and both closures of the diagonal block in one kernel; the panel solves are GEMMs
        #
        Tb, W = trsm_workspace(T, n₁ * n₁ + 2 * n₁ * n₂)
        TU = view(Tb, 1:n₁, 1:n₁)
        TL = reshape(view(W, 1:(n₁ * n₁)), n₁, n₁)
        @phase FTIMER[] :lu_diag @cuda threads = diag_threads(n₁) front_diag_kernel!(s, scale, L₁₁, TL, TU)
        @phase FTIMER[] :assemble copyupper_gpu!(U₁₁, L₁₁)
        L₂₁ = reshape(view(LL, Lp:(Lp + n₁ * n₂ - 1)), n₂, n₁)
        U₁₂ = reshape(view(UL, Lp:(Lp + n₁ * n₂ - 1)), n₁, n₂)
        @phase FTIMER[] :assemble addblock_gpu!(s, L₂₁, Fj, n₁, 0)
        @phase FTIMER[] :assemble addblock_gpu!(s, U₁₂, Fj, 0, n₁)
        front_panels!(s, L₂₁, U₁₂, TL, TU, W, n₁, n₂)
        M = reshape(view(Mg, out:(out + n₂ * n₂ - 1)), n₂, n₂)
        @phase FTIMER[] :assemble addblock_gpu!(s, M, Fj, n₁, n₁; overwrite = true)
        @phase FTIMER[] :schur_gemm sgemx_gpu!(s, M, L₂₁, U₁₂)
        return
    end

    sgetrf_gpu!(s, L₁₁)
    @phase FTIMER[] :assemble copyupper_gpu!(U₁₁, L₁₁)

    if ispositive(n₂)
        L₂₁ = reshape(view(LL, Lp:(Lp + n₁ * n₂ - 1)), n₂, n₁)
        U₁₂ = reshape(view(UL, Lp:(Lp + n₁ * n₂ - 1)), n₁, n₂)
        @phase FTIMER[] :assemble addblock_gpu!(s, L₂₁, Fj, n₁, 0)
        @phase FTIMER[] :assemble addblock_gpu!(s, U₁₂, Fj, 0, n₁)
        @phase FTIMER[] :panel_trsm strsx_gpu!(s, Val(:N), scale, Val(:U), L₂₁, L₁₁)
        @phase FTIMER[] :panel_trsm strsx_left_gpu!(s, Val(:N), L₁₁, U₁₂)
        M = reshape(view(Mg, out:(out + n₂ * n₂ - 1)), n₂, n₂)
        @phase FTIMER[] :assemble addblock_gpu!(s, M, Fj, n₁, n₁; overwrite = true)
        @phase FTIMER[] :schur_gemm sgemx_gpu!(s, M, L₂₁, U₁₂)
    end

    return
end

