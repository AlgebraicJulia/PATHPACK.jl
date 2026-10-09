# ===== the top of the tree on the rows that need it =====
#
# Row t of the closure starts as e_{s_t} (s_t its source). After the U sweep it is nonzero only in the
# columns of the fronts on the root path of s_t's front: a front f of the top of the tree changes row t
# in the U sweep only when s_t lies in f's subtree. The other rows hold the semiring zero in f's
# residual columns, and f's step leaves them as they are (zero ⊗ x = zero, x ⊕ zero = x). Fronts are
# postordered and their columns are contiguous in elimination order, so the sources of f's subtree are
# the vertices with elimination index in Rptr[fd[f]] : Rptr[f + 1] - 1 (fd[f]: the first front of the
# subtree), a contiguous range of the rows sorted by the elimination index of their source. So each
# top front's U step runs on its subtree's rows only: the root on all n, its children on their halves,
# and so on (on meshes the U top does 0.16–0.38 of the work, on road networks ~0.01).
#
# In the L sweep every row needs every front, but rows outside f's subtree enter it with C₁ = zero:
# for them C₁ ← C₂ (L₂₁ L₁₁*), k = na instead of nn + na. So the dense L step is split:
#
#   X ← C₁[R, :]                    (the subtree's rows R)
#   C₁ ← C₂ K₂,   K₂ = L₂₁ L₁₁*     (all rows: for rows outside R the whole step)
#   C₁[R, :] ← C₁[R, :] ⊕ X K₁,  K₁ = L₁₁*
#
# The same products, ⊕ over the same terms (in another grouping): bit-identical for idempotent ⊕ (min,
# max), a rounding change for +.
#
# The rows of the closure are in the vertex labels (apsp_gpu: row t is source t), so the rows of a
# subtree are a gather, rperm[a:b]: the kernel below reads and writes them through the index (on meshes
# and road networks consecutive elimination indices are mostly near each other in the labels, 1.0–1.8×
# the 32-byte sectors of contiguous rows; up to 3× on social graphs). With the rows in elimination order
# (closure_gpu!, the multi-GPU blocks) they are a contiguous range.
#

#
# The rows of W in the order of their sources' elimination indices: row rows[q] has a source with
# elimination index key[q] (key sorted). rows: a UnitRange (rows in elimination order) or a device vector.
#
struct RowOrder{K <: AbstractVector{<:Integer}, R <: Union{UnitRange{Int}, CuVector}}
    key::K
    rows::R
end

struct TopRows{R}
    rows::R                             # RowOrder.rows
    rng::Dict{Int, UnitRange{Int}}      # top front → the positions in `rows` of its subtree's rows
    srows::Union{Nothing, CuVector{Int32}}  # gathered rows (rows a vector): each top front's rows, sorted by storage row, one after the other
    pos::Dict{Int, UnitRange{Int}}      # ... front f's at srows[pos[f]]
    rg::CuVector{Int32}                 # (first, last) position for each entry of G.top (the batched U kernel): in srows, or in rows
    maxlen::Vector{Int}                 # by level of G.topptr: the most rows of a small front
end

top_rows_on(G::GPUSLU, order) = !isnothing(order) && config().top_rows && !isnothing(G.ops[])

#
# A front's rows are visited in the order of their storage rows (the GEMM rows are independent): rows
# whose labels are close then share 32-byte sectors, and runs of 4 consecutive aligned rows are read and
# written as one 16-byte access (per row, 2.5–3× the cost of contiguous rows in label order, ~1.2× sorted).
#
top_rows_key(order::RowOrder) = Symbol(:toprows_, hash((order.key, order.rows isa CuVector ? objectid(order.rows) : order.rows)))   # (G.cache: Symbol keys)

# the top rows of G for this order: cached, made ahead (top_rows_ahead!), or made now
function top_rows(G::GPUSLU, order::RowOrder)
    key = top_rows_key(order)
    tr = get(G.cache, key, nothing)
    isnothing(tr) || return tr::TopRows
    ahead = get(G.cache, :toprows_ahead, nothing)

    if !isnothing(ahead)
        delete!(G.cache, :toprows_ahead)
        tr = ahead[1] === key ? (try fetch(ahead[2]) catch; nothing end) : nothing
    end

    isnothing(tr) && (tr = build_top_rows(top_rows_inputs(G, order)...))
    G.cache[key] = tr
    return tr::TopRows
end

#
# Make them on another thread (from apsp_single, while the operators are computed and the result is
# allocated): ~1–10 ms on the host for 10⁴–10⁵ vertices. The task gets the host arrays it reads (G's
# cache is not touched off the solve's thread).
#
function top_rows_ahead!(G::GPUSLU, order::RowOrder)
    (config().top_rows && Threads.nthreads() > 1 && !CUDA.is_capturing()) || return
    key = top_rows_key(order)
    haskey(G.cache, key) && return
    inputs = top_rows_inputs(G, order)
    dev = CUDA.device()
    G.cache[:toprows_ahead] = (key, Threads.@spawn begin
        CUDA.device!(dev)
        tr = build_top_rows(inputs...)
        CUDA.synchronize()                  # (the uploads, on this task's stream)
        tr
    end)
    return
end

function top_rows_inputs(G::GPUSLU, order::RowOrder)
    h = solve_host(G)
    gathered = order.rows isa CuVector
    hrows = gathered ? (order.rows === G.rperm ? host_rperm(G) : Array(order.rows)) : nothing
    top = get!(() -> Array(G.top), G.cache, :htop)
    return (G.n, G.nf, G.hRptr, h.pnt, h.istop, top, G.topptr, order, hrows)
end

function build_top_rows(n, nf, hRptr, pnt, istop, top, topptr, order::RowOrder, hrows)
    fsz = ones(Int, nf)

    for f in 1:nf
        p = pnt[f]
        iszero(p) || (fsz[p] += fsz[f])
    end

    rng = Dict{Int, UnitRange{Int}}()

    for f in 1:nf
        istop[f] || continue
        c0 = Int(hRptr[f - fsz[f] + 1]); c1 = Int(hRptr[f + 1]) - 1
        rng[f] = searchsortedfirst(order.key, c0):searchsortedlast(order.key, c1)
    end

    gathered = !isnothing(hrows)
    pos = Dict{Int, UnitRange{Int}}()
    srows = nothing

    if gathered
        #
        # each front's rows in increasing storage row, without sorting: visit the rows in that order and
        # append each to the lists of the top fronts on its source's root path (Σ m_f appends)
        #
        ftop = zeros(Int32, nf)                         # the first top front on the root path

        for f in nf:-1:1
            ftop[f] = istop[f] ? f : (iszero(pnt[f]) ? 0 : ftop[pnt[f]])
        end

        colfront = Vector{Int32}(undef, n)

        for f in 1:nf, c in hRptr[f]:(hRptr[f + 1] - 1)
            colfront[c] = f
        end

        nxt = zeros(Int, nf); o = 0

        for f in 1:nf
            istop[f] || continue
            pos[f] = (o + 1):(o + length(rng[f])); nxt[f] = o + 1; o += length(rng[f])
        end

        sv = Vector{Int32}(undef, o)
        byrow = isperm(hrows) ? invperm(hrows) : sortperm(hrows)    # positions in increasing row

        @inbounds for q in byrow
            g = Int(ftop[colfront[order.key[q]]])
            r = Int32(hrows[q])

            while !iszero(g)
                sv[nxt[g]] = r; nxt[g] += 1
                g = Int(pnt[g])
            end
        end

        srows = upload(sv)
    end

    rg = Vector{Int32}(undef, 2 * length(top))
    maxlen = zeros(Int, length(topptr) - 1)

    for l in 1:(length(topptr) - 1), i in topptr[l]:(topptr[l + 1] - 1)
        f = Int(top[i]); r = gathered ? pos[f] : rng[f]
        rg[2i - 1] = first(r); rg[2i] = last(r)
        maxlen[l] = max(maxlen[l], length(r))
    end

    return TopRows(order.rows, rng, srows, pos, upload(rg), maxlen)
end

# the row map of front f's rows (for the row kernels): an offset, or a device view of its sorted rows
frontrows(tr::TopRows, f) = isnothing(tr.srows) ? rowmap(tr.rows, tr.rng[f]) : view(tr.srows, tr.pos[f])

# the rows the batched kernel indexes with tr.rg
batched_rows(tr::TopRows) = isnothing(tr.srows) ? tr.rows : tr.srows

#
# The order of k rows whose sources are the vertices src (labels of A, on the host): by the sources'
# elimination indices (a gather, or the rows themselves when they are in that order already).
#
function row_order(G::GPUSLU, src::AbstractVector{<:Integer})
    hq = get!(() -> Array(G.cinvp), G.cache, :hcinvp)
    key = [Int(hq[v]) for v in src]
    issorted(key) && return RowOrder(key, 1:length(key))
    p = sortperm(key)
    return RowOrder(key[p], CuVector{Int}(p))
end

# the storage row of position q (from 1) of a row map: rows is an Int offset (row = offset + q) or a vector
@inline rowat(rows::Integer, q) = rows + q
@inline rowat(rows::AbstractVector, q) = @inbounds rows[q]

# the rows of positions r: a device view, or an offset
rowmap(rows::UnitRange{Int}, r::UnitRange{Int}) = first(rows) + first(r) - 2
rowmap(rows::CuVector, r::UnitRange{Int}) = view(rows, r)

#
# The batched U step of the small top fronts, on their subtrees' rows (upward_kernel! otherwise).
#
function upward_toprows_kernel!(s::AbstractSemiring, trans::Val, scale::Val, C::AbstractMatrix{T}, order, off::Int, rg, rows,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    i = off + blockIdx().x
    f = @inbounds order[i]
    a = @inbounds rg[2i - 1]; b = @inbounds rg[2i]
    q = a + Int32(threadIdx().x - 1 + (blockIdx().y - 1) * blockDim().x)

    while q <= b
        t = rowat(rows, q)
        upward_front!(s, trans, scale, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, Val(true))
        q += Int32(blockDim().x * gridDim().y)
    end

    return
end

# rows of a row map as a kernel argument (an offset as Int, a view as is)
kernel_rows(rows::Integer) = Int(rows)
kernel_rows(rows) = rows

#
#   X[q, :] ← C[rowat(rows, q), :]     (q = 1:size(X, 1); C may be an index view)
#
function gather_rows_gpu!(X::AbstractMatrix, C::AbstractMatrix, rows)
    function kernel(X, C, rows)
        q = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if q <= size(X, 1)
            t = rowat(rows, q)
            @inbounds while j <= size(X, 2)
                X[q, j] = C[t, j]
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(X, 1), size(X, 2), X, C, kernel_rows(rows))
    return X
end

# ===== U and L steps of the large fronts of a level on their subtrees' rows =====
#
# The fronts of one level of the top (one height in the U sweep, one depth in the L sweep) are
# independent: none is an ancestor of another, so they write disjoint columns (L) or disjoint rows (U:
# their subtrees' rows; the separator columns they update may be shared). On its own rows a front is
# a small GEMM (a few row tiles below the root's children), so the fronts of a level run side by side on
# TOP_STREAMS streams. The arrays they share are used without CUDA.jl's implicit synchronization (which
# would wait on the host whenever another stream touches them) and are handed back to the solve's stream
# after the join. Each front gets its own slice of a workspace (gathered rows).

const TOP_STREAMS = 8
const ROWS_INPLACE = 128            # the in-place row GEMM is one tile wide: up to this many columns

# does f take the row path in the U sweep? (else upward_large! on all rows, or nothing on no rows). A
# row of the row kernel costs ~1.1–1.4 of one of the full GEMM (gathered, its 16-byte accesses only where
# rows run consecutively), so fronts with more than ROWS_U_MAX of the rows run on all of them.
up_rows(tr::TopRows, f, n, ::Type{T}) where {T} = (m = length(tr.rng[f]); T === Float32 && 0 < m && m < ROWS_U_MAX[] * n)

const ROWS_U_MAX = Ref(0.9)
const ROWS_L_MIN = Ref(0.1)

# ... in the L sweep: :full (downward_large! on all rows), :root (a root: its rows only), :split (when it
# saves at least ROWS_L_MIN of the front's work: the rows outside the subtree skip the nn × nn part)
function down_rows(tr::TopRows, f, n, nn, na, ::Type{T}) where {T}
    m = length(tr.rng[f])
    (m == n || T !== Float32) && return :full
    iszero(na) && return isempty(tr.rng[f]) ? :none : :root
    return (n - m) * nn >= ROWS_L_MIN[] * n * (nn + na) ? :split : :full
end

# a workspace of at least len elements, kept in G's cache (grown on the solve's stream, between levels)
function rows_workspace(G::GPUSLU, ::Type{T}, len::Integer) where {T}
    w = get(G.cache, :rows_ws, nothing)

    if isnothing(w) || length(w) < len
        @assert !CUDA.is_capturing() "rows workspace must grow before graph capture"
        isnothing(w) || CUDA.unsafe_free!(w)
        w = CuVector{T}(undef, max(len, isnothing(w) ? 0 : 2 * length(w)))
        CUDA.enable_synchronization!(w, false)
        G.cache[:rows_ws] = w
    end

    return w::CuVector{T}
end

#
# The workspace at the most any level of the top needs, at once (before the sweep): the levels toward the
# root have more rows, so growing it level by level reallocated it several times, each a fresh
# allocation in a cold call.
#
function reserve_rows_workspace!(G::GPUSLU, ::Type{T}, tr::TopRows) where {T}
    need(f) = cld(length(tr.rng[f]), 4) * 4 * Int(G.hRptr[f + 1] - G.hRptr[f])
    w = 0
    for l in G.toplarge; w = max(w, sum(need, l; init = 0)); end
    for (_, large) in sweep_plan(G).toplevels; w = max(w, sum(need, large; init = 0)); end
    ispositive(w) && rows_workspace(G, T, w)
    return
end

# slices of a workspace for fronts fs: matrices of m_f rows (leading dimension rounded up to 4, so that the
# row kernels load them as 16-byte vectors) × w_f columns (w_f = 0: none), 16-byte aligned
function rows_slices(G::GPUSLU, ::Type{T}, fs, ms, ws) where {T}
    offs = Int[]; o = 0

    for (m, w) in zip(ms, ws)
        push!(offs, o)
        o += cld(m, 4) * 4 * w
    end

    buf = rows_workspace(G, T, max(o, 1))
    return [iszero(w) ? nothing : reshape(view(buf, (offs[i] + 1):(offs[i] + cld(m, 4) * 4 * w)), cld(m, 4) * 4, w) for (i, (m, w)) in enumerate(zip(ms, ws))]
end

#
# body(i) for i = 1:k, each on one of the streams (round robin), after the work queued on the solve's
# stream; the solve's stream then waits for all of them. shared: the arrays the bodies use.
#
function on_streams(body, G::GPUSLU, k::Integer, shared)
    k <= 1 && return (k == 1 && body(1); nothing)
    main = CUDA.stream()
    streams = get!(() -> [CuStream() for _ in 1:TOP_STREAMS], G.cache, :top_streams)::Vector{CuStream}
    ns = min(TOP_STREAMS, k)
    sync = [x.data[].synchronizing for x in shared]
    foreach(x -> CUDA.enable_synchronization!(x, false), shared)
    fork = CuEvent(CUDA.EVENT_DISABLE_TIMING)
    record(fork, main)
    foreach(st -> CUDA.wait(fork, st), streams[1:ns])

    try
        for i in 1:k
            CUDA.stream!(() -> body(i), streams[mod1(i, ns)])
        end
    finally
        join_streams!(streams[1:ns])
        for (x, e) in zip(shared, sync)             # (back to the solve's stream: no wait for the streams on the next use)
            pointer(x); CUDA.enable_synchronization!(x, e)
        end
    end

    return
end

# the device arrays the row steps of the top use (C's storage, the structure, the operators, the maps)
function top_shared(G::GPUSLU, C, tr::TopRows)
    ops = G.ops[]
    v = Any[storage_matrix(C), G.Stgt, ops.KU, ops.KL, G.rperm]
    tr.rows isa CuVector && push!(v, tr.rows)
    isnothing(tr.srows) || push!(v, tr.srows)
    C isa ColMapped && push!(v, C.cm, mapped_cache(G, C.cm, :wtgt, G.Stgt))   # (made here, before the streams)
    if haskey(G.cache, :opscols)                                            # (frontcols: the columns [res; sep] of the fronts)
        dcols = G.cache[:opscols][1]; push!(v, dcols)
        C isa ColMapped && push!(v, mapped_cache(G, C.cm, (:wcols, objectid(dcols)), dcols))
    end
    haskey(G.cache, :rows_ws) && push!(v, G.cache[:rows_ws])
    return unique(objectid, v)
end

function upward_top_level!(G::GPUSLU{<:Any, T}, C, M, fronts, trans::Val, scale::Val, tr::TopRows) where {T}
    n = size(C, 1)
    rf = Int[]

    for f in fronts
        if up_rows(tr, f, n, T)
            push!(rf, f)
        elseif !isempty(tr.rng[f])                  # (no rows: nothing to do)
            upward_large!(G, C, M, f, trans, scale)
        end
    end

    isempty(rf) && return
    nn(f) = Int(G.hRptr[f + 1] - G.hRptr[f])
    ws = rows_slices(G, T, rf, [length(tr.rng[f]) for f in rf], [nn(f) for f in rf])
    on_streams(i -> upward_toprows_step!(G, C, rf[i], tr, ws[i]), G, length(rf), top_shared(G, C, tr))
    return
end

#   X ← C[R, res];   C[R, sep] ← C[R, sep] ⊕ X (U₁₁* U₁₂);   C[R, res] ← X U₁₁*     (R: f's subtree's rows)
#
# (X, gathered once, is the A operand of both GEMMs: contiguous, so it is read as 16-byte vectors by every
# column tile, instead of each tile gathering the rows of C again)
function upward_toprows_step!(G::GPUSLU{<:Any, T}, C, f, tr::TopRows, X) where {T}
    s = G.s; ops = G.ops[]; r = tr.rng[f]
    Rp = G.hRptr[f]; nn = G.hRptr[f + 1] - Rp
    Sp = G.hSptr[f]; na = G.hSptr[f + 1] - Sp
    R = frontrows(tr, f); m = length(r)
    C₁ = rescols(C, Rp, nn)
    KU = reshape(view(ops.KU, ops.off[f][2]:(ops.off[f][2] + nn * (nn + na) - 1)), nn, nn + na)
    gather_rows_gpu!(view(X, 1:m, :), C₁, R)
    ispositive(na) && rows_gemm!(s, sepcols(G, C, Sp, na), X, view(KU, :, (nn + 1):(nn + na)), m, R, nothing)
    rows_gemm!(s, C₁, X, view(KU, :, 1:nn), m, R, nothing; overwrite = true)
    return
end

#
# A level of the L sweep's large fronts. Its fronts are independent (disjoint residual columns, separators
# final), so they run side by side on the streams, each a GEMM over all n rows of a few hundred to a few
# thousand blocks: one after the other, every one ends in a partly filled wave (1.1–1.7 waves of resident
# blocks on a B200), side by side the waves overlap. Per front:
#
#   :full    C₁ ← [C₁ | C₂] [L₁₁* ; L₂₁ L₁₁*]  (downward_large!, in place: nn ≤ 128; wider ones before, alone)
#   :split   X ← C₁[R, :] (before, on the solve's stream);  C₁ ← C₂ (L₂₁ L₁₁*);  C₁[R, :] ← C₁[R, :] ⊕ X L₁₁*
#   :root    X ← C₁[R, :];  C₁[R, :] ← X L₁₁*   (the other rows are zero and stay so)
#
# The GEMM configurations are chosen (and tuned, the first time) on the solve's stream before the streams
# start: the tuner cannot time candidates while other streams run.
#
function downward_top_level!(G::GPUSLU{<:Any, T}, C, M, fronts, trans::Val, tr::TopRows) where {T}
    s = G.s; ops = G.ops[]; n = size(C, 1)
    nn(f) = Int(G.hRptr[f + 1] - G.hRptr[f]); na(f) = Int(G.hSptr[f + 1] - G.hSptr[f])
    KL(f) = reshape(view(ops.KL, ops.off[f][1]:(ops.off[f][1] + (nn(f) + na(f)) * nn(f) - 1)), nn(f) + na(f), nn(f))
    jobs = Tuple{Symbol, Int}[]

    for f in fronts
        k = down_rows(tr, f, n, nn(f), na(f), T)

        if k === :full
            inplace_ok(nn(f)) ? push!(jobs, (:full, f)) : downward_large!(G, C, M, f, trans)
        elseif k in (:root, :split)
            push!(jobs, (k, f))
        end
    end

    isempty(jobs) && return
    X = rows_slices(G, T, last.(jobs), [k === :full ? 0 : length(tr.rng[f]) for (k, f) in jobs], [k === :full ? 0 : nn(f) for (k, f) in jobs])
    cfg = Vector{Any}(nothing, length(jobs))

    for (i, (k, f)) in enumerate(jobs)
        C₁ = rescols(C, G.hRptr[f], nn(f))

        if k === :full
            cfg[i] = select_gemm(s, C₁, frontcols(G, C, ops, f), KL(f), true, true)
        else
            isempty(tr.rng[f]) || gather_rows_gpu!(view(X[i], 1:length(tr.rng[f]), :), C₁, frontrows(tr, f))
            k === :split && (cfg[i] = select_gemm(s, C₁, sepcols(G, C, G.hSptr[f], na(f)), view(KL(f), (nn(f) + 1):(nn(f) + na(f)), :), true, false))
        end
    end

    on_streams(G, length(jobs), top_shared(G, C, tr)) do i
        k, f = jobs[i]; r = tr.rng[f]
        C₁ = rescols(C, G.hRptr[f], nn(f))

        if k === :full
            launch!(s, C₁, frontcols(G, C, ops, f), KL(f), cfg[i], Val(true); inplace = true)
        else
            k === :split && launch!(s, C₁, sepcols(G, C, G.hSptr[f], na(f)), view(KL(f), (nn(f) + 1):(nn(f) + na(f)), :), cfg[i], Val(true); inplace = false)
            isempty(r) || rows_gemm!(s, C₁, X[i], view(KL(f), 1:nn(f), :), length(r), frontrows(tr, f), nothing; overwrite = k === :root)
        end
    end

    return
end
