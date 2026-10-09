# ===== multi-GPU closure =====
#
# The rows of the closure are independent: row t holds the distances from one source, and the solve
# for a block of rows (sssp_gpu!) reads only the factor and writes only those rows. So every GPU gets
# its own copy of the solve structure and the factor (megabytes, next to gigabytes of output) and
# computes a contiguous block of rows with the single-GPU kernels, with no communication between
# GPUs. Each GPU holds only its block, so the output can be larger than one GPU's memory.
#
#   MG = MultiGPUSLU(P; devices)                              # P: a factorized FactorPlan
#   blocks = closure_multigpu!(MG)                            # [(rows, D_g)], D_g = D[rows, :] on devices[g]
#   blocks = closure_multigpu!(MG; labels = :original)        # D_g[t, j] = A*[rows[t], j]
#
# The solve structure is made once, on the device that holds the factor (the primary), and copied to
# the others device to device (NVLink / PCIe peer copies, no host round trip): its arrays, the merged
# factor, the dense operators of the large fronts and the host-side plans of the L sweep (layered.jl),
# which are built once on the host instead of once per device. Each device is driven by its own task
# (own thread, own streams), so the host issues the devices' kernels concurrently.

struct MultiGPUSLU{G}
    devices::Vector{CuDevice}
    parts::Vector{G}                    # parts[g] lives on devices[g]
    n::Int
end

function MultiGPUSLU(P::FactorPlan; devices = collect(CUDA.devices()), large::Integer = 8192, ops::Bool = true)
    G1 = CUDA.device!(CUDA.device(P.LD)) do
        G = GPUSLU(P; large)
        ops && precompute_ops!(G)
        G
    end

    return MultiGPUSLU(G1, devices)
end

# from a factor anywhere (host or a device): uploaded to devices[1], then copied device to device
function MultiGPUSLU(F::ChordalSLU, factor::NTuple{4, AbstractVector}; devices = collect(CUDA.devices()), large::Integer = 8192, ops::Bool = true)
    G1 = CUDA.device!(first(devices)) do
        G = GPUSLU(F; large, factor = map(x -> CuVector(Array(x)), factor))
        ops && precompute_ops!(G)
        G
    end

    return MultiGPUSLU(G1, devices)
end

# G1 on its device, and copies on the other devices (a device listed twice gets two copies)
function MultiGPUSLU(G1::GPUSLU, devices)
    devices = collect(CuDevice, devices)
    parts = replicas(G1, devices)
    return MultiGPUSLU(devices, parts, G1.n)
end

# ===== copies of the solve structure =====

# the device holding G
solver_device(G::GPUSLU) = CUDA.device(G.LDval)

# a copy of x on the current device, device to device (CUDA.jl copies on the source's stream of this
# task, peer to peer where the devices allow it; the first use of the copy waits for it)
function copy_here(x::CuArray{T, N}) where {T, N}
    y = CuArray{T, N}(undef, size(x))
    isempty(x) || copyto!(y, x)
    return y
end

#
# CUDA.jl allocates from each device's stream-ordered memory pool, whose memory other devices cannot
# access even with peer access enabled (CUDA.jl leaves the pools private), so the driver stages every
# copy out of it: 37 GB/s between two B200s on NVLink, 700 GB/s once the other device may read the pool.
# So the pool of the device that holds the solve structure is opened to the devices that copy from it,
# where the hardware allows (once per pair, for the process; the pool's later allocations included).
#
const POOL_PEERS = Set{Tuple{CuDevice, CuDevice}}()
const POOL_PEERS_LOCK = ReentrantLock()

function share_pool!(src::CuDevice, devices)
    CUDA.CUDACore.stream_ordered(src) || return

    lock(POOL_PEERS_LOCK) do
        for d in unique(devices)
            (d == src || (src, d) in POOL_PEERS) && continue
            push!(POOL_PEERS, (src, d))
            CUDA.can_access_peer(d, src) || continue
            CUDA.CUDACore.maybe_enable_peer_access(d, src) == 1 || continue     # (d reads src)
            CUDA.CUDACore.access!(CUDA.CUDACore.memory_pool(src), d, CUDA.CUDACore.ACCESS_FLAGS_PROT_READWRITE)
        end
    end

    return
end

# G's device arrays, which the copies read from other tasks and streams
device_arrays(G::GPUSLU) = (G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval, G.UDval, G.ULval, G.up, G.down,
    G.idx, G.pnt, G.istop, G.top, G.cinvp, G.rperm)

#
# Make G ready to be copied from other tasks while its own solve runs: its pending work done, the
# schedules every copy needs built once (on the host, here), and a snapshot of what the copies read
# (G's cache is a Dict that its solve writes to, so the copies must not read it). G is read-only from
# here on (as in a solve), so its arrays are taken off CUDA.jl's cross-stream tracking: otherwise every
# task that reads one takes it over, and the next reader (another copy, or G's own solve) waits for it.
# Returns the snapshot, for replicate.
#
function share_structure!(G::GPUSLU)
    return CUDA.device!(solver_device(G)) do
        sweep_plan(G); top_columns(G); first_descendants(G); factor_fill(G)
        get!(() -> Dict{Any, Any}(), G.cache, :mapped)             # (made by the solve otherwise: see mapped_cache)
        CUDA.synchronize()
        ops = G.ops[]
        sp = G.cache[:sweep]
        arrays = (device_arrays(G)..., (isnothing(ops) ? () : (ops.KL, ops.KU))...,
            G.cache[:topcols], G.cache[:firstdesc], (first(t) for t in sp.toplevels)...)
        foreach(x -> CUDA.enable_synchronization!(x, false), arrays)
        (; G, ops, host = solve_host(G), hrperm = host_rperm(G), fill = G.cache[:fill], topcols = G.cache[:topcols],
           firstdesc = G.cache[:firstdesc], sweep = sp, opscols = haskey(G.cache, :opscols))
    end
end

function replicas(G1::GPUSLU, devices::Vector{CuDevice}, snap = share_structure!(G1))
    src = solver_device(G1)
    share_pool!(src, devices)
    own = findfirst(==(src), devices)               # (G1 itself serves one block on its device)
    parts = Vector{Any}(undef, length(devices))

    @sync for g in eachindex(devices)
        Threads.@spawn begin
            CUDA.device!(devices[g])
            parts[g] = g == own ? G1 : replicate(snap)
        end
    end

    return [p for p in parts]
end

#
# A copy of snap.G (share_structure!) on the current device: its device arrays copied device to device,
# its host arrays shared (nothing modifies them), and the dense operators of its large fronts and its
# sweep schedules copied (each device keeps its own caches from here on).
#
function replicate(snap::NamedTuple)
    G = snap.G
    return replicate(G, snap)
end

function replicate(G::GPUSLU{Sem, T, I}, snap::NamedTuple) where {Sem, T, I}
    c = copy_here
    ops = snap.ops
    cache = Dict{Symbol, Any}(:host => snap.host, :hrperm => snap.hrperm, :fill => snap.fill, :mapped => Dict{Any, Any}(),
        :topcols => c(snap.topcols), :firstdesc => c(snap.firstdesc),
        :sweep => SweepPlan{I}([(c(small), large) for (small, large) in snap.sweep.toplevels], snap.sweep.nrest, snap.sweep.restwork))
    rops = nothing

    if !isnothing(ops)
        cols, dcols, ranges = ops_columns(G, ops, snap.host.Stgt)
        rops = SolveOps{T}(c(ops.KL), c(ops.KU), ops.off, Ref{CuVector{T}}(CuVector{T}(undef, 1)), cols)
        snap.opscols && (cache[:opscols] = (dcols, ranges))
    end

    return GPUSLU{Sem, T, I}(G.s, G.n, G.nf, c(G.Rptr), c(G.Sptr), c(G.Stgt), c(G.Dptr), c(G.Lptr), G.hRptr, G.hSptr, G.hDptr, G.hLptr,
        c(G.LDval), c(G.LLval), c(G.UDval), c(G.ULval), c(G.up), G.upptr, G.uplarge, c(G.down), G.downptr, G.downlarge, G.maxna,
        c(G.idx), c(G.pnt), c(G.istop), c(G.top), G.topptr, G.toplarge, c(G.cinvp), c(G.rperm), Ref{Any}(rops), cache)
end

# the columns [res; sep] of every large front of the operators (precompute_ops!'s "columns" step), uploaded
# here: (front → its columns, all of them, front → their range)
function ops_columns(G::GPUSLU, ops::SolveOps, hStgt::AbstractVector)
    fronts = sort!(collect(keys(ops.off)))
    ptr = ones(Int, length(fronts) + 1)

    for (i, f) in enumerate(fronts)
        ptr[i + 1] = ptr[i] + Int(G.hRptr[f + 1] - G.hRptr[f]) + Int(G.hSptr[f + 1] - G.hSptr[f])
    end

    hcols = Vector{Int}(undef, ptr[end] - 1)

    for (i, f) in enumerate(fronts)
        k = ptr[i]
        for v in Int(G.hRptr[f]):(Int(G.hRptr[f + 1]) - 1); hcols[k] = v; k += 1; end
        for e in Int(G.hSptr[f]):(Int(G.hSptr[f + 1]) - 1); hcols[k] = hStgt[e]; k += 1; end
    end

    dcols = upload(hcols)
    cols = Dict{Int, CuVector{Int}}(f => view(dcols, ptr[i]:(ptr[i + 1] - 1)) for (i, f) in enumerate(fronts))
    return cols, dcols, Dict(f => ptr[i]:(ptr[i + 1] - 1) for (i, f) in enumerate(fronts))
end

# ===== the L sweep's plans, once for all devices =====
#
# The plans of the L sweep below the top (layer plan, slot plan: layered.jl) are made on the host from
# the structure alone, and cost the host ~0.3–0.5 µs per front; every device's solve needs the same
# ones (same structure, same kernel, the same number of rows to within one). They are made once, on a
# task (slot_plan_ahead's), for the primary's solve; each other device's solve gets a task that copies
# them when they are done (sssp_gpu!'s `ahead`, adopted when its L sweep reaches them). The tasks
# return nothing when the plans are not used or not made ahead; the solves then make their own.
#
function shared_plans(G1::GPUSLU, W1::Union{CuMatrix, ColMapped}, zr)
    t = CUDA.device!(() -> slot_plan_ahead(G1, W1, Val(:N), zr), solver_device(G1))
    isnothing(t) && return nothing

    return Threads.@spawn begin
        H = fetch(t)
        isnothing(H) || CUDA.device!(solver_device(G1)) do
            CUDA.synchronize()                      # (slot_plan_ahead's uploads, on this task's stream)
            foreach(x -> CUDA.enable_synchronization!(x, false), plan_arrays(H.cache))
        end
        H
    end
end

isplan(v) = v isa LayerPlan || v isa SlotPlan

function plan_arrays(cache::AbstractDict)
    out = CuArray[]

    for v in values(cache)
        if v isa LayerPlan
            foreach(((a, b, _),) -> push!(out, a, b), v.layers)
        elseif v isa SlotPlan
            foreach(((a, _),) -> push!(out, a), v.layers)
            push!(out, v.hdr, v.ent, v.cmap)
        end
    end

    return out
end

# the shared plans copied to the current device, as a cache to adopt (adopt_plan!)
function plans_here(shared::Task)
    H = fetch(shared)
    isnothing(H) && return nothing
    c = copy_here
    out = Dict{Symbol, Any}()

    for (k, v) in H.cache
        if v isa LayerPlan
            out[k] = typeof(v)([(c(a), c(b), m) for (a, b, m) in v.layers], v.m, v.maxchain, v.hlayers)       # (hlayers: host arrays, shared)
        elseif v isa SlotPlan
            out[k] = typeof(v)([(c(a), m) for (a, m) in v.layers], c(v.hdr), c(v.ent), c(v.cmap), copy_here_undef(v.coef),
                v.slots, v.rows, v.sread, v.gread)
        elseif isnothing(v) && startswith(String(k), "slots")
            out[k] = nothing                        # (the slot plan does not apply: the plain layered walk)
        end
    end

    return out
end

# a fresh array of x's size and type on the current device
copy_here_undef(x::CuArray{T, N}) where {T, N} = CuArray{T, N}(undef, size(x))

# ===== the closure of each device's rows =====

# contiguous row blocks, as equal as possible
function row_blocks(n::Integer, ng::Integer)
    q, r = divrem(n, ng)
    stops = cumsum([q + (g <= r) for g in 1:ng])
    return [(g == 1 ? 1 : stops[g - 1] + 1):stops[g] for g in 1:ng]
end

"""
    closure_multigpu!(MG; out = nothing, timer = nothing, labels = :elimination) -> Vector{Tuple{UnitRange{Int}, CuMatrix}}

The closure split by rows over `MG.devices`: block g is `D[rows_g, :]` on device g. With
`labels = :elimination` (default), in elimination coordinates as closure_gpu!: `D[i, j] = A*[p[i], p[j]]`,
`p = rperm`. With `labels = :original`, in the labels of A: `D[i, j] = A*[i, j]` (rows_g are then
vertices; the columns are written in place, as apsp_gpu's mapped closure, or relabelled after).
`out` may hold preallocated blocks (one per device, of the right sizes, on their devices). With
`timer`, `timer[g]` is device g's wall time in seconds.
"""
function closure_multigpu!(MG::MultiGPUSLU; out = nothing, timer = nothing, labels::Symbol = :elimination, buffer::Integer = 0)
    labels in (:elimination, :original) || throw(ArgumentError("closure_multigpu!: labels must be :elimination or :original"))
    ng = length(MG.devices)
    rows = row_blocks(MG.n, ng)
    T = eltype(MG.parts[1].LDval)
    blocks = Vector{Any}(undef, ng)
    G1 = MG.parts[1]
    mapped = labels === :original && MAPPED_CLOSURE[] && factor_fill(G1) <= MAPPED_MAX_FILL[] * MG.n
    shared = nothing

    if ng > 1 && config().plan_overlap && length(rows[1]) > 0
        X1 = isnothing(out) ? nothing : out[1]
        W1 = CUDA.device!(MG.devices[1]) do
            X = something(X1, CuMatrix{T}(undef, length(rows[1]), 0))       # (only its type and rows matter)
            mapped ? ColMapped(X, G1.rperm) : X
        end
        zr = CUDA.device!(() -> CuVector{Int}(labels === :original ? rows[1] : 1:length(rows[1])), MG.devices[1])
        shared = rowmajor_path(W1) && config().skip_fill ? shared_plans(G1, W1, zr) : shared_plans(G1, W1, nothing)
    end

    walls = zeros(ng)          # (each task its own slot: timer, e.g. a Dict, is written after the tasks, on this one)

    @sync for g in 1:ng
        Threads.@spawn begin
            CUDA.device!(MG.devices[g])
            t0 = time()
            X = isnothing(out) ? nothing : out[g]
            k = length(rows[g])
            X = closure_rows!(MG.parts[g], rows[g], X, labels, mapped, buffer; ahead = plans_for(shared, g), inplace = !isnothing(out))
            walls[g] = time() - t0
            blocks[g] = (rows[g], X)
        end
    end

    isnothing(timer) || foreach(g -> (timer[g] = walls[g]), 1:ng)

    return [b for b in blocks]
end

# the plans for device g's solve: the shared task itself on the first device, a task copying its plans on the others
plans_for(::Nothing, g) = :auto
plans_for(shared::Task, g) = g == 1 ? shared : (dev = CUDA.device(); Threads.@spawn (CUDA.device!(dev); plans_here(shared)))

#
#   X ← D[rows, :] on the current device (allocated if nothing), in elimination coordinates
#   (D[i, j] = A*[p[i], p[j]]) or in the labels of A (labels = :original: rows are vertices; the
#   columns written in place when mapped, else relabelled after the solve: in place when `inplace`, as
#   for a block of the caller's, else possibly into a new matrix, X then freed)
#
function closure_rows!(G::GPUSLU{<:Any, T}, rows::UnitRange{Int}, X, labels::Symbol, mapped::Bool, buffer::Integer;
        ahead = :auto, release = nothing, inplace::Bool = false) where {T}
    n = G.n
    k = length(rows)

    own = isnothing(X)

    if own
        need_memory(k * n * sizeof(T), "a $k × $n closure block")
        X = CuMatrix{T}(undef, k, n)
    end

    @assert size(X) == (k, n)

    if k > 0
        sources = CuVector{Int}(labels === :original ? rows : host_rperm(G)[rows])
        M = CuMatrix{T}(undef, k, G.maxna)
        W = mapped ? ColMapped(X, G.rperm) : X

        try
            order = labels === :original ? row_order(G, rows) : RowOrder(rows, 1:k)      # (toprows.jl)
            sssp_gpu!(W, G, sources; W, M, permute = false, ahead, order)
            CUDA.synchronize()
        catch
            own && CUDA.unsafe_free!(X)             # (a block of the caller's is the caller's to free)
            rethrow()
        finally
            CUDA.unsafe_free!(M)
        end
    end

    isnothing(release) || release()

    if k > 0 && labels === :original && !mapped
        X = relabel_cols(X, G.cinvp, buffer; inplace)
        CUDA.synchronize()
    end

    return X
end

# ===== apsp_gpu on several devices =====
#
#   D[rows_g, :] ← A*[rows_g, :]   on devices[g], rows_g a contiguous block of vertices (labels of A)
#
# The pipeline of apsp_single, with the closure split: every device allocates its block while the host
# orders the graph; the primary (devices[1]) factorizes and builds the solve structure; the others get
# copies of it device to device, and all solve their rows concurrently, each on its own task.
#
# MGPU_TIMES[]: a Dict to record the call's timeline (seconds from its start: "factor", "setup",
# and per device g ("replica", g), ("closure", g), ("relabel", g), ("done", g)); off when nothing.
const MGPU_TIMES = Ref{Any}(nothing)

function apsp_multigpu(A::SparseMatrixCSC{T}, s::AbstractSemiring, devices::Vector{CuDevice}, output::Symbol, buffer::Integer) where {T}
    n = size(A, 1)
    ng = length(devices)
    iszero(n) && return output === :host ? Matrix{T}(undef, 0, 0) : Tuple{Vector{Int}, CuMatrix{T}}[]
    rows = row_blocks(n, ng)
    #
    # every device must hold its blocks (a repeated device holds several)
    #
    for d in unique(devices)
        k = sum(length(rows[g]) for g in 1:ng if devices[g] == d)

        CUDA.device!(d) do
            need_apsp_memory(k * n * sizeof(T), "$k rows of the $n × $n closure", "use more GPUs or compute blocks of rows (apsp_gpu(A, sources))")
        end
    end

    tl = MGPU_TIMES[]; t0 = time_ns(); tlock = ReentrantLock()
    mark!(key) = (isnothing(tl) || (t = (time_ns() - t0) / 1e9; lock(() -> (tl[key] = t), tlock)); nothing)
    H = output === :host ? host_matrix(T, n, n) : nothing
    blocks = [CUDA.device!(() -> allocate_async(T, length(rows[g]), n), devices[g]) for g in 1:ng]   # (meanwhile)
    pdev = first(devices)

    return with_host_buffers() do B
        P = nothing; G1 = nothing
        parts = Vector{Any}(nothing, ng)                # (what a failed call frees: replicas, result blocks)
        out = Vector{Any}(nothing, ng)

        try
            P, st = CUDA.device!(() -> apsp_factor(s, A; host = B, solve = 8192), pdev)
            mark!("factor")
            G1 = CUDA.device!(pdev) do
                G = GPUSLU(P; large = 8192, structure = structure_of(st))
                precompute_ops!(G)
                G
            end
            snap = share_structure!(G1)
            share_pool!(pdev, devices)
            mark!("setup")
            mapped = MAPPED_CLOSURE[] && factor_fill(G1) <= MAPPED_MAX_FILL[] * n
            CUDA.device!(pdev) do
                need_apsp_memory(length(rows[1]) * G1.maxna * sizeof(T), "the workspace of the closure", "use more GPUs")
            end
            #
            # the L sweep's plans, made once (the primary's rows stand for all: their number differs by at most one)
            #
            X1 = fetch_result(blocks[1])
            zr = CUDA.device!(() -> CuVector{Int}(rows[1]), pdev)
            W1 = mapped ? ColMapped(X1, G1.rperm) : X1
            shared = isempty(rows[1]) ? nothing : shared_plans(G1, W1, config().skip_fill && rowmajor_path(W1) ? zr : nothing)

            @sync for g in 1:ng
                Threads.@spawn begin
                    CUDA.device!(devices[g])
                    G = g == 1 ? G1 : replicate(snap)
                    g == 1 || (parts[g] = G)
                    mark!(("replica", g))
                    X = fetch_result(blocks[g])
                    release = g == 1 ? nothing : () -> free_solver!(G)
                    X = closure_rows!(G, rows[g], X, :original, mapped, buffer; ahead = plans_for(shared, g), release)
                    out[g] = X
                    mark!(("closure", g))

                    if !isnothing(H)
                        isempty(rows[g]) || scatter_rows!(H, collect(rows[g]), X, relabel_length(T, length(rows[g]), n, buffer))
                        CUDA.unsafe_free!(X)
                    end

                    mark!(("done", g))
                end
            end

            CUDA.device!(pdev) do
                free_solver!(G1); free_plan!(P)
            end

            isnothing(H) || return H
            return [(collect(rows[g]), out[g]) for g in 1:ng]
        catch
            #
            # free what the call made at once (a failure on one device must not leave the others' blocks,
            # replicas or the factor to the garbage collector: the next call would find the memory taken)
            #
            for g in 1:ng
                CUDA.device!(devices[g]) do
                    isnothing(out[g]) || CUDA.unsafe_free!(out[g])
                    isnothing(parts[g]) || free_solver!(parts[g])
                    discard(blocks[g])
                end
            end

            CUDA.device!(pdev) do
                isnothing(G1) || free_solver!(G1)
                isnothing(P) || free_plan!(P)
            end

            rethrow()
        end
    end
end
