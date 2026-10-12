# The one-GPU call of the benchmarks (apsp_gpu!(D, A), code ours-auto of gpu_apsp.jl: Float32, ordering auto, hybrid
# factorization, the output preallocated and first touched before the timer), untimed: no step timer, so no
# synchronization inside the call. Every @step of every task is recorded on the host as it runs (PATHPACK's trace hook,
# X.TRACE: the thread, the start and the end, from time_ns), and the GPU's work is placed with CUDA events.
#
#   plain   apsp_gpu!(D, A) itself, host trace only (the call as gpu_apsp.jl times it)
#   events  the same call with CUDA events recorded on its stream between its steps: apsp_gpu!'s own code (api.jl,
#           sgetrs.jl at the library commit), re-issued here step by step with the library's functions, plus an event
#           where the closure turns from X <- U* to X <- XL*, and a host-trace step around the solve structure's upload.
#           Each event is recorded right after the work it closes is issued, on the same stream, so it completes
#           when the GPU has finished that work; the first is recorded with the GPU idle, at the call's start.
#           The result is checked bitwise against the plain call's.
#   timed   the same call under PATHPACK's step timer (with_phases, as gpu_phases.jl: every step synchronized), its phases
#           and steps; and the ordering alone (as gpu_phases.jl's time_ordering), for the timed call's split of its symbolic phase.
#           All three sets in one process, interleaved, so that they share the node's conditions.
# Cold calls as gpu_lib.jl (pool returned to the driver, pinned arena freed, malloc_trim), wall time with the device
# synchronized before and after. Writes OUT/trace.csv (every call, every step) and OUT/summary.txt.
#   julia -t 8,0 --project=ENV measure.jl MTX OUT [reps] [commit]
using CUDA, PATHPACK, SparseArrays, Statistics, Printf
const X = Base.get_extension(PATHPACK, :PATHPACKCUDAExt)
const GP = PATHPACK.GPU
const CPU = PATHPACK.CPU
const MTX, OUT = ARGS[1], ARGS[2]
const REPS = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 11
const COMMIT = length(ARGS) > 3 ? ARGS[4] : "unknown"
mkpath(OUT)
say(x...) = (println(x...); flush(stdout))

# a package image precompiled with a GPU holds a stream of the precompiling process in SIDE_STREAMS (reset_caches! does
# not clear it), whose first use segfaults: dropped in this process when present (the library is not changed)
const STALE_STREAMS = length(X.SIDE_STREAMS)
STALE_STREAMS > 0 && empty!(X.SIDE_STREAMS)

# ---------------- gpu_lib.jl's conventions (as written there) ----------------
syncall() = (for d in CUDA.devices(); CUDA.device!(() -> CUDA.device_synchronize(), d); end)
function cold!()
    GC.gc(true)
    for d in CUDA.devices(); CUDA.device!(() -> CUDA.reclaim(), d); end
    X.settle_arena()
    lock(X.ARENA_LOCK) do
        isnothing(X.ARENA.mem) || X.CC.free(X.ARENA.mem)
        X.ARENA.mem = nothing; X.ARENA.size = 0
    end
    ccall(:malloc_trim, Cint, (Csize_t,), 0)
    return
end
function read_mtx_values(path)                  # graphs.jl's reader: |a_ij| (pattern: 1; explicit zeros: 1), symmetric, min over duplicates
    W = Dict{Tuple{Int, Int}, Float64}(); n = 0
    open(path) do io
        pattern = occursin("pattern", lowercase(readline(io)))
        line = readline(io); while startswith(line, "%"); line = readline(io); end
        n = parse(Int, split(line)[1])
        for l in eachline(io)
            isempty(l) && continue
            t = split(l); i = parse(Int, t[1]); j = parse(Int, t[2]); i == j && continue
            w = pattern ? 1.0 : abs(parse(Float64, t[3])); w == 0 && (w = 1.0)
            key = (min(i, j), max(i, j)); W[key] = min(get(W, key, Inf), w)
        end
    end
    I = Int[]; J = Int[]; V = Float64[]
    for ((i, j), w) in W; push!(I, i, j); push!(J, j, i); push!(V, w, w); end
    return sparse(I, J, V, n, n)
end

# ---------------- the ordering alone (gpu_phases.jl's order_graphs and time_ordering, as written there) ----------------
const PC = PATHPACK.CPU
function order_graphs(A::SparseMatrixCSC{<:Any, I}) where {I}
    n = I(size(A, 2)); ptr = PC.getcolptr(A)
    whole = PC.BipartiteGraph{I, I}(n, n, ptr[n + one(I)] - one(I), ptr, PC.rowvals(A))
    sym, comps = PC.psymmetric_components(whole)
    if sym
        Bptr, invp, nBptr = comps
    else
        scc = PC.sccs(A); nBptr = PC.nv(scc); Bptr = PC.pointers(scc); invp = PC.targets(scc); PC.stable_components!(invp, Bptr, nBptr)
    end
    big = [c for c in 1:Int(nBptr) if Bptr[c + 1] - Bptr[c] > 2 || isone(nBptr)]
    graphs = Dict{Int, Any}()
    if isone(nBptr)
        graphs[1] = sym || PC.psymmetric_pattern(whole) ? whole : PC.symmetric(whole, 'N')
    elseif sym
        loc = Vector{I}(undef, n)
        for c in 1:Int(nBptr), a in Bptr[c]:(Bptr[c + 1] - 1); loc[invp[a]] = a - Bptr[c] + one(I); end
        for c in big; graphs[c] = PC.componentgraph(A, invp, loc, Bptr, I(c)); end
    else
        g = PC.permute_pattern(A, invp, invp)
        for c in big; sub = PC.subgraph(g, Bptr[c], Bptr[c + 1] - one(I)); graphs[c] = PC.psymmetric_pattern(sub) ? sub : PC.symmetric(sub, 'N'); end
    end
    return graphs, big, Bptr, Int(nBptr)
end
function time_ordering(A, alg)
    graphs, big, Bptr, ncomp = order_graphs(A)
    out = Dict{Int, Any}(); lk = ReentrantLock()
    t = @elapsed PC.eachcomponent(c -> (r = PC.fastpermutation(PC.CTR.Ones{Int}(PC.nv(graphs[c])), graphs[c], alg); lock(() -> (out[c] = r), lk)), big, Bptr)
    return t, ncomp, length(big)
end

# ---------------- the host trace (PATHPACK's TRACE hook: what with_trace sets, kept here in raw time_ns) ----------------
function traced(f)
    tr = Tuple{String, Int, UInt64, UInt64}[]
    X.TRACE[] = tr
    try
        r = f()
        return r, tr
    finally
        X.TRACE[] = nothing
    end
end
# a host interval of our own in the same trace (the structure's upload, which the library does not mark)
function mark!(name, t0)
    tr = X.TRACE[]
    isnothing(tr) || lock(() -> push!(tr, (String(name), Threads.threadid(), t0, time_ns())), X.TRACE_LOCK)
    return
end

# ---------------- apsp_gpu!(D, A) with CUDA events between its steps ----------------
mutable struct Events
    ev::Vector{Tuple{Symbol, CuEvent, UInt64}}          # (name, event, host time_ns when recorded)
end
Events() = Events(Tuple{Symbol, CuEvent, UInt64}[])
rec!(E::Events, name::Symbol) = (e = CuEvent(); CUDA.record(e); push!(E.ev, (name, e, time_ns())); nothing)

# api.jl:895-905, with the upload marked on the host
function structure_async_ev(F, large::Integer)
    dev = CUDA.device()
    return Threads.@spawn begin
        CUDA.device!(dev)
        st = X.solve_structure(F; large)
        t0 = time_ns()
        st = merge(st, (; dev = X.upload_structure(st, F)))
        CUDA.synchronize()
        mark!("structure upload", t0)
        st
    end
end

# api.jl:872-893 (host = the call's host buffers, solve = 8192)
function apsp_factor_ev(s, A::SparseMatrixCSC{T}, host) where {T}
    Q, S = X.@step "symbolic" CPU.ssymbolic(A; alg = X.elimination_algorithm())
    F = X.@step "factor storage" X.host_slu(host, s, T, S, Q)
    X.check_coupling(F)
    st = structure_async_ev(F, 8192)
    P = try
        X.factor_plan(F, A)                             # (factor_overlapped, api.jl:947-984)
    catch
        st isa Task && try wait(st) catch end
        rethrow()
    end
    return P, st
end

# sgetrs.jl:388-451 (sssp_gpu!), the event :U_done between the U sweep and the L sweep
function sssp_ev!(Xm, G, sources, E; W, M, nthreads::Int = 64, permute::Bool = true, ahead = :auto, order = nothing)
    @assert size(Xm) == (length(sources), G.n)
    @assert size(W) == size(Xm)
    @assert permute || W === Xm
    s = G.s
    trans = Val(:N)
    scale = Val(!CPU.isintegral(s))
    nrhs = size(Xm, 1)
    tb = min(nthreads, 32 * cld(nrhs, 32))
    skip = X.config().skip_fill && X.rowmajor_path(W)
    ahead === :auto && (ahead = X.slot_plan_ahead(G, W, trans, skip ? sources : nothing))
    if skip
        X.fill_top!(W, G)
    else
        fill!(W, CPU.szero(s, eltype(W), trans))
    end
    @cuda threads = 128 blocks = cld(32 * nrhs, 128) X.upward_path_warp_kernel!(s, trans, scale, W, sources, G.cinvp, G.idx, G.pnt, G.istop,
        G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval, Val(skip))
    tr = X.top_rows_on(G, order) ? X.top_rows(G, order) : nothing
    isnothing(tr) || X.reserve_rows_workspace!(G, eltype(W), tr)
    kernel = @cuda launch = false X.upward_kernel!(s, trans, scale, W, G.top, 0, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval)
    for l in 1:(length(G.topptr) - 1)
        strt = G.topptr[l]
        nfl = G.topptr[l + 1] - strt
        if X.ispositive(nfl) && isnothing(tr)
            kernel(s, trans, scale, W, G.top, strt - 1, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval; threads = tb, blocks = (nfl, X.rhs_blocks(nrhs, tb)))
        elseif X.ispositive(nfl) && X.ispositive(tr.maxlen[l])
            @cuda threads = tb blocks = (nfl, X.rhs_blocks(tr.maxlen[l], tb)) X.upward_toprows_kernel!(s, trans, scale, W, G.top, strt - 1, tr.rg, X.batched_rows(tr),
                G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval)
        end
        if isnothing(tr)
            for f in G.toplarge[l]
                X.upward_large!(G, W, M, f, trans, scale)
            end
        else
            X.upward_top_level!(G, W, M, G.toplarge[l], trans, scale, tr)
        end
    end
    rec!(E, :U_done)
    X.downward_sweep!(G, W, M, trans, tb, X.rhs_blocks(nrhs, tb), nothing; zr = skip ? sources : nothing, ahead, tr)
    permute && X.permutecols_gpu!(Xm, W, G.rperm)
    return Xm
end

# api.jl:254-280 (closure_labels with the caller's D)
function closure_labels_ev(G, buffer::Integer, release, D::CuMatrix{T}, E) where {T}
    n = G.n
    D, M = X.@step "allocate" (D, CuMatrix{T}(undef, n, G.maxna))
    mapped = X.MAPPED_CLOSURE[] && X.factor_fill(G) <= X.MAPPED_MAX_FILL[] * n
    W = mapped ? X.ColMapped(D, G.rperm) : D
    rec!(E, :closure_start)
    X.@step "closure" sssp_ev!(W, G, X.upload_side(collect(1:n)), E; W, M, permute = false, order = X.RowOrder(1:n, G.rperm))
    rec!(E, :closure_done)
    X.RELABEL_AHEAD[] || CUDA.synchronize()
    X.@step "free" (CUDA.unsafe_free!(M); release())
    mapped && return D
    hq = X.RELABEL_AHEAD[] ? get(G.cache, :hcinvp, nothing) : nothing
    D = X.@step "relabel columns" X.relabel_cols(D, G.cinvp, buffer; hq, inplace = true)
    rec!(E, :relabel_done)
    return D
end

# api.jl:606-610 (apsp_gpu!(D, A)) and 158-198 (apsp_single, output = :device, D given)
function call_events!(D::CuMatrix{T}, A::SparseMatrixCSC{T}, E) where {T}
    s = CPU.MinPlus()
    rec!(E, :start)                                     # (the GPU is idle: completes at once)
    X.check_apsp(A, s, :device)
    return X.without_early_gc() do
        n = size(A, 1)
        advice = "split the rows over several GPUs (devices = [...]) or compute blocks of rows (apsp_gpu(A, sources))"
        X.@step "host matrix" nothing
        X.with_host_buffers() do B
            P, st = apsp_factor_ev(s, A, B)
            rec!(E, :top_done)                          # (factor_top! is the last work issued: done with the top)
            G = X.@step "solve setup" X.GPUSLU(P; large = 8192, structure = X.structure_of(st))
            rec!(E, :merged_done)
            X.top_rows_ahead!(G, X.RowOrder(1:n, G.rperm))
            X.@step "operators" X.precompute_ops!(G)
            rec!(E, :operators_done)
            X.@step "memory check" X.need_apsp_memory(n * G.maxna * sizeof(T), "the workspace of the $n × $n closure", advice)
            release = () -> (X.free_solver!(G); X.free_plan!(P))
            closure_labels_ev(G, 0, release, D, E)
        end
    end
end

# ---------------- the runs ----------------
A0 = read_mtx_values(MTX); A = SparseMatrixCSC{Float32, Int}(A0); n = size(A, 1)
dev = CUDA.device()
say(@sprintf("graph %s n=%d nnz=%d | %s | threads %d (+%d interactive) | stale side streams dropped: %d | commit %s",
    basename(MTX), n, nnz(A), CUDA.name(dev), Threads.nthreads(), Threads.nthreads(:interactive), STALE_STREAMS, COMMIT))
D = CuMatrix{Float32}(undef, n, n); fill!(D, 0f0); syncall()           # (preallocated and first touched, outside the timer)
plain() = GP.apsp_gpu!(D, A; alg = nothing)                           # (closure! of gpu_lib.jl, ordering auto)
function timed_call(f)                                                # gpu_lib.jl's timed: device idle before and after
    syncall(); GC.gc(false)
    syncall(); t0 = time_ns(); r = f(); syncall(); t1 = time_ns()
    return r, t0, t1
end

t_first = @elapsed (plain(); syncall())                               # (compiles every path the graph needs)
plain(); syncall()
call_events!(D, A, Events()); syncall()                              # (compiles the event path)
X.with_phases(plain); syncall()                                       # (compiles the timed path)
time_ordering(A, X.elimination_algorithm())                           # (compiles the ordering alone)
say(@sprintf("first call %.2f s", t_first))
Dref = copy(D); syncall()                                             # (the plain call's result, for the bitwise check)

calls = []                                                            # (set, call, total ms, host trace in ms, events in ms)
totals = Dict("plain" => Float64[], "events" => Float64[])
same = Bool[]
const TREPS = min(REPS, 7)                                            # timed calls (gpu_phases.jl: 7) and orderings alone
timed_calls = []; orders = Float64[]
for r in 1:REPS, set in ("plain", "events", "timed", "ordering")
    if set == "timed"
        r <= TREPS || continue
        cold!(); syncall()
        _, ph = X.with_phases(plain); syncall()
        push!(timed_calls, ph)
        say(@sprintf("timed  call %2d: %.1f ms (sum of its steps)", length(timed_calls), 1e3 * ph.total))
        continue
    elseif set == "ordering"
        r <= TREPS || continue
        cold!()
        push!(orders, time_ordering(A, X.elimination_algorithm())[1])
        continue
    end
    cold!()
    if set == "plain"
        (_, t0, t1), tr = traced(() -> timed_call(plain))
        ev = nothing
    else
        Ec = Events()
        (_, t0, t1), tr = traced(() -> timed_call(() -> call_events!(D, A, Ec)))
        push!(same, (syncall(); all(reinterpret(UInt32, D) .== reinterpret(UInt32, Dref))))
        ev = Ec
    end
    tot = (t1 - t0) / 1e6; push!(totals[set], tot)
    ms(t) = (Int64(t) - Int64(t0)) / 1e6
    steps = [(name, tid, ms(a), ms(b)) for (name, tid, a, b) in tr]
    evs = Tuple{Symbol, Float64, Float64}[]                           # (name, host time recorded, GPU time completed)
    if !isnothing(ev)
        e0 = ev.ev[1][2]; h0 = ms(ev.ev[1][3])                      # (:start completes when recorded: the GPU was idle)
        for (name, e, h) in ev.ev
            push!(evs, (name, ms(h), h0 + 1e3 * CUDA.elapsed(e0, e)))
        end
    end
    push!(calls, (set, length(totals[set]), tot, steps, evs))
    say(@sprintf("%-6s call %2d: %.1f ms%s", set, length(totals[set]), tot, isnothing(ev) ? "" : (same[end] ? "  (result bitwise equal to the plain call's)" : "  RESULT DIFFERS")))
end


# ---------------- bytes of the copies (the same symbolic factorization, outside the measurements) ----------------
Q, S = CPU.ssymbolic(A; alg = X.elimination_algorithm())
F = CPU.ChordalSLU(CPU.MinPlus(), Float32, S, Q.perm, Q.invp, Q.perm, Q.invp)
B = X.bottom_plan(F; large = 256, nt = Threads.nthreads())
spans = X.bottom_spans(F, B.workers)
SS = F.S.S; Dp = SS.Dptr; Lp = SS.Lptr; nf = Int(X.MF.nv(SS.res)); nsep = Int(X.MF.pointers(SS.sep)[nf + 1]) - 1; m = nnz(A)
bD = sum(Int(Dp[hi + 1]) - Int(Dp[lo]) for (lo, hi) in spans; init = 0); bL = sum(Int(Lp[hi + 1]) - Int(Lp[lo]) for (lo, hi) in spans; init = 0)
nD = length(F.LDval); nL = length(F.LLval)
bytes_bottom = 4 * (2bD + 2bL + B.nbndval)                            # upload_bottom!: the four arrays' bottom spans, the boundary updates
bytes_whole = 4 * (2nD + 2nL + B.nbndval)                             # upload! (the timed path): the whole factor, the updates
bytes_atop = 4 * (n + 1) + 4m + 4m + 3 * 4n + 4 * 8 * (nf + 1) + 4nsep + 8nf + nf     # device_entries!'s uploads (sgetrf.jl:821-825)
st = X.solve_structure(F; large = 8192)
bytes_struct = sum(sizeof, (st.Rptr, st.Sptr, st.Stgt, st.Dptr, st.Lptr, st.up, st.down, st.idx, st.pnt, st.top)) + length(st.istop) +
    sizeof(F.cinvp) + sizeof(F.rperm)                                 # upload_structure (sgetrs.jl:166-170)
P0 = X.FactorPlan(F, B; graph = false, nstreams = 8)
launches = X.use_batched_top(P0) ? length(X.top_batch(P0).launches) : -1
levels = length(P0.levels); X.free_plan!(P0)
# every call, every step: host steps (lane cpu, the task that runs them), the copies (lane pcie: the host steps that make
# them, which return when they are complete; bytes from the symbolic factorization), the GPU's work (lane gpu: from
# the events, events set only), and the events themselves (lane event: host time recorded, GPU time completed)
TASK = Dict("host matrix" => "main", "symbolic" => "main", "factor storage" => "main", "top selection" => "main", "bottom forest" => "main",
    "worker stacks" => "main", "device factor" => "main", "plan (top)" => "main", "top batch" => "main", "upload bottom" => "main",
    "top (GPU)" => "main", "solve setup" => "main", "operators" => "main", "memory check" => "main", "allocate" => "main",
    "closure" => "main", "free" => "main", "relabel columns" => "main", "copy entries" => "entries", "bottom (CPU)" => "subtrees",
    "upload factor" => "device entries", "structure" => "structure", "merge maps" => "structure", "levels" => "structure",
    "structure upload" => "structure")
PCIE = Dict("upload factor" => ("A's top entries and index arrays", bytes_atop), "upload bottom" => ("subtrees' factors and updates", bytes_bottom),
    "structure upload" => ("solve structure", bytes_struct))
GPU = [("top", nothing, :top_done), ("merged factor and operators", :top_done, :operators_done), ("X <- U*", :closure_start, :U_done),
    ("X <- XL*", :U_done, :closure_done), ("relabel", :closure_done, :relabel_done)]
gpuname = replace(CUDA.name(dev), "," => " "); hostname_ = gethostname(); graph = replace(basename(MTX), ".mtx" => "")
open(joinpath(OUT, "trace.csv"), "w") do io
    println(io, "graph,commit,gpu,host,set,call,call_ms,step,task,lane,start_ms,end_ms,bytes,source")
    row(c, step, task, lane, a, b, bytes, src) = println(io, join((graph, COMMIT, gpuname, hostname_, c[1], c[2], @sprintf("%.4f", c[3]), step, task, lane,
        @sprintf("%.4f", a), @sprintf("%.4f", b), bytes, src), ","))
    for c in calls
        set, k, tot, steps, evs = c
        for (name, tid, a, b) in steps
            row(c, name, get(TASK, name, "nested (thread $tid)"), "cpu", a, b, "", "host trace thread $tid")
            haskey(PCIE, name) && row(c, PCIE[name][1], get(TASK, name, ""), "pcie", a, b, PCIE[name][2], "host step $(name)")
        end
        isempty(evs) && continue
        at(nm) = only(x[3] for x in evs if x[1] == nm)
        issue = only(a for (name, tid, a, b) in steps if name == "top (GPU)")
        for (step, from, to) in GPU
            row(c, step, "gpu", "gpu", isnothing(from) ? issue : at(from), at(to), "", isnothing(from) ? "issued (host step top (GPU)) to event top_done" : "events $(from) to $(to)")
        end
        for (nm, h, g) in evs
            row(c, String(nm), "event", "event", h, g, "", "cuda event: recorded (host) and completed (gpu)")
        end
    end
    # the timed calls: each phase's and each step's time (with_phases; durations: start 0); the orderings alone
    for (k, ph) in enumerate(timed_calls)
        c = ("timed", k, 1e3 * ph.total)
        for p in (:symbolic, :numeric_setup, :numeric_cpu, :transfer, :numeric_gpu, :solve_setup, :inverse, :solve, :other, :relabel)
            row(c, String(p), "phase", "phase", 0.0, 1e3 * getfield(ph, p), "", "with_phases: the phase's time")
        end
        for (path, v) in ph.steps
            row(c, path, "step", "step", 0.0, 1e3 * v, "", "with_phases: the step's time")
        end
    end
    for (k, t_) in enumerate(orders)
        row(("ordering", k, 1e3 * t_), "ordering", "phase", "phase", 0.0, 1e3 * t_, "", "the ordering alone (gpu_phases.jl's time_ordering)")
    end
end
open(joinpath(OUT, "summary.txt"), "w") do io
    med(v) = median(v)
    for (name, v) in (("plain", totals["plain"]), ("events", totals["events"]))
        println(io, @sprintf("%s: median %.2f ms, min %.2f, max %.2f, n %d, all %s", name, med(v), minimum(v), maximum(v), length(v), join(round.(v; digits = 2), " ")))
    end
    println(io, "events calls bitwise equal to the plain call: ", all(same), " (", count(same), "/", length(same), ")")
    isempty(timed_calls) || println(io, @sprintf("timed: median %.2f ms (sum of steps), n %d, all %s", median(1e3 .* [p.total for p in timed_calls]), length(timed_calls),
        join(round.(1e3 .* [p.total for p in timed_calls]; digits = 2), " ")))
    isempty(orders) || println(io, @sprintf("ordering alone: median %.2f ms, n %d", 1e3 * median(orders), length(orders)))
    println(io, @sprintf("bytes: subtrees' factors and updates %d; A's top entries and index arrays %d; solve structure %d; whole factor and updates (timed path) %d", bytes_bottom, bytes_atop, bytes_struct, bytes_whole))
    println(io, @sprintf("fronts %d, factorization top %d fronts, %d levels, %d launches (batched %s); stale side streams dropped %d; first call %.2f s",
        nf, count(B.istop), levels, launches, launches >= 0, STALE_STREAMS, t_first))
end
say(read(joinpath(OUT, "summary.txt"), String))
