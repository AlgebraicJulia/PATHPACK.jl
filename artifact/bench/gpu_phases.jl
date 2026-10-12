# The phase breakdown of the one-GPU closure, robust: per graph, BENCH_PHREPS (7) cold, synchronized with_phases calls of
# apsp_gpu!(D, A) (ordering auto, hybrid factorization, the output preallocated as in gpu_apsp.jl), after one untimed
# call (compilation). Each phase recorded as its MEDIAN over the calls (phases.*, the same keys as gpu_apsp.jl's single
# call, so readers of phases.* need no change; phases.total = the median of the calls' totals), with every call's value
# (phases_runs.*) and phases_reps. One row per graph, code ours-auto-phases (no timed calls: t_total stays gpu_apsp.jl's).
# Check: the output prefilled with NaN has none left after the last call; 64 columns against Dijkstra; with integer
# weights, the row hashes against the export's (written by gpu_apsp.jl). phases.ordering: the fill-reducing ordering
# alone, the median of BENCH_PHREPS cold calls (see time_ordering; phases.symbolic stays the whole symbolic analysis). Usage: julia -t T gpu_phases.jl GRAPH...
include("gpu_lib.jl")
const KIND = gpukind()
const PHREPS = parse(Int, get(ENV, "BENCH_PHREPS", "7"))
const CODE = "ours-auto-phases"
let A = SparseMatrixCSC{Float32, Int}(fromgraph(Graphs.grid([12, 12]))); n = size(A, 1); D = CuMatrix{Float32}(undef, n, n)   # compile the common paths
    closure!(D, A; alg = ORDERINGS["auto"]); phases_of(() -> closure!(D, A; alg = ORDERINGS["auto"])); CUDA.unsafe_free!(D); GC.gc()
end
function medphases(phs)
    out = Dict{String, Any}(); runs = Dict{String, Any}()
    for k in keys(phs[1])
        if k == "steps"
            sk = union((keys(p["steps"]) for p in phs)...)
            out[k] = Dict(String(s) => median([Float64(get(p["steps"], s, 0.0)) for p in phs]) for s in sk)
        elseif phs[1][k] isa Real
            v = [Float64(p[k]) for p in phs]; out[k] = median(v); runs[k] = v
        end
    end
    out, runs
end
# ---- the fill-reducing ordering alone (phases.ordering; phases.symbolic stays the whole symbolic analysis) ----
# CPU.ssymbolic(A; alg) as the closure calls it (alg = the library's elimination_algorithm(): AutoOrder by default)
# orders, per component of more than two vertices, the component's symmetric graph without loops with
# fastpermutation(Ones(nc), graph, alg) inside pcliquetree, the components concurrently (eachcomponent, largest
# first). Here the same graphs are built as ssymbolic builds them (psymmetric_components, or sccs + permute_pattern +
# symmetric when the pattern is not symmetric) outside the timer, and only the eachcomponent loop of fastpermutation
# calls is timed (components of one or two vertices, ordered once by cliquetree, are left out).
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
for name in ARGS
    isfile(gresultfile("gpuapsp", "$(name)_$(KIND)_$(CODE)")) && continue
    A = graph(name); n = size(A, 1); A32 = SparseMatrixCSC{Float32, Int}(A)
    if 4.0 * n^2 * parse(Float64, get(ENV, "BENCH_MEM_MARGIN", "1.1")) > CUDA.totalmem(CUDA.device())
        @printf "%-15s n=%d: too large for one GPU: skipped\n" name n
        gsave("gpuapsp", "$(name)_$(KIND)_$(CODE)", Dict("graph" => name, "code" => CODE, "n" => n, "m" => nnz(A), "status" => "too large for one GPU")); continue
    end
    cols = sample_vertices(n); R = ref_dijkstra(A, cols)
    exact = integral(A) && n * maximum(nonzeros(A); init = 0.0) < 2^24
    cold!()
    D, t_map, t_touch = alloc_output(Float32, n, n)
    call() = closure!(D, A32; alg = ORDERINGS["auto"])
    local phs, nnan, c_dij, hok
    try
        _, t_first = timed(call)
        fill!(D, NaN32); syncall()
        phs = [(cold!(); phases_of(call)) for _ in 1:PHREPS]
        syncall(); nnan = count(isnan, D)
        G = Array(D[:, cols]); c_dij = check_cols(G, R)
        hf = joinpath(EXPORT, "$name.rowhash"); hok = nothing
        exact && isfile(hf) && (hok = [string(x; base = 16, pad = 16) for x in rowhashes(D, 1:n)] == readlines(hf))
    catch err
        msg = sprint(showerror, err)[1:min(end, 300)]
        @printf "%-15s %s FAILED: %s\n" name CODE msg
        gsave("gpuapsp", "$(name)_$(KIND)_$(CODE)", Dict("graph" => name, "code" => CODE, "n" => n, "m" => nnz(A), "status" => "failed: " * msg))
        CUDA.unsafe_free!(D); D = nothing; GC.gc(); CUDA.reclaim(); continue
    end
    ok = nnan == 0 && c_dij.ok && hok !== false
    ph, runs = medphases(phs)
    oalg = X.elimination_algorithm()
    tord = [NaN]; ocomp = (0, 0); oerr = ""
    try
        time_ordering(A32, oalg)                                               # (compiles this graph's ordering path; excluded)
        tord = Float64[]
        for _ in 1:PHREPS; cold!(); t, nc, nb = time_ordering(A32, oalg); push!(tord, t); ocomp = (nc, nb); end
        ph["ordering"] = median(tord); runs["ordering"] = tord
    catch err
        oerr = sprint(showerror, err)[1:min(end, 300)]; ph["ordering"] = NaN
        @printf "%-15s ordering alone FAILED: %s\n" name oerr
    end
    gsave("gpuapsp", "$(name)_$(KIND)_$(CODE)", Dict("graph" => name, "code" => CODE, "status" => ok ? "ok" : "check failed", "n" => n, "m" => nnz(A),
        "ordering" => ordering_label("auto"), "auto_choice" => auto_choice(A), "factor" => "hybrid", "eltype" => "Float32",
        "output" => "device, preallocated (apsp_gpu!); original labels",
        "phases" => ph, "phases_reps" => PHREPS, "phases_runs" => runs, "phases_scope" => "median per phase over $PHREPS cold synchronized with_phases calls; ordering: median of $PHREPS cold calls of the ordering alone",
        "ordering_alg" => string(oalg), "ordering_components" => ocomp[1], "ordering_components_ordered" => ocomp[2], "ordering_error" => oerr,
        "ordering_scope" => "fastpermutation(Ones, graph, elimination_algorithm()) on each component's symmetric graph as CPU.ssymbolic builds it (built outside the timer), components of more than two vertices, concurrently as eachcomponent",
        "check" => Dict("ok" => ok, "nan_left" => nnan, "rtol" => RTOL, "columns" => length(cols), "dijkstra_ok" => c_dij.ok,
            "dijkstra_maxrel" => c_dij.maxrel, "exact" => exact, "rowhash_ok" => something(hok, "")),
        "t_alloc" => t_map + t_touch))
    p = ph; tot = runs["total"]
    @printf "%-15s n=%7d %s | median of %d: sym %.4f numset %.4f cpu %.4f gpu %.4f xfer %.4f setup %.4f inv %.4f solve %.4f relabel %.4f other %.4f total %.4f (calls %.4f-%.4f) | ordering %.4f (%.4f-%.4f; %d components, %d ordered) | check %s\n" name n CODE PHREPS p["symbolic"] p["numeric_setup"] p["numeric_cpu"] p["numeric_gpu"] p["transfer"] p["solve_setup"] p["inverse"] p["solve"] p["relabel"] p["other"] p["total"] minimum(tot) maximum(tot) p["ordering"] minimum(tord) maximum(tord) ocomp[1] ocomp[2] ok
    flush(stdout)
    CUDA.unsafe_free!(D); D = nothing; GC.gc(); CUDA.reclaim()
    A = A32 = R = nothing; GC.gc()
end
