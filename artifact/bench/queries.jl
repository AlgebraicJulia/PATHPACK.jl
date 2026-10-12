# Single-source / single-target queries and blocks of queries (tropical semiring), against Dijkstra.
# Ours, in elimination coordinates (renumbering excluded): one query with sgetre! (single target :N, single source :T) on
# 1 and NT threads; blocks of k right-hand sides through the general path and the subtree path (TDWorkspace), in the
# column layout (A* B) and the row layout (B A*), on 1 and NT threads. Baseline: PATHPACK's Dijkstra on the same query
# vertices (one thread), and its throughput with one query per thread on NT threads. Public-API answers are checked
# against Dijkstra. Writes the graph (forward arcs, full precision) and the query vertices for GAP and PHAST.
include("lib.jl"); include("graphs.jl")
const KS = [1, 8, 32, 128, 256, 512, 1024]
const NDIJ = parse(Int, get(ENV, "BENCH_NDIJ", "16"))         # Dijkstra samples (slow on the largest networks)
s = MinPlus()
nts = unique([1, NT])
let A = fromgraph(Graphs.grid([12, 12])); n = size(A, 1); F = P.ChordalSLU(s, A; alg = AMF()); copyto!(F, A); lu!(F; nt = NT)
    for nt in nts; ew = elemwork(F, nt); elem!(F, 1, Val(:N), ew); elem!(F, 1, Val(:T), ew)
        B = unitvecs(s, Float64, n, 1:8); gen_solve!(F, copy(B), Val(:L), nt); sub_solve!(F, copy(B), Val(:L), subwork(F, 8, nt)) end
    x = zeros(n); dij!(s, A, 1, dijwork(A), x) end
for name in ARGS
    isfile(resultfile("queries", name)) && continue
    A = graph(name); n = size(A, 1); AT = sparse(transpose(A))
    t_sym = @elapsed (F = P.ChordalSLU(s, A; alg = AMF())); st = fstats(F)
    t_num = Dict(nt => minimum((copyto!(F, A); @elapsed lu!(F; nt)) for _ in 1:2) for nt in nts)
    rng = MersenneTwister(2); qs = rand(rng, 1:n, NSRC)
    # single queries
    single = Dict{String, Any}()
    for nt in nts, (tr, lab) in ((:N, "target"), (:T, "source"))
        ew = elemwork(F, nt); ts = [(j = elimindex(F, q, tr); @elapsed elem!(F, j, Val(tr), ew)) for q in qs]
        single["$(lab)_nt$(nt)"] = Dict("median" => median(ts), "mean" => mean(ts))
    end
    # Dijkstra, the same vertices, one thread: single source on A^T, single target on A
    x = Vector{Float64}(undef, n); dw = dijwork(A); dq = qs[1:min(end, NDIJ)]
    t_ds = [@elapsed dij!(s, AT, q, dw, x) for q in dq]; t_dt = [@elapsed dij!(s, A, q, dw, x) for q in dq]
    # check: public API (original labels) against Dijkstra, single source
    b = Vector{Float64}(undef, n)
    for q in qs[1:min(end, 3)]
        sgetre!(F, Val(:L), Val(:T), b, q; nt = NT); dij!(s, AT, q, dw, x); fin = isfinite.(x)
        (fin == isfinite.(b) && maximum(abs.(b[fin] .- x[fin]); init = 0.0) <= 1e-9 * maximum(abs, x[fin]; init = 1.0)) || error("$name: query disagrees with Dijkstra")
    end
    # Dijkstra throughput, one query per thread on NT threads
    nq = 4 * NT; t_dpar = @elapsed dijkstra!(s, AT, unitvecs(s, Float64, n, qs[1:min(end, nq)]); nt = NT)
    # blocks
    blocks = Dict{String, Any}()
    for k in filter(k -> k * n * 8 * 2 <= RHS_GB * 1e9, KS), nt in nts
        pos = rand(MersenneTwister(k), 1:n, k); B = unitvecs(s, Float64, n, pos); Bt = permutedims(B); w = subwork(F, k, nt)
        tm(f, X) = minimum((Y = copy(X); @elapsed f(Y)) for _ in 1:2) / k
        blocks["k$(k)_nt$(nt)"] = Dict("general_col" => tm(Y -> gen_solve!(F, Y, Val(:L), nt), B), "general_row" => tm(Y -> gen_solve!(F, Y, Val(:R), nt), Bt),
                                     "subtree_col" => tm(Y -> sub_solve!(F, Y, Val(:L), w), B), "subtree_row" => tm(Y -> sub_solve!(F, Y, Val(:R), w), Bt))
        B = Bt = w = nothing; GC.gc()
    end
    if !isfile(joinpath(RESULTS, "queries_$(name).wel"))   # for GAP: forward arcs (single-source), full precision
        open(joinpath(RESULTS, "queries_$(name).wel"), "w") do io; for (i, j, w) in zip(findnz(A)...); println(io, i - 1, ' ', j - 1, ' ', repr(w)); end; end
        open(joinpath(RESULTS, "queries_$(name).src"), "w") do io; foreach(q -> println(io, q - 1), qs); end
    end
    save("queries", name, Dict("n" => n, "m" => nnz(A), "ncc" => P.ncc(F), "fill" => st.nnzLU / nnz(A), "nnzLU" => st.nnzLU,
        "maxfront" => st.maxfront, "t_sym" => t_sym, "t_num" => Dict(string(k) => v for (k, v) in t_num), "single" => single,
        "dijkstra_source" => Dict("median" => median(t_ds), "mean" => mean(t_ds)), "dijkstra_target" => Dict("median" => median(t_dt), "mean" => mean(t_dt)),
        "dijkstra_parallel_per_query" => t_dpar / min(nq, length(qs)), "blocks" => blocks, "nsrc" => length(qs), "ndij" => length(dq)))
    @printf "%-15s n=%8d nt=%d | sym %.2f num %.3f | single source %.3f ms (Dijkstra %.2f ms) | blocks: %d sizes\n" name n NT t_sym t_num[1] 1e3 * single["source_nt1"]["median"] 1e3 * median(t_ds) length(blocks)
    flush(stdout); F = A = AT = nothing; GC.gc()
end
