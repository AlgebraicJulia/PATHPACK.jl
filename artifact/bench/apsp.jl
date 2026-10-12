# All-pairs shortest paths: the closure A* in the tropical semiring, phase by phase, against n single-source runs.
# Ours: symbolic analysis, numeric factorization (sgetrf), triangular inverse (U*), solve with n right-hand sides (X L*),
# all in elimination coordinates; the renumbering to original labels is timed separately (t_perm) and not counted.
# The output's first touch (faulting in n^2 words from the OS) is timed as its own phase, t_alloc, before the inverse.
# Baseline: n runs of PATHPACK's Dijkstra, one per target, spread over NT threads, writing into an n x n output that is
# allocated and filled before the timer starts (the same convention as our closure). 64 columns are first checked against
# the closure, which is then freed, so that one n x n matrix exists at a time.
# Also writes the graph and a sample of targets for GAP (double weights): arcs reversed, so GAP's single-source run
# from t computes column t of the closure. Usage: julia -t NT apsp.jl GRAPH...
include("lib.jl"); include("graphs.jl")
s = MinPlus()
let A = fromgraph(Graphs.grid([12, 12])); F = P.ChordalSLU(s, A; alg = BestFill(AMF(), METIS())); copyto!(F, A); lu!(F; nt = NT)
    C = Matrix{Float64}(undef, size(A)...); touch!(C, NT); closure_phases!(F, C, NT); dijkstra!(s, A, unitvecs(s, Float64, size(A, 1), 1:8); nt = NT) end
for name in ARGS
    isfile(resultfile("apsp", name)) && continue
    A = graph(name); n = size(A, 1)
    if 8.0 * n^2 * 1.15 > OUT_GB * 1e9
        @printf "%-15s n=%d: closure needs %.1f GB > BENCH_OUT_GB=%.1f, skipped\n" name n 8.0 * n^2 * 1.15 / 1e9 OUT_GB; continue
    end
    t_sym = @elapsed (F = P.ChordalSLU(s, A; alg = BestFill(AMF(), METIS()))); st = fstats(F)
    t_num = minimum((copyto!(F, A); @elapsed lu!(F; nt = NT)) for _ in 1:2)
    ok = phases_match_fused(F, NT)
    C = Matrix{Float64}(undef, n, n); t_alloc = @elapsed touch!(C, NT)
    t_clo, t_inv, t_sol = closure_phases!(F, C, NT)
    work = Vector{Float64}(undef, 8n)
    t_perm = @elapsed (P.permuterows!(C, work, F.rperm); P.permutecols!(C, work, F.cperm))
    cols = sort(randperm(MersenneTwister(1), n)[1:min(n, 64)])            # check: 64 columns against Dijkstra
    B = unitvecs(s, Float64, n, cols); dijkstra!(s, A, B; nt = NT)
    Cs = C[:, cols]; fin = isfinite.(B)
    (fin == isfinite.(Cs) && maximum(abs.(B[fin] .- Cs[fin]); init = 0.0) <= 1e-9 * maximum(abs, B[fin]; init = 1.0)) || error("$name: closure and Dijkstra disagree")
    Cs = B = C = nothing; GC.gc()
    B = unitvecs(s, Float64, n, 1:n)                                         # the baseline's n x n output, prepared before its timer
    t_dij = @elapsed dijkstra!(s, A, B; nt = NT)                              # all n targets
    B = nothing; GC.gc()
    wel = joinpath(RESULTS, "apsp_$(name).wel")
    if !isfile(wel)                                  # for GAP: reversed arcs, full precision; a sample of targets
        open(wel, "w") do io; for (i, j, w) in zip(findnz(A)...); println(io, j - 1, ' ', i - 1, ' ', repr(w)); end; end
    end
    save("apsp", name, Dict("n" => n, "m" => nnz(A), "ncc" => P.ncc(F), "fill" => st.nnzLU / nnz(A), "nnzLU" => st.nnzLU,
        "maxfront" => st.maxfront, "muladds" => st.muladds, "t_sym" => t_sym, "t_num" => t_num, "t_closure" => t_clo,
        "t_alloc" => t_alloc, "t_inverse" => t_inv, "t_solve" => t_sol, "phases_match_fused" => ok, "t_perm" => t_perm,
        "t_dijkstra_all" => t_dij, "dijkstra_targets" => n))
    @printf "%-15s n=%7d nt=%d | sym %.3f num %.3f alloc %.3f closure %.3f (inverse %s, solve %s) perm %.3f | n x Dijkstra %.3f s\n" name n NT t_sym t_num t_alloc t_clo something(t_inv, "-") something(t_sol, "-") t_perm t_dij
    flush(stdout); F = A = nothing; GC.gc()
end
