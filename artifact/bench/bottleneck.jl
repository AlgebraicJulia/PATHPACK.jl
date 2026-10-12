# Genericity: all-pairs bottleneck (widest) paths, the closure in the (max, min) semiring, against n widest-path
# Dijkstra runs (PATHPACK's generic Dijkstra) and Boost's generic Dijkstra (baselines/bottleneck_boost.cpp).
# Every arc gets its own random capacity (uniform in [1, 100], seed 19): the problem is directed, so the maximum-spanning-
# tree method for symmetric capacities does not apply. Same structure as apsp.jl; checked column by column (exactly:
# max and min do not round). Writes the graph (arcs reversed) and the targets for Boost.
include("lib.jl"); include("graphs.jl")
s = MaxMin()
let A = fromgraph(Graphs.grid([12, 12])); F = P.ChordalSLU(s, A; alg = BestFill(AMF(), METIS())); copyto!(F, A); lu!(F; nt = NT)
    C = Matrix{Float64}(undef, size(A)...); touch!(C, NT); closure_phases!(F, C, NT); dijkstra!(s, A, unitvecs(s, Float64, size(A, 1), 1:8); nt = NT) end
for name in ARGS
    isfile(resultfile("bottleneck", name)) && continue
    A0 = graph(name); n = size(A0, 1); I_, J_, _ = findnz(A0); keep = I_ .!= J_
    A = sparse(I_[keep], J_[keep], 1.0 .+ 99.0 .* rand(MersenneTwister(19), count(keep)), n, n)
    8.0 * n^2 * 1.15 > OUT_GB * 1e9 && (@printf("%-15s n=%d: closure too large for BENCH_OUT_GB, skipped\n", name, n); continue)
    t_sym = @elapsed (F = P.ChordalSLU(s, A; alg = BestFill(AMF(), METIS()))); st = fstats(F)
    t_num = minimum((copyto!(F, A); @elapsed lu!(F; nt = NT)) for _ in 1:2)
    ok = phases_match_fused(F, NT)
    C = Matrix{Float64}(undef, n, n); t_alloc = @elapsed touch!(C, NT); t_clo, t_inv, t_sol = closure_phases!(F, C, NT)
    work = Vector{Float64}(undef, 8n); t_perm = @elapsed (P.permuterows!(C, work, F.rperm); P.permutecols!(C, work, F.cperm))
    cols = sort(randperm(MersenneTwister(1), n)[1:min(n, 64)])            # check: 64 columns, exactly
    B = unitvecs(s, Float64, n, cols); dijkstra!(s, A, B; nt = NT)
    isequal(B, C[:, cols]) || error("$name: closure and widest-path Dijkstra disagree")
    ref = C[:, cols[1]]; B = C = nothing; GC.gc()
    B = unitvecs(s, Float64, n, 1:n); t_dij = @elapsed dijkstra!(s, A, B; nt = NT)     # all n targets, output prepared before the timer
    B = nothing; GC.gc()
    bc = 1:n                                                                  # Boost: all n targets as well
    open(joinpath(RESULTS, "bottleneck_$(name).wel"), "w") do io; for (i, j, c) in zip(findnz(A)...); println(io, j - 1, ' ', i - 1, ' ', repr(c)); end; end
    open(joinpath(RESULTS, "bottleneck_$(name).ref"), "w") do io; foreach(x -> println(io, repr(x)), ref); end
    open(joinpath(RESULTS, "bottleneck_$(name).src"), "w") do io; println(io, cols[1] - 1); foreach(t -> println(io, t - 1), bc); end   # first: the checked target
    save("bottleneck", name, Dict("n" => n, "m" => nnz(A), "ncc" => P.ncc(F), "fill" => st.nnzLU / nnz(A), "t_sym" => t_sym, "t_num" => t_num,
        "t_closure" => t_clo, "t_alloc" => t_alloc, "t_inverse" => t_inv, "t_solve" => t_sol, "phases_match_fused" => ok, "t_perm" => t_perm,
        "t_dijkstra_all" => t_dij, "dijkstra_targets" => n, "capacity_seed" => 19))
    @printf "%-15s n=%7d nt=%d | sym %.3f num %.3f closure %.3f (inverse %s, solve %s) | n x Dijkstra %.3f s\n" name n NT t_sym t_num t_clo something(t_inv, "-") something(t_sol, "-") t_dij
    flush(stdout); F = A = nothing; GC.gc()
end
