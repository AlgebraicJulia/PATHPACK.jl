# Gate: no result may depend on the thread count. For each graph and semiring, factorize with nt = 1, 2, 4, 8 (up to
# julia -t) and compare, bit for bit, everything the benchmarks time or report: single queries in both directions,
# blocks through the general and the subtree path in both layouts, residual solves, and (small graphs) the closure.
# Exits with status 1 on any difference, so no experiment runs on a nondeterministic build.
include("lib.jl"); include("graphs.jl")
function outputs(s, A, nt)
    n = size(A, 1); F = mlu(s, A; alg = AMF(), nt); T = eltype(F.Nval); rng = MersenneTwister(5); k = 4
    out = Any[]
    for j in rand(rng, 1:n, 3), tr in (Val(:N), Val(:T))
        b = Vector{T}(undef, n); sgetre!(F, Val(:L), tr, b, j; nt); push!(out, b)
    end
    B = unitvecs(s, T, n, rand(rng, 1:n, k))
    push!(out, lmul!(F, copy(B); nt)); push!(out, rmul!(permutedims(B), F; nt))
    W = TDWorkspace(F, k; nt); push!(out, sgetrs!(F, Val(:L), Val(:N), copy(B), W; nt))
    W = TDWorkspace(F, k; nt); push!(out, sgetrs!(F, Val(:R), Val(:N), permutedims(B), W; nt))
    R = rand(rng, T, n, k)
    try push!(out, ldiv!(F, copy(R); nt)); push!(out, rdiv!(permutedims(R), F; nt)) catch e; e isa MethodError || rethrow(); push!(out, "no residuals") end
    n <= 8000 && push!(out, mstar(s, A; alg = AMF(), nt))      # closures only on small graphs: two are held at once
    return out
end
bad = String[]
for name in ARGS, s in (MinPlus(), MaxMin())
    GC.gc(); A = graph(name); nts = filter(<=(Threads.nthreads()), [1, 2, 4, 8])
    ref = outputs(s, A, 1)
    diff = [nt for nt in nts[2:end] if (o = outputs(s, A, nt); d = !isequal(o, ref); o = nothing; GC.gc(); d)]
    @printf "%-15s %-7s n=%7d | nt=%s | %s\n" name nameof(typeof(s)) size(A, 1) join(nts, ",") (isempty(diff) ? "all modes bitwise identical" : "DIFFERS at nt=$(join(diff, ","))")
    isempty(diff) || push!(bad, "$name/$(nameof(typeof(s)))")
end
isempty(bad) || (println("GATE FAILED: ", join(bad, " ")); exit(1))
println("GATE PASSED")
