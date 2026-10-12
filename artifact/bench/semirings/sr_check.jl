# CPU side of gpu_semirings.jl (a CPU-only job): the reference columns per graph and semiring (PATHPACK.CPU closure in a
# Float64 copy of the same weights; Boolean: BFS), compared with the GPU's saved columns; writes the check into each row.
# Usage: BENCH_DATA=... BENCH_NT=16 SR_RAW="dir1 dir2" julia -t 16 sr_check.jl GRAPH...
include(joinpath(get(ENV, "SR_BENCH", joinpath(@__DIR__, "..")), "lib.jl"))
include(joinpath(get(ENV, "SR_BENCH", joinpath(@__DIR__, "..")), "graphs.jl"))
include(joinpath(@__DIR__, "sr_common.jl"))
const RAWS = split(ENV["SR_RAW"])
sample_vertices(n; k = NSRC) = sort(randperm(MersenneTwister(1), n)[1:min(n, k)])
for name in ARGS
    A0 = nothing
    rows = [(d, f) for d in RAWS for f in readdir(d) if startswith(f, "gpuapsp_$(name)_") && endswith(f, ".json")]
    isempty(rows) && continue
    A0 = loadgraph(name); n = size(A0, 1); cols = sample_vertices(n)
    refs = Dict{String, Any}()
    for (d, f) in rows
        path = joinpath(d, f); row = JSON.parsefile(path)
        haskey(row, "check") && haskey(row["check"], "ok") && continue
        get(row, "status", "") in ("unchecked", "entries not written") || continue
        key = row["key"]; jls = joinpath(d, "cols", key * ".jls"); isfile(jls) || continue
        cfg = row["semiring"] * ":" * row["eltype"]; nm, s, T = parsecfg(cfg)
        sv = deserialize(jls); sv.cols == cols || error("$key: sampled columns differ")
        R, t_ref = get!(refs, cfg) do
            W = weights(nm, T, A0)
            t = @elapsed (r = nm == "OrAnd" ? bfs_reference(s, W, cols) : reference(s, W, cols)); (r, t)
        end
        ck = check(nm, s, T, sv.G, R)
        ok = ck.ok && row["check"]["nan_left"] == 0
        row["check"] = merge(row["check"], Dict("ok" => ok, "cpu_closure_ok" => ck.ok, "cpu_closure_maxrel" => ck.maxrel, "mode" => ck.mode,
            "reference" => nm == "OrAnd" ? "BFS reachability" : "PATHPACK.CPU closure, Float64 copy of the weights, AMF"))
        row["status"] = ok ? "ok" : "check failed"; row["t_reference"] = t_ref
        open(io -> JSON.print(io, row, 1), path, "w")
        @printf "%-15s %-18s %-30s %s (%.1e)\n" name cfg get(row, "variant", "") ok ck.maxrel; flush(stdout)
    end
end
println("CHECK DONE")
