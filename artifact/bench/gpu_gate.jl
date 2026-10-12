# Gate for the GPU runs (the GPU counterpart of gate.jl): the library's closure on graphs with integer weights, whose
# distances are exact in Float32, against reference row hashes from earlier verified runs (BENCH_GATE_REF: one file per
# graph, line i = rowhash(i), gpu_lib.jl's definition). Any correct closure hashes the same, whatever the ordering or the
# rounding, so every entry point the benchmarks use is checked bit for bit:
#   apsp_gpu(A)                                   the library's default ordering, its own output
#   apsp_gpu!(D, A; alg = BestFill(AMF, METIS))   a preallocated output, the all-pairs ordering
#   the same with factor_gpu = false              the whole factorization on the CPU
#   apsp_gpu(A; devices = all GPUs of the job)    the rows split over the GPUs (when there are several)
# Graphs: BENCH_GATE_MTX/<graph>.mtx, coordinate format, both directions stored, duplicates take the minimum (the format
# of the earlier comparison's suite). Exits with status 1 on any difference. Usage: julia gpu_gate.jl GRAPH...
include("gpu_lib.jl")
const GP = PATHPACK.GPU
const GDIR = ENV["BENCH_GATE_MTX"]; const RDIR = ENV["BENCH_GATE_REF"]
function read_plain(path)
    I = Int[]; J = Int[]; V = Float32[]; n = 0; header = true
    for line in eachline(path)
        startswith(line, '%') && continue
        a = split(line)
        if header; n = parse(Int, a[1]); header = false
        else push!(I, parse(Int, a[1])); push!(J, parse(Int, a[2])); push!(V, parse(Float32, a[3])); end
    end
    return sparse(I, J, V, n, n, min)
end
hexes(h) = [string(x; base = 16, pad = 16) for x in h]
function device_hash(f, n)                       # f() -> n x n device matrix in the labels of A
    D = f(); syncall(); h = hexes(rowhashes(D, 1:n)); CUDA.unsafe_free!(D); GC.gc(); CUDA.reclaim(); return h
end
bad = String[]
for name in ARGS
    A = read_plain(joinpath(GDIR, name * ".mtx")); n = size(A, 1)
    ref = readlines(joinpath(RDIR, name * ".rowhash"))
    results = Pair{String, Vector{String}}[]
    push!(results, "apsp_gpu" => device_hash(() -> GP.apsp_gpu(A), n))
    push!(results, "apsp_gpu! BestFill" => device_hash(() -> GP.apsp_gpu!(CuMatrix{Float32}(undef, n, n), A; alg = ORDER), n))
    push!(results, "factor_gpu=false" => device_hash(() -> GP.with_config(() -> GP.apsp_gpu!(CuMatrix{Float32}(undef, n, n), A; alg = ORDER); factor_gpu = false), n))
    if length(CUDA.devices()) > 1
        blocks = GP.apsp_gpu(A; devices = collect(CUDA.devices()), alg = ORDER)
        h = fill("", n)
        for (dev, (src, Dg)) in zip(CUDA.devices(), blocks)
            CUDA.device!(dev) do; h[src] = hexes(rowhashes(Dg, 1:n)); CUDA.unsafe_free!(Dg); end
        end
        push!(results, "$(length(CUDA.devices())) GPUs" => h)
    end
    for (what, h) in results
        nbad = count(i -> h[i] != ref[i], 1:n)
        @printf "%-18s n=%7d %-20s %s\n" name n what (nbad == 0 ? "bitwise identical to the reference" : "DIFFERS in $nbad rows")
        nbad == 0 || push!(bad, "$name/$what")
    end
    flush(stdout); A = nothing; GC.gc()
end
isempty(bad) || (println("GPU GATE FAILED: ", join(bad, " ")); exit(1))
println("GPU GATE PASSED")
