# Shared setup for the GPU benchmarks: lib.jl (environment, provenance, results, Dijkstra) and graphs.jl, plus CUDA and the
# solver's GPU backend (PATHPACK's CUDA extension), a description of the GPU, the timing conventions, and the calls we time.
# Run in the GPU environment (bench/gpu/Project.toml): BENCH_DATA=... BENCH_NT=8 julia -t 8 --project=gpu gpu_apsp.jl GRAPH...
#
# Conventions (every GPU row records them):
#   - Float32 on the GPU (the graphs' weights, rounded to Float32); the reference is the CPU's Float64 result.
#   - The n x n output is allocated on the device and first touched (written once) BEFORE the timer: t_alloc, its own phase.
#   - Times are wall clock with the whole device synchronized before the call and after it (CUDA.device_synchronize).
#   - first: the first call on a graph in this process (compiles any kernel specialization the graph needs); excluded.
#   - warm:  repeated calls in one process (CUDA.jl's memory pool and the pinned host arena kept).
#   - cold:  before every call, CUDA.jl's pool is returned to the driver, the library's pinned host arena freed, the host heap
#            trimmed (malloc_trim); compiled code stays (as for a compiled binary). The headline time is the cold median.
#   - GEMM autotuning off (generic kernel choice, SEMIRINGGPU_TUNE=off), unless the job says otherwise (row: gemm_tune).
include("lib.jl"); include("graphs.jl")
using CUDA
const X = Base.get_extension(PATHPACK, :PATHPACKCUDAExt)          # the GPU backend
X === nothing && error("PATHPACK's CUDA extension did not load (is CUDA.jl in this environment?)")
const ORDER = BestFill(AMF(), METIS())                              # the ordering of the CPU all-pairs runs (apsp.jl)
const RTOL = parse(Float64, get(ENV, "BENCH_RTOL", "1e-4"))         # Float32 vs Float64 check, relative (see check_cols)
const REPS = parse(Int, get(ENV, "BENCH_REPS", "5"))                # cold timed calls (median reported)
const WREPS = parse(Int, get(ENV, "BENCH_WREPS", "3"))              # warm timed calls
const EXPORT = get(ENV, "BENCH_EXPORT", joinpath(RESULTS, "export")); mkpath(EXPORT)   # graphs and samples for the baselines

# ---------------- the GPU and its host ----------------
gpukind(dev = CUDA.device()) = (nm = CUDA.name(dev); occursin("B200", nm) ? "b200" : occursin("RTX", nm) ? "rtx" : lowercase(replace(nm, r"[^A-Za-z0-9]+" => "")))
function gpuinfo()
    dev = CUDA.device()
    cfg = PATHPACK.GPU.config()
    return Dict("name" => CUDA.name(dev), "kind" => gpukind(dev), "memory_gb" => CUDA.totalmem(dev) / 1e9,
        "count" => length(CUDA.devices()), "capability" => string(CUDA.capability(dev)), "sms" => CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        "driver" => string(CUDA.driver_version()), "runtime" => string(CUDA.runtime_version()), "cuda_jl" => string(pkgversion(CUDA)),
        "host_cpu" => Sys.cpu_info()[1].model, "host_cores_visible" => Sys.CPU_THREADS, "julia_threads" => Threads.nthreads(),
        "gemm_tune" => cfg.gemm_tune, "library_ordering" => cfg.ordering)
end
# results of the GPU scripts: script id, then graph and GPU kind in the name (collect.py: gpuapsp -> gpu_apsp.csv, ...)
# (as lib.jl's save, but the row's "graph" is the graph itself, given in d, not the file key, which adds GPU and variant)
function gsave(script, key, d; dir = RESULTS)
    haskey(d, "graph") || error("gsave: no graph in the result for $key")
    d = merge(d, Dict("gpu" => gpuinfo(), "script" => script, "key" => key, "provenance" => provenance()))
    open(io -> JSON.print(io, jclean(d), 1), joinpath(dir, basename(resultfile(script, key))), "w")
end
gresultfile(script, name) = resultfile(script, name)

# ---------------- timing ----------------
syncall() = (for d in CUDA.devices(); CUDA.device!(() -> CUDA.device_synchronize(), d); end)
function timed(f)                                                   # (result, seconds), every device idle before and after
    syncall(); GC.gc(false)
    syncall(); t0 = time_ns(); r = f(); syncall()
    return r, (time_ns() - t0) / 1e9
end
function cold!()                                                    # see "cold" above
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
summ(ts) = Dict("median" => median(ts), "min" => minimum(ts), "max" => maximum(ts), "all" => ts)

# ---------------- output allocation (outside the timer): a CuMatrix, first touched (t_alloc_map + t_alloc_touch) ----
function alloc_output(::Type{T}, m, n) where {T}
    D, t_map = timed(() -> CuMatrix{T}(undef, m, n))
    _, t_touch = timed(() -> fill!(D, zero(T)))
    return D, t_map, t_touch
end

# ---------------- the calls we time: the library's public API ----------------
# apsp_gpu!(D, A; alg): the whole pipeline (symbolic phase on the CPU; the hybrid numeric factorization, bottom subtrees
# on CPU threads and top fronts on the GPU, or all on the CPU with factor_gpu = false; the solve structure; the closure,
# X <- U* then X <- X L*, every vertex a source) into the caller's n x n device matrix D, D[i, j] = A*[i, j] in the labels
# of A (columns written in place, or moved after: the "relabel" phase). with_phases(f) times the steps of one call; it
# synchronizes the device around every step, so its phases add up to more than the unsynchronized call.
const GP = PATHPACK.GPU
# (auto: alg = nothing, the library's default ordering; from lib-e that is PATHPACK.CPU.AutoOrder, a per-graph choice)
const ORDERINGS = Dict{String, Any}("bestfill" => ORDER, "amf" => AMF(), "auto" => nothing)   # the all-pairs rule, AMF alone, the library's default
ordering_label(o) = o == "bestfill" ? "BestFill(AMF(), METIS())" : o == "amf" ? "AMF()" : o == "auto" ? "auto ($(GP.config().ordering))" : o
# what AutoOrder chooses for A (from the degrees of its symmetric pattern without the diagonal: hub = HubAMF, nd = BFSND,
# amf = AMF), the rule the library applies; "" when the library has no AutoOrder
function auto_choice(A::SparseMatrixCSC)
    isdefined(PATHPACK.CPU, :AutoOrder) || return ""
    try
        P = A .!= 0; S = P .| permutedims(P); S[diagind(S)] .= false; S = sparse(S); dropzeros!(S)
        FO = PATHPACK.CPU.FastOrderings
        c, x = FO.autochoice(size(S, 1), S.colptr, PATHPACK.CPU.AutoOrder())
        return c === :hub ? "HubAMF(alpha = $x)" : c === :nd ? "BFSND(depthcheck = $x)" : "AMF"
    catch err
        return "unknown: " * sprint(showerror, err)[1:min(end, 100)]
    end
end
closure!(D, A; alg = ORDER, factor_gpu = true) = factor_gpu ? GP.apsp_gpu!(D, A; alg) : GP.with_config(() -> GP.apsp_gpu!(D, A; alg); factor_gpu = false)
phases_of(f) = (r = GP.with_phases(f); ph = r[2]; Dict(String(k) => (k === :steps ? Dict(ph.steps) : v) for (k, v) in pairs(ph)))

# ---------------- checks ----------------
# Float32 against Float64: the same unreachable pairs, and |d32 - d64| <= RTOL d64 everywhere else (d64 = 0: exact).
# A shortest-path distance of L arcs, summed in Float32 from weights rounded to Float32, is off by at most about
# (2L + 1) * 2^-24 relative; RTOL = 1e-4 covers paths of up to ~800 arcs at that bound (integer weights: exact).
function check_cols(G32::AbstractMatrix, R64::AbstractMatrix)
    fin = isfinite.(R64)
    pattern = fin == isfinite.(G32)
    err = 0.0
    for i in eachindex(R64)
        fin[i] || continue
        r = R64[i]; g = Float64(G32[i])
        e = r == 0 ? (g == 0 ? 0.0 : Inf) : abs(g - r) / abs(r)
        err = max(err, e)
    end
    return (ok = pattern && err <= RTOL, pattern = pattern, maxrel = err)
end
sample_vertices(n; k = NSRC) = sort(randperm(MersenneTwister(1), n)[1:min(n, k)])     # apsp.jl's 64 columns
# reference columns A* e_v (paths into v) in Float64: Dijkstra, and the CPU closure from PATHPACK's CPU factorization
ref_dijkstra(A, cols) = (B = unitvecs(MinPlus(), Float64, size(A, 1), cols); dijkstra!(MinPlus(), A, B; nt = NT); B)
function ref_cpu_factor(A, cols)
    F = P.ChordalSLU(MinPlus(), A; alg = ORDER); copyto!(F, A); lu!(F; nt = NT)
    B = lmul!(F, unitvecs(MinPlus(), Float64, size(A, 1), cols); nt = NT)
    return B, merge(fstats(F), (; ncc = P.ncc(F)))
end
integral(A) = all(w -> isinteger(w) && abs(w) < 2^20, nonzeros(A))   # then Float32 distances are exact (n * max w < 2^24 checked below)

# position-sensitive row hashes (the baselines' ROME_ROWHASH / Fast-APSP check; uint64 wrapping arithmetic):
#   v = round(d) if finite else 0xFFFFFFFF;  rowhash(i) = sum_j v(i, j) * (j K1 + K2), j the column's label (1-based)
const K1 = 0x9E3779B97F4A7C15; const K2 = 0x632BE59BD9B4E019
function rowhash_kernel!(h, D, key)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= size(D, 1)
        acc = UInt64(0)
        for j in 1:size(D, 2)
            @inbounds d = D[i, j]
            x = isfinite(d) ? unsafe_trunc(UInt64, round(Float64(d))) : UInt64(0xFFFFFFFF)
            @inbounds acc += x * key[j]
        end
        @inbounds h[i] = acc
    end
    return
end
# the hashes of the rows of D, whose column k holds the label labels[k]
function rowhashes(D::CuMatrix, labels::AbstractVector{<:Integer})
    key = CuVector{UInt64}([UInt64(j) * K1 + K2 for j in labels]); h = CuVector{UInt64}(undef, size(D, 1))
    @cuda threads = 256 blocks = cld(size(D, 1), 256) rowhash_kernel!(h, D, key)
    r = Array(h); CUDA.unsafe_free!(key); CUDA.unsafe_free!(h); return r
end
# (files shared by jobs on several nodes: written to a private name, then renamed into place)
function atomic_write(f, path)
    tmp = "$path.$(gethostname()).$(getpid()).part"; open(f, tmp, "w"); mv(tmp, path; force = true)
end
writehash(path, h) = atomic_write(io -> foreach(x -> println(io, string(x; base = 16, pad = 16)), h), path)

# per-row fingerprints, the check of baselines on real weights (baselines/gpu_fingerprint.py writes theirs): for row i,
# the number of unreachable entries, the sum of the others (Float64), and the entries in the probe columns (the first
# NPROBE sampled vertices). Ours are written once per graph: $EXPORT/<graph>.fp, lines "label unreachable sum p_1 ... p_P".
const NPROBE = 16
probes_of(n) = sample_vertices(n)[1:min(NPROBE, n)]
function fp_kernel!(sm, cn, D)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= size(D, 1)
        acc = 0.0; k = UInt64(0)
        for j in 1:size(D, 2)
            @inbounds d = D[i, j]
            if isfinite(d); acc += Float64(d); else; k += UInt64(1); end
        end
        @inbounds sm[i] = acc; @inbounds cn[i] = k
    end
    return
end
# lines of the fingerprints of the rows of D (labels[t] = label of row t; columns in the labels of A)
function fingerprints(D::CuMatrix, labels, probes)
    sm = CuVector{Float64}(undef, size(D, 1)); cn = CuVector{UInt64}(undef, size(D, 1))
    @cuda threads = 256 blocks = cld(size(D, 1), 256) fp_kernel!(sm, cn, D)
    S = Array(sm); C = Array(cn); Pv = Array(D[:, probes]); CUDA.unsafe_free!(sm); CUDA.unsafe_free!(cn)
    f9(x) = isfinite(x) ? @sprintf("%.9g", x) : "Inf"
    return [string(labels[t], " ", C[t], " ", @sprintf("%.17g", S[t]), " ", join(f9.(Pv[t, :]), " ")) for t in 1:size(D, 1)]
end
function write_fp(name, lines)
    atomic_write(io -> foreach(l -> println(io, l), lines), joinpath(EXPORT, "$name.fp"))
end
# compare a baseline's fingerprint lines with ours: same unreachable count; sum and probes within RTOL (relative)
function compare_fp(theirs, ours::AbstractDict)
    rows = 0; bad = 0; maxrel = 0.0; missing_rows = 0
    rel(a, b) = (a == b ? 0.0 : (isinf(a) || isinf(b)) ? Inf : abs(a - b) / max(abs(b), 1e-30))
    for l in theirs
        t = split(l); isempty(t) && continue
        o = get(ours, parse(Int, t[1]), nothing); o === nothing && (missing_rows += 1; continue)
        rows += 1
        e = parse(Int, t[2]) == parse(Int, o[2]) ? rel(parse(Float64, t[3]), parse(Float64, o[3])) : Inf
        for q in 4:min(length(t), length(o))
            a = t[q] == "Inf" ? Inf : parse(Float64, t[q]); b = o[q] == "Inf" ? Inf : parse(Float64, o[q])
            e = max(e, (isinf(a) && isinf(b)) ? 0.0 : rel(a, b))
        end
        maxrel = max(maxrel, e); e <= RTOL || (bad += 1)
    end
    return (rows = rows, bad = bad, maxrel = maxrel, unknown = missing_rows, ok = rows > 0 && bad == 0 && missing_rows == 0)
end
read_fp(path) = Dict(parse(Int, first(split(l))) => split(l) for l in eachline(path) if !isempty(strip(l)))

# ---------------- the graphs, written for the baselines (Float32 weights, printed so that they read back exactly) ----
# ROME: lower triangle, "real symmetric" (it reads |i > j| entries and adds both directions). Fast-APSP: both directions,
# "real general" (the format of its harness's converter, mtx2fap). The sampled vertices (apsp.jl's 64), one per line.
f32str(w) = @sprintf("%.9g", Float32(w))
function export_graph(name, A; general = false)
    issymmetric(A) || @warn "$name: not symmetric; ROME reads an undirected graph"
    p = joinpath(EXPORT, "$name.rome.mtx")
    if !isfile(p)
        I, J, V = findnz(tril(A, -1))
        atomic_write(p) do io
            println(io, "%%MatrixMarket matrix coordinate real symmetric"); println(io, size(A, 1), " ", size(A, 2), " ", length(I))
            for (i, j, w) in zip(I, J, V); println(io, i, " ", j, " ", f32str(w)); end
        end
    end
    q = joinpath(EXPORT, "$name.general.mtx")
    if general && !isfile(q)
        I, J, V = findnz(A)
        atomic_write(q) do io
            println(io, "%%MatrixMarket matrix coordinate real general"); println(io, size(A, 1), " ", size(A, 2), " ", length(I))
            for (i, j, w) in zip(I, J, V); println(io, i, " ", j, " ", f32str(w)); end
        end
    end
    s = joinpath(EXPORT, "$name.sample")
    isfile(s) || atomic_write(io -> foreach(v -> println(io, v), sample_vertices(size(A, 1))), s)
    pr = joinpath(EXPORT, "$name.probes")
    isfile(pr) || atomic_write(io -> foreach(v -> println(io, v), probes_of(size(A, 1))), pr)
    return
end
