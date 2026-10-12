# The GPU closure on every semiring it supports (genericity), one B200, the gpu_apsp.jl protocol: the whole call
# apsp_gpu!(D, A; semiring, alg = nothing) into a preallocated device matrix, cold median of REPS calls, phases of one cold
# call (with_phases), and a check of 64 sampled columns against the CPU closure (PATHPACK.CPU) in the same semiring.
# Parts (SR_PARTS): sass  - instruction counts of the GEMM kernels per semiring
#                   kernels - sgemx!, strsx! (L unit, U), strtri! per semiring, n = SR_KSIZES, vs cuBLAS
#                   closure - every configuration (SR_CONFIGS) on every graph (ARGS)
# Weights (from the benchmark graph, A0 = graph(name), |a_ij| symmetric):
#   tropical (MinPlus, MinPlusLaw): the graph's weights; MaxPlus: their negatives (longest paths, no positive cycle)
#   bottleneck (MaxMin, MinMax): per-arc random capacities 1 + 99 U(0,1), seed 19 (bench/bottleneck.jl's), directed
#   MinProd: 1 + 0.01 U per arc; MaxProd (Viterbi): 1 - 0.01 U per arc (no overflow / underflow over long paths)
#   PlusProd (Fletcher): the capacities, row i divided by 1.1 x its row sum (spectral radius < 1, A* = (I - A)^-1)
#   OrAnd (Boolean, Int32): the pattern, weight 1
#   MaxMin Int32: integer capacities 1:100
# All but tropical/MaxPlus drop the diagonal. References: Float64 copies of the GPU's weights (exactly the same values).
include(joinpath(get(ENV, "SR_BENCH", joinpath(@__DIR__, "..")), "gpu_lib.jl"))
using .P: MaxPlus, MinMax, MinProd, MaxProd, PlusProd, OrAnd, AndOr, MinPlusLaw, szero, sone
const KIND = gpukind()
include(joinpath(@__DIR__, "sr_common.jl"))
const PARTS = split(get(ENV, "SR_PARTS", "sass kernels closure"))
const TAG = get(ENV, "SR_TAG", "base")
const ALLCFG = ["MinPlus:Float32", "MaxPlus:Float32", "MinPlusLaw:Float32", "MaxMin:Float32", "MinMax:Float32", "MinProd:Float32",
                "MaxProd:Float32", "PlusProd:Float32", "OrAnd:Int32", "MaxMin:Int32", "MinPlus:Float64", "MaxMin:Float64", "PlusProd:Float64"]
const CONFIGS = split(get(ENV, "SR_CONFIGS", join(ALLCFG, " ")))
println("GPU ", CUDA.name(CUDA.device()), " cap ", CUDA.capability(CUDA.device()), "; PATHPACK ", provenance()["pathpack_commit"], "; tag ", TAG,
        "; tune ", GP.config().gemm_tune, "; parts ", PARTS); flush(stdout)

opcounts(sass) = (d = Dict{String, Int}(); for l in split(sass, '\n'); m = match(r"^\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?\s.*;", l)
    m === nothing || (d[m[1]] = get(d, m[1], 0) + 1); end; d)
sassof(f) = (io = IOBuffer(); CUDA.@device_code_sass io = io f(); String(take!(io)))

# ======================= SASS =======================
if "sass" in PARTS
    dir = joinpath(RESULTS, "sass"); mkpath(dir)
    keep = ["FMNMX", "FMNMX3", "FADD", "FADD2", "FMUL", "FMUL2", "FFMA", "FFMA2", "FSETP", "FSEL", "FMNMX.NAN", "DMNMX", "DADD", "DMUL", "DFMA", "DSETP",
            "IMNMX", "VIMNMX", "VIMNMX3", "IADD3", "LOP3", "SEL", "ISETP", "LDS", "LDG", "STG", "ATOMG", "RED", "BAR", "CALL", "STL", "LDL"]
    for c in ["MinPlus:Float32", "MaxPlus:Float32", "MinPlusLaw:Float32", "MaxMin:Float32", "MinMax:Float32", "MinProd:Float32", "MaxProd:Float32",
              "PlusProd:Float32", "OrAnd:Int32", "MaxMin:Int32", "MinPlus:Float64", "MaxMin:Float64", "PlusProd:Float64"]
        nm, s, T = parsecfg(c)
        M = CuArray(ones(T, 512, 512)); out = Dict{String, Any}()
        kernels = Pair{String, Function}["v2 128x128" => () -> X.sgemx_gpu!(s, M, M, M; tiling = X.TILING_LARGE)]
        T === Float32 && push!(kernels, "v7 128x64 (heuristic)" => () -> X.launch7!(s, M, M, M, Val(128), Val(64), Val(8), Val(false); tn = 8))
        T === Float32 && X.pair_ok(s, T) && push!(kernels, "v8 128x64" => () -> X.launch7!(s, M, M, M, Val(128), Val(64), Val(8), Val(false); tn = 8, pair = true))
        for (kn, f) in kernels
            try
                txt = sassof(f); write(joinpath(dir, replace("$(nm)_$(T)_$(kn)", r"[^A-Za-z0-9_]+" => "_") * ".sass"), txt)
                d = opcounts(txt); out[kn] = Dict(k => get(d, k, 0) for k in keep if get(d, k, 0) > 0)
                @printf "SASS %-10s %-8s %-22s %s\n" nm T kn join(["$k=$(v)" for (k, v) in sort(collect(out[kn]))], " ")
            catch err
                @printf "SASS %-10s %-8s %-22s FAILED %s\n" nm T kn first(sprint(showerror, err), 200)
            end
        end
        gsave("gpusass", "$(nm)-$(T)_$(KIND)_$(TAG)", Dict("graph" => "dense", "semiring" => nm, "T" => string(T), "tag" => TAG, "counts" => out))
        CUDA.unsafe_free!(M)
    end
    flush(stdout)
end

# variants (SR_VARIANTS, e.g. "pair0 pair1"): settings of the library under test; "default" = as it is
function with_variant(f, v)
    v == "default" && return f()
    if startswith(v, "pair")
        isdefined(X, :HEURISTIC_PAIR) || error("variant $v: this library has no HEURISTIC_PAIR")
        old = X.HEURISTIC_PAIR[]; X.HEURISTIC_PAIR[] = v == "pair1"
        try return f() finally X.HEURISTIC_PAIR[] = old end
    end
    error("unknown variant $v")
end
const VARIANTS_ = split(get(ENV, "SR_VARIANTS", "default"))

# ======================= kernels =======================
function rate(f, work)
    f(); syncall()
    t1 = @elapsed (f(); syncall()); reps = max(1, ceil(Int, 0.2 / max(t1, 1e-6)))
    t = minimum(begin syncall(); t0 = time_ns(); for _ in 1:reps; f(); end; syncall(); (time_ns() - t0) / 1e9 / reps end for _ in 1:3)
    return t, work / t / 1e9
end
function krecord(op, impl, sem, T, n, work, t, g; variant = "default")
    gsave("gpukernels", "$(op)-$(impl)-$(sem)-$(T)-$(n)_$(KIND)_$(TAG)_$(variant)", Dict("graph" => "dense", "op" => op, "impl" => impl, "semiring" => sem, "T" => string(T),
        "n" => n, "work_muladds" => work, "time" => t, "gmuladd" => g, "tag" => TAG, "variant" => variant))
    @printf "KERN %-6s %-7s %-10s %-8s %-7s n=%6d %10.3f ms %9.1f G/s\n" op impl sem T variant n 1e3t g; flush(stdout)
end
randm(::Type{T}, n, rng) where {T <: AbstractFloat} = CuArray(rand(rng, T, n, n))
randm(::Type{T}, n, rng) where {T <: Integer} = CuArray(rand(rng, T(0):T(1000), n, n))
if "kernels" in PARTS
    CUBLAS_ = CUDA.CUBLAS
    KS = parse.(Int, split(get(ENV, "SR_KSIZES", "1024 2048 4096 8192 16384")))
    KCFG = split(get(ENV, "SR_KCONFIGS", "MinPlus:Float32 MaxPlus:Float32 MaxMin:Float32 MinMax:Float32 MinProd:Float32 MaxProd:Float32 PlusProd:Float32 OrAnd:Int32 MaxMin:Int32 MinPlus:Float64 MaxMin:Float64 PlusProd:Float64"))
    for n in KS
        for T in (Float32, Float64)
            any(c -> parsecfg(c)[3] === T, KCFG) || continue
            8.0 * sizeof(T) * n^2 > 0.5 * CUDA.totalmem(CUDA.device()) && continue
            A = CUDA.rand(T, n, n); B = CUDA.rand(T, n, n); C = CUDA.rand(T, n, n); W = similar(B)
            t, g = rate(() -> CUBLAS_.gemm!('N', 'N', one(T), A, B, one(T), C), Float64(n)^3); krecord("gemm", "cublas", "PlusTimes", T, n, Float64(n)^3, t, g)
            L = CuArray(tril(rand(MersenneTwister(n), T, n, n)) + (n + 1) * I); U = CuArray(triu(rand(MersenneTwister(n + 1), T, n, n)) + (n + 1) * I)
            t, g = rate(() -> (copyto!(W, B); CUBLAS_.trsm!('R', 'L', 'N', 'U', one(T), L, W)), Float64(n)^3 / 2); krecord("trsmL", "cublas", "PlusTimes", T, n, Float64(n)^3 / 2, t, g)
            t, g = rate(() -> (copyto!(W, B); CUBLAS_.trsm!('R', 'U', 'N', 'N', one(T), U, W)), Float64(n)^3 / 2); krecord("trsmU", "cublas", "PlusTimes", T, n, Float64(n)^3 / 2, t, g)
            foreach(CUDA.unsafe_free!, (A, B, C, W, L, U)); GC.gc(); CUDA.reclaim()
        end
        for c in KCFG
            nm, s, T = parsecfg(c)
            8.0 * sizeof(T) * n^2 > 0.5 * CUDA.totalmem(CUDA.device()) && continue
            rng = MersenneTwister(n)
            A = randm(T, n, rng); B = randm(T, n, rng); C = randm(T, n, rng); W = similar(B)
            Lh = Array(randm(T, n, rng)); Uh = copy(Lh)
            L = CuArray(tril(Lh)); U = CuArray(triu(Uh))
            try
                for v in split(get(ENV, "SR_KVARIANTS", "default"))
                    v != "default" && T !== Float32 && continue
                    with_variant(v) do
                        t, g = rate(() -> P.sgemx!(s, Val(:N), Val(:N), C, A, B), Float64(n)^3); krecord("gemm", "ours", nm, T, n, Float64(n)^3, t, g; variant = v)
                        t, g = rate(() -> (copyto!(W, B); P.strsx!(s, Val(:R), Val(:N), Val(:L), Val(:U), L, W)), Float64(n)^3 / 2); krecord("trsmL", "ours", nm, T, n, Float64(n)^3 / 2, t, g; variant = v)
                        t, g = rate(() -> (copyto!(W, B); P.strsx!(s, Val(:R), Val(:N), Val(:U), Val(:N), U, W)), Float64(n)^3 / 2); krecord("trsmU", "ours", nm, T, n, Float64(n)^3 / 2, t, g; variant = v)
                        t, g = rate(() -> (copyto!(W, U); P.strtri!(s, Val(:U), Val(:N), W)), Float64(n)^3 / 6); krecord("trtri", "ours", nm, T, n, Float64(n)^3 / 6, t, g; variant = v)
                    end
                end
            catch err
                @printf "KERN %-10s %-8s n=%d FAILED: %s\n" nm T n first(sprint(showerror, err), 300)
            end
            foreach(CUDA.unsafe_free!, (A, B, C, W, L, U)); GC.gc(); CUDA.reclaim()
        end
    end
end

# ======================= closures =======================
sentinel(::Type{T}) where {T <: AbstractFloat} = T(NaN)
sentinel(::Type{T}) where {T <: Integer} = T(-12345)
leftover(D, x) = x isa AbstractFloat && isnan(x) ? count(isnan, D) : count(==(x), D)

# a hash of the bits of the whole result (bit-identity across library versions): per row sum_j bits(D[i, j]) (j K1 + K2), folded
function bits_kernel!(h, D, key)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= size(D, 1)
        acc = UInt64(0)
        for j in 1:size(D, 2)
            @inbounds x = D[i, j]
            b = sizeof(x) == 4 ? UInt64(reinterpret(UInt32, x)) : reinterpret(UInt64, x)
            @inbounds acc += b * key[j]
        end
        @inbounds h[i] = acc
    end
    return
end
function bithash(D::CuMatrix)
    key = CuVector{UInt64}([UInt64(j) * K1 + K2 for j in 1:size(D, 2)]); h = CuVector{UInt64}(undef, size(D, 1))
    @cuda threads = 256 blocks = cld(size(D, 1), 256) bits_kernel!(h, D, key)
    r = Array(h); CUDA.unsafe_free!(key); CUDA.unsafe_free!(h)
    return string(foldl((a, (i, x)) -> a + x * (UInt64(i) * 0xD6E8FEB86659FD93 + 1), enumerate(r); init = UInt64(0)); base = 16, pad = 16)
end

if "closure" in PARTS
    REPS_ = parse(Int, get(ENV, "SR_REPS", "5")); WREPS_ = parse(Int, get(ENV, "SR_WREPS", "2"))
    colsdir = joinpath(RESULTS, "cols"); mkpath(colsdir)
    let A = SparseMatrixCSC{Float32, Int}(fromgraph(Graphs.grid([12, 12])))                       # compile the common paths
        for v in VARIANTS_, c in CONFIGS
            nm, s, T = parsecfg(c); W = weights(nm, T, A); D = CuMatrix{T}(undef, size(A)...)
            try with_variant(v) do; GP.apsp_gpu!(D, W; semiring = s); phases_of(() -> GP.apsp_gpu!(D, W; semiring = s)); end catch err; println("warmup $v $c: ", first(sprint(showerror, err), 300)) end
            CUDA.unsafe_free!(D)
        end
        GC.gc()
    end
    for name in ARGS
        A0 = loadgraph(name); n = size(A0, 1); cols = sample_vertices(n)
        for c in CONFIGS
            nm, s, T = parsecfg(c)
            if 1.1 * sizeof(T) * n^2 > CUDA.totalmem(CUDA.device())
                for v in VARIANTS_
                    gsave("gpuapsp", "$(name)_$(KIND)_$(TAG)_$(v)_$(nm)-$(T)", Dict("graph" => name, "semiring" => nm, "eltype" => string(T), "tag" => TAG, "variant" => v, "n" => n, "status" => "too large for one GPU"))
                end
                continue
            end
            W = weights(nm, T, A0)
            cold!(); D, t_map, t_touch = alloc_output(T, n, n)
            for v in VARIANTS_
                key = "$(name)_$(KIND)_$(TAG)_$(v)_$(nm)-$(T)"
                isfile(gresultfile("gpuapsp", key)) && continue
                base = Dict{String, Any}("graph" => name, "code" => "ours-auto", "semiring" => nm, "eltype" => string(T), "tag" => TAG, "variant" => v, "n" => n, "m" => nnz(W))
                call() = with_variant(() -> GP.apsp_gpu!(D, W; semiring = s), v)
                local t_first, t_warm, t_cold, ph_cold, ph_warm, nleft, bh
                try
                    _, t_first = timed(call)
                    t_warm = [timed(call)[2] for _ in 1:WREPS_]
                    t_cold = [(cold!(); timed(call)[2]) for _ in 1:REPS_]
                    cold!(); ph_cold = phases_of(call); ph_warm = phases_of(call)
                    fill!(D, sentinel(T)); syncall(); call(); syncall()
                    nleft = leftover(D, sentinel(T)); bh = bithash(D)
                    serialize(joinpath(colsdir, key * ".jls"), (cols = cols, G = Array(D[:, cols])))
                catch err
                    msg = first(sprint(showerror, err), 300); @printf "%-15s %-18s %s FAILED: %s\n" name c v msg
                    gsave("gpuapsp", key, merge(base, Dict("status" => "failed: " * msg))); continue
                end
                gsave("gpuapsp", key, merge(base, Dict("status" => nleft == 0 ? "unchecked" : "entries not written", "ordering" => ordering_label("auto"),
                    "auto_choice" => auto_choice(W), "output" => "device, preallocated (apsp_gpu!); original labels",
                    "t_total" => median(t_cold), "t_cold" => summ(t_cold), "t_warm" => summ(t_warm), "t_first" => t_first,
                    "t_alloc" => t_map + t_touch, "t_alloc_map" => t_map, "t_alloc_touch" => t_touch, "phases" => ph_cold, "phases_warm" => ph_warm,
                    "bithash" => bh, "check" => Dict("nan_left" => nleft, "columns" => length(cols)))))
                p = ph_cold
                @printf "%-15s n=%7d %-18s %-7s total %.4f (min %.4f max %.4f; warm %.4f) | sym %.4f num %.4f+%.4f+%.4f xfer %.4f setup %.4f inv %.4f solve %.4f relabel %.4f | left %d hash %s\n" name n c v median(t_cold) minimum(t_cold) maximum(t_cold) median(t_warm) p["symbolic"] p["numeric_setup"] p["numeric_cpu"] p["numeric_gpu"] p["transfer"] p["solve_setup"] p["inverse"] p["solve"] p["relabel"] nleft bh
                flush(stdout)
            end
            CUDA.unsafe_free!(D); D = nothing; W = nothing; GC.gc(); CUDA.reclaim()
        end
        A0 = nothing; GC.gc()
    end
end
println("ALL DONE")
