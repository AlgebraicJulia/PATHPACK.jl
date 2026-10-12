# Dense kernels on one GPU, Float32, against NVIDIA's libraries on the same GPU; one result file per measurement.
# Ours (PATHPACK.CPU's dense names on device matrices, the kernels inside the closure's large fronts):
#   sgemx!   C <- C (+) A (x) B             min-plus (and max-min), n^3 multiply-adds    vs cuBLAS SGEMM  C <- C + A B
#   strsx!   B <- B L* (unit lower), B <- B U* (upper), n x n right-hand sides, n^3/2     vs cuBLAS STRSM  B <- B L^-1, B U^-1
#   strtri!  U <- U* (upper triangle; computed as I U*: n^3/2 executed, n^3/6 needed)
#                                                                                        vs cuSOLVER's triangular inverse
#                                                                                           (STRTRI, n^3/6), when CUDA.jl has it
# Rates in G multiply-adds/s of the needed work (n^3, n^3/2, n^3/6). A min-plus multiply-add is an add and a minimum, two
# instructions where a BLAS multiply-add is one fused multiply-add (on sm_100 the kernel pairs adds and uses a three-input
# minimum, so about one instruction per multiply-add there); the rows record the GPU so the comparison is per device.
# Time: the minimum over 3 batches of calls (each batch at least 0.2 s, device synchronized around it), after a warm-up.
include("gpu_lib.jl")
using Random
const KIND = gpukind()
const CUBLAS_ = isdefined(CUDA, :CUBLAS) ? CUDA.CUBLAS : nothing
const CUSOLVER_ = isdefined(CUDA, :CUSOLVER) ? CUDA.CUSOLVER : nothing
CUBLAS_ === nothing && error("CUDA.CUBLAS is not available in this CUDA.jl")
const SIZES = parse.(Int, split(get(ENV, "BENCH_KSIZES", "256 512 1024 2048 4096 8192 16384")))

function rate(f, work)                                              # (seconds per call, G multiply-adds/s)
    f(); syncall()
    t1 = @elapsed (f(); syncall()); reps = max(1, ceil(Int, 0.2 / max(t1, 1e-6)))
    t = minimum(begin syncall(); t0 = time_ns(); for _ in 1:reps; f(); end; syncall(); (time_ns() - t0) / 1e9 / reps end for _ in 1:3)
    return t, work / t / 1e9
end
function record(op, impl, sem, n, work, t, g)
    gsave("gpukernels", "$(op)-$(impl)-$(sem)-$(n)_$(KIND)", Dict("graph" => "dense", "op" => op, "impl" => impl, "semiring" => sem, "T" => "Float32", "n" => n,
        "work_muladds" => work, "time" => t, "gmuladd" => g))
    @printf "%-6s %-9s %-7s n=%6d  %10.3f ms  %9.1f G multiply-adds/s\n" op impl sem n 1e3t g
    flush(stdout)
end
# a well-conditioned triangle for STRSM (diagonal 1 + n), and random nonnegative weights for the semiring kernels
tri(n, uplo) = (H = rand(MersenneTwister(n), Float32, n, n); H = uplo == 'L' ? tril(H) : triu(H); H[diagind(H)] .= n + 1f0; CuArray(H))
# (the CPU names must reach the GPU kernels, not PATHPACK.CPU's host code)
let M = CuArray{Float32, 2, CUDA.DeviceMemory}
    for (f, T) in ((P.sgemx!, Tuple{MinPlus, Val{:N}, Val{:N}, M, M, M}), (P.strsx!, Tuple{MinPlus, Val{:R}, Val{:N}, Val{:L}, Val{:U}, M, M}),
                   (P.strtri!, Tuple{MinPlus, Val{:U}, Val{:N}, M}))
        m = which(f, T); occursin("PATHPACKCUDAExt", string(m.file)) || error("$(nameof(f)) on a CuMatrix dispatches to $(m.file), not the GPU backend")
    end
end
for n in SIZES
    4.0 * 4n^2 * 1.2 > CUDA.totalmem(CUDA.device()) && continue
    A = CUDA.rand(Float32, n, n); B = CUDA.rand(Float32, n, n); C = CUDA.rand(Float32, n, n)
    # ---- GEMM ----
    for (sem, s) in (("MinPlus", MinPlus()), ("MaxMin", MaxMin()))
        t, g = rate(() -> P.sgemx!(s, Val(:N), Val(:N), C, A, B), Float64(n)^3); record("gemm", "ours", sem, n, Float64(n)^3, t, g)
    end
    t, g = rate(() -> CUBLAS_.gemm!('N', 'N', 1f0, A, B, 1f0, C), Float64(n)^3); record("gemm", "cublas", "PlusTimes", n, Float64(n)^3, t, g)
    # ---- triangular solves with n right-hand sides (rows of B), from the right ----
    L = tri(n, 'L'); U = tri(n, 'U'); s = MinPlus(); W = similar(B)
    t, g = rate(() -> (copyto!(W, B); P.strsx!(s, Val(:R), Val(:N), Val(:L), Val(:U), L, W)), Float64(n)^3 / 2); record("trsmL", "ours", "MinPlus", n, Float64(n)^3 / 2, t, g)
    t, g = rate(() -> (copyto!(W, B); P.strsx!(s, Val(:R), Val(:N), Val(:U), Val(:N), U, W)), Float64(n)^3 / 2); record("trsmU", "ours", "MinPlus", n, Float64(n)^3 / 2, t, g)
    t, g = rate(() -> (copyto!(W, B); CUBLAS_.trsm!('R', 'L', 'N', 'U', 1f0, L, W)), Float64(n)^3 / 2); record("trsmL", "cublas", "PlusTimes", n, Float64(n)^3 / 2, t, g)
    t, g = rate(() -> (copyto!(W, B); CUBLAS_.trsm!('R', 'U', 'N', 'N', 1f0, U, W)), Float64(n)^3 / 2); record("trsmU", "cublas", "PlusTimes", n, Float64(n)^3 / 2, t, g)
    # (the copy into W is in both: its time, n^2 words, is small next to n^3/2 multiply-adds from n = 1024 on)
    t, g = rate(() -> copyto!(W, B), 0.0); record("copy", "cuda", "-", n, 0.0, t, 0.0)
    # ---- triangular inverse ----
    t, g = rate(() -> (copyto!(W, U); P.strtri!(s, Val(:U), Val(:N), W)), Float64(n)^3 / 6); record("trtri", "ours", "MinPlus", n, Float64(n)^3 / 6, t, g)
    if CUSOLVER_ !== nothing && isdefined(CUSOLVER_, :trtri!)
        t, g = rate(() -> (copyto!(W, U); CUSOLVER_.trtri!('U', 'N', W)), Float64(n)^3 / 6); record("trtri", "cusolver", "PlusTimes", n, Float64(n)^3 / 6, t, g)
    end
    foreach(CUDA.unsafe_free!, (A, B, C, L, U, W)); GC.gc(); CUDA.reclaim()
end
