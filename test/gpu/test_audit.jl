# Regression tests for the solver bugs of the 2026-10-05 audit (audit/REPORT.md: M7, minor 2–5).
#
#   julia --project=. -t auto test/test_audit.jl
#
# The merge-map kernel test needs ~20 GB of free GPU memory (skipped otherwise); the last testset
# redefines precompute_ops! to fail, so it stays last (and this file runs in a process of its own).
haskey(ENV, "SEMIRINGGPU_TUNE_FILE") || (ENV["SEMIRINGGPU_TUNE_FILE"] = tempname())
include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "graphs.jl"))
using .Ext
using .CPU: MinPlus, ChordalSLU, mlu
using CUDA, SparseArrays, Random, Test, Logging
const S = Ext
Random.seed!(11)

closure(s, A) = Matrix(mlu(s, A))
errorof(f) = try f(); nothing catch e; e end
path(n, w, ::Type{T}) where {T} = (I = collect(1:(n - 1)); sparse([I; I .+ 1], [I .+ 1; I], fill(T(w), 2n - 2), n, n))

@testset "audit regressions" begin
    @testset "merge maps: member entries past Int32 (M7.1)" begin
        # front 1 (a residual vertices, separator = the a of the root, front 2): D (2a² = 1.8e9) and L (a² = 9e8)
        # each fit Int32, the entries the map kernel numbers (2.7e9) do not. Was: maps left all zero (an
        # all-Inf merged factor, silently); now no merge (as for D or L too large), before any GPU memory.
        a = 30000
        Rptr = [1, a + 1, 2a + 1]; Sptr = [1, a + 1, a + 1]; Stgt = collect((a + 1):2a)
        Dptr = [1, a^2 + 1, 2a^2 + 1]; Lptr = [1, a^2 + 1, a^2 + 1]; pnt = [2, 0]; idx = [fill(1, a); fill(2, a)]
        @test isnothing(S.amalgamate_fronts(Int, 8, 0.5, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx))
        # and the map builder refuses such a count loudly instead of returning
        e = CuVector{Int32}(undef, 0)
        @test_throws ArgumentError S.amalgamate_maps_gpu!(e, e, e, e, Int32[1, 2], [1, 2], [(1, 1), (2, 2)],
            Rptr, Sptr, Stgt, Dptr, Lptr, Dptr, Lptr)
    end

    @testset "merge maps: no Int32 wrap of the kernel's index (M7.2)" begin
        # one front of nm² entries in [2³¹ - 2²⁴, 2³¹ - 2]: with 2¹⁶ blocks of 256 threads (stride 2²⁴) the
        # Int32 index passed 2³¹ and wrapped negative (out-of-bounds writes). Its map is the identity.
        nm = 46160; N = nm^2
        @assert 2^31 - 2^24 <= N <= 2^31 - 2

        if S.available_memory() < 24 * 2^30
            @test_skip "needs ~20 GB of GPU memory"
        else
            A = S.amalgamate_fronts(Int, 8, 0.5, [1, nm + 1], [1, 1], Int[], [1, N + 1], [1, 1], [0], fill(1, nm))
            @test !isnothing(A) && length(A.mLD) == N && isempty(A.mLL)
            CUDA.synchronize()
            @test minimum(A.mLD) == 1 && maximum(A.mLD) == N && sum(Int64, A.mLD) == N * (N + 1) ÷ 2
            @test Array(view(A.mLD, (N - 999):N)) == (N - 999):N && Array(view(A.mLD, 1:1000)) == 1:1000
            @test A.mUD == A.mLD
            foreach(CUDA.unsafe_free!, (A.mLD, A.mUD, A.mLL, A.mUL))
        end
    end

    @testset "Float32 exactness bound (minor 2)" begin
        G32 = grid(30, 30, Float32)
        @test S.distance_bound(G32) <= 899 * 100
        # (n - 1) max |w| ≥ 2²⁴, the per-column bound is not: no warning
        H = path(300, 1f0, Float32); H[150, 151] = H[151, 150] = 1f5
        @test 299 * 1f5 >= 2^24 && S.distance_bound(H, 2^24) < 2^24
        @test (@test_logs min_level = Logging.Warn apsp_gpu(G32; output = :host)) == closure(MinPlus(), G32)
        @test (@test_logs min_level = Logging.Warn apsp_gpu(H; output = :host)) == closure(MinPlus(), H)
        # the audit's case: a path of 300 vertices, weights 1e6 + 1 (distances up to 3e8)
        P64 = path(300, 1e6 + 1, Float64); P32 = path(300, 1e6 + 1, Float32)
        @test S.distance_bound(P32) == 299 * (1e6 + 1) >= 2^24
        D = @test_logs min_level = Logging.Warn apsp_gpu(P64; output = :host)        # Float64: exact, no warning
        @test D[1, 300] == 299 * (1e6 + 1) && D == [abs(i - j) * (1e6 + 1) for i in 1:300, j in 1:300]
        @test_logs (:warn, r"Float32 distances may reach .* Use Float64") apsp_gpu(P32; output = :host)
        @test_logs (:warn, r"Float32 distances may reach") apsp_gpu(P32, [1, 2])     # every entry point checks
        P32[1, 2] = Inf32                                                           # an infinite weight: no arc
        @test isfinite(S.distance_bound(P32))
    end

    @testset "negative weights: the closure over the extended reals (minor 2)" begin
        # no negative cycle (potentials: w(u, v) + p(u) - p(v) on positive weights): the shortest distances,
        # which are those of the positive weights shifted by p(i) - p(j)
        A = grid(20, 20, Float64); n = size(A, 1); pot = rand(-60:60, n)
        B = copy(A); rows = rowvals(B)
        for j in 1:n, p in nzrange(B, j)
            B.nzval[p] += pot[rows[p]] - pot[j]
        end
        @test any(<(0), nonzeros(B))
        C = closure(MinPlus(), A)
        D = apsp_gpu(B; output = :host)
        @test D == [C[i, j] + pot[i] - pot[j] for i in 1:n, j in 1:n]
        @test D == closure(MinPlus(), B) && Array(apsp_gpu(B)) == D
        # a negative edge of an undirected graph is a negative cycle: -Inf wherever a path can reach it (here all)
        U = grid(10, 10, Float32); U[1, 2] = U[2, 1] = -1f0
        D = apsp_gpu(U; output = :host)
        @test all(==(-Inf32), D) && D == closure(MinPlus(), U)
    end

    @testset "concurrent output = :host calls (minor 3)" begin
        if Threads.nthreads() < 2
            @test_skip "needs threads"
        else
            stage, stagemin = S.HOST_STAGE[], S.HOST_STAGE_MIN[]
            S.HOST_STAGE[] = 2^16; S.HOST_STAGE_MIN[] = 0           # every call streams, ~400 chunks each
            S.free_host_stages!()

            try
                As = [grid(50, 50, Float32) for _ in 1:4]
                Cs = [closure(MinPlus(), A) for A in As]
                @test all(apsp_gpu(As[i]; output = :host) == Cs[i] for i in 1:4)        # (warm: compiled and tuned)

                for _ in 1:3
                    ts = [Threads.@spawn apsp_gpu(As[i]; output = :host) for i in 1:4]
                    @test all(fetch(ts[i]) == Cs[i] for i in 1:4)
                end
            finally
                S.HOST_STAGE[] = stage; S.HOST_STAGE_MIN[] = stagemin
                S.free_host_stages!()
            end
        end
    end

    @testset "plans built ahead do not read G's cache (minor 4)" begin
        with_config(; layered_min_rows = 1, layer_cache_min = 0) do
            A = grid(40, 40, Float32); n = size(A, 1)
            F = ChordalSLU(MinPlus(), A); copyto!(F, A)
            P = FactorPlan(F; large = 256, graph = false, nstreams = 8); factorize!(P)
            G = GPUSLU(P; large = 64)
            W = CuMatrix{Float32}(undef, n, n)
            junk(k) = startswith(String(k), "junk")

            for _ in 1:30
                t = S.slot_plan_ahead(G, W, Val(:N), nothing)
                @test t isa Task
                k = 0

                while !istaskdone(t) && k < 10^6            # G's cache rehashed meanwhile, as the solve's writes do
                    G.cache[Symbol(:junk, k % 4096)] = k; k += 1
                    k % 4096 == 0 && filter!(kv -> !junk(kv[1]), G.cache)
                end

                H = fetch(t)
                @test H isa GPUSLU && H.cache !== G.cache
                filter!(kv -> !junk(kv[1]), G.cache)
            end

            precompute_ops!(G)
            D = Array(closure_gpu(G)); p = Array(G.rperm)
            X = similar(D); X[p, p] = D
            @test X == closure(MinPlus(), A)
        end
    end

    @testset "error paths free the call's GPU memory (minor 5)" begin
        if ismissing(CUDA.used_memory())
            @test_skip "the pool does not report its use"
        else
            A = grid(60, 60, Float32); src = [1, 7, 3600]
            # GPU memory a call leaves allocated (live, uncollected: the collector is off meanwhile)
            function residual(f)
                GC.gc(true); CUDA.reclaim(); CUDA.synchronize()
                GC.enable(false)
                u0 = CUDA.used_memory()
                e = errorof(f); CUDA.synchronize()
                u = CUDA.used_memory() - u0
                GC.enable(true); GC.gc(true)
                return u, e
            end

            with_config(; merge = 1) do                         # (no merge maps: they are cached on purpose)
                for _ in 1:2
                    apsp_gpu(A; output = :host); apsp_gpu(A, src; output = :host)
                end

                ok1, e1 = residual(() -> apsp_gpu(A; output = :host))
                ok2, e2 = residual(() -> apsp_gpu(A, src; output = :host))
                @test isnothing(e1) && isnothing(e2)
                # a failure after the factorization and the solve structure exist
                @eval S precompute_ops!(G::GPUSLU{Sem, T}) where {Sem, T} = error("injected failure")
                bad1, e1 = residual(() -> Base.invokelatest(apsp_gpu, A; output = :host))
                bad2, e2 = residual(() -> Base.invokelatest(apsp_gpu, A, src; output = :host))
                @test e1 isa ErrorException && occursin("injected", e1.msg)
                @test e2 isa ErrorException && occursin("injected", e2.msg)
                println("  residual GPU memory (bytes): ok $ok1, $ok2; failed $bad1, $bad2")
                @test bad1 <= ok1 && bad2 <= ok2
            end
        end
    end
end
