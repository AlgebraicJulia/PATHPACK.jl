# All-pairs shortest paths on several GPUs of one node, through the library's public API: the rows of the closure split
# into contiguous blocks, one per GPU; the GPU that holds the factor (the first) builds the solve structure, every other
# GPU gets a copy of it device to device (the only data that moves between GPUs, before the closure), then each computes
# its rows, X <- U*, X <- X L*, with no communication. The output stays distributed (block g on GPU g, original labels).
# For each graph and every GPU count g in BENCH_NGPUS (default 1 2 ... 8, up to the GPUs present):
#   whole call   apsp_gpu(A; devices, alg) (cold median, gpu_lib.jl): from A in host memory to every block computed and
#                every device synchronized; the library allocates the blocks inside the call (overlapped with the symbolic
#                phase), as for one GPU. Freeing the result blocks afterwards is not part of it: timed on its own (t_free,
#                after each timed call); t / t_total leave it out. Its timeline (factor done, structure ready, per GPU:
#                copy ready, closure done) from one more call
#   closure      the lower-level API on one factorization: MultiGPUSLU(P; devices) (structure, operators, copies), then
#                closure_multigpu!(MG; out, timer) into blocks allocated before the timer: the closure alone, per-GPU
#                times and their balance (median of BENCH_MREPS)
# A count whose largest block (and its workspace) exceeds one GPU's memory is reported as such ("too large for g GPUs",
# with the ordering, as every row). BENCH_FIT_ONLY=1: only those reports, for the counts that do not fit (no runs).
# Checks, every count: 64 rows (apsp.jl's sample as sources) against Dijkstra in Float64, from the whole call and from
# the closure; with integer weights, the row hashes of the whole result against gpu_apsp.jl's (or the first count's).
# Usage: julia -t T gpu_multigpu.jl GRAPH...
include("gpu_lib.jl")
using Random
s = MinPlus()
const KIND = gpukind()
const NGPUS = filter(<=(length(CUDA.devices())), parse.(Int, split(get(ENV, "BENCH_NGPUS", "1 2 3 4 5 6 7 8"))))
const MREPS = parse(Int, get(ENV, "BENCH_MREPS", "3"))
const MORDER = get(ENV, "BENCH_MORDERING", "bestfill")
const MSUFFIX = MORDER == "bestfill" ? "" : "_$(MORDER)"           # (result files of other orderings: their own keys)
const FIT_ONLY = get(ENV, "BENCH_FIT_ONLY", "0") == "1"
blockneed(n, g) = maximum(length.(X.row_blocks(n, g))) * n * 4.0 * 1.08   # (bytes of the largest block, with workspace)
function toolarge(name, n, g)
    @printf "%-15s n=%d on %d GPUs: a block needs %.1f GB, more than one GPU holds: skipped\n" name n g blockneed(n, g) / 1e9
    gsave("gpumulti", "$(name)_$(KIND)_g$(g)$(MSUFFIX)", Dict("code" => "ours", "graph" => name, "ngpus" => g, "n" => n, "ordering" => MORDER,
        "status" => "too large for $g GPUs", "need_gb" => blockneed(n, g) / 1e9, "gpu_memory_gb" => minimum(CUDA.totalmem.(devs(g))) / 1e9))
end
devs(g) = collect(CUDA.devices())[1:g]
# apsp_gpu(A; devices) as blocks [(sources_g, D_g)] (with one device it returns the n x n matrix itself)
multi(A, devices; alg) = (r = GP.apsp_gpu(A; devices, alg); r isa CuMatrix ? [(collect(1:size(A, 1)), r)] : r)
freeblocks(blocks, devices) = for (d, (_, B)) in zip(devices, blocks); CUDA.device!(() -> CUDA.unsafe_free!(B), d); end
function free_mg!(MG, devices)      # (no public free: the backend's own, as apsp_gpu does; the first part may share the plan's
    for (k, (d, G)) in enumerate(zip(devices, MG.parts))         #  factor arrays, so it only drops its operators)
        CUDA.device!(() -> (k == 1 ? X.free_ops!(G) : X.free_solver!(G)), d)
    end
end
# rows srcs of the result, in the labels of A, from blocks [(sources_g, D_g)] (D_g[t, :] = row sources_g[t])
function gather_rows(blocks, devices, srcs, n)
    G = Matrix{Float32}(undef, n, length(srcs)); where = Dict{Int, Tuple{Int, Int}}()
    for (k, (sg, _)) in enumerate(blocks), (t, v) in enumerate(sg); where[v] = (k, t); end
    for (c, v) in enumerate(srcs)
        k, t = where[v]; G[:, c] = CUDA.device!(() -> Array(blocks[k][2][t, :]), devices[k])
    end
    return G
end
function hash_rows(blocks, devices, n)
    h = Vector{String}(undef, n)
    for (d, (sg, B)) in zip(devices, blocks); h[sg] = [string(x; base = 16, pad = 16) for x in CUDA.device!(() -> rowhashes(B, 1:n), d)]; end
    return h
end

let A = SparseMatrixCSC{Float32, Int}(fromgraph(Graphs.grid([12, 12]))); n = size(A, 1)     # compile the common paths
    for g in NGPUS
        freeblocks(multi(A, devs(g); alg = ORDERINGS[MORDER]), devs(g))
    end
    GC.gc()
end
for name in ARGS
    if FIT_ONLY
        n = size(graph(name), 1)
        for g in NGPUS; blockneed(n, g) > minimum(CUDA.totalmem.(devs(g))) && toolarge(name, n, g); end
        continue
    end
    A = graph(name); n = size(A, 1); A32 = SparseMatrixCSC{Float32, Int}(A); alg = ORDERINGS[MORDER]
    export_graph(name, A; general = true)
    srcs = sample_vertices(n)
    R = ref_dijkstra(sparse(transpose(A)), srcs)              # column t = row srcs[t] of A* (sources), Float64
    exact = integral(A) && n * maximum(nonzeros(A); init = 0.0) < 2^24
    ref_hash = isfile(joinpath(EXPORT, "$name.rowhash")) ? readlines(joinpath(EXPORT, "$name.rowhash")) : nothing
    # ---- new weights, same graph (the reuse workload): every arc's weight times a factor in [1, 2] (seeded) ----
    A2 = copy(A); nonzeros(A2) .*= 1 .+ rand(Random.MersenneTwister(20261007), nnz(A2)); A2_32 = SparseMatrixCSC{Float32, Int}(A2)
    R2 = ref_dijkstra(sparse(transpose(A2)), srcs)
    # ---- one factorization for the closure-only runs (lower-level API, on the first GPU) ----
    F = P.ChordalSLU(s, A32; alg = something(alg, isdefined(X, :elimination_algorithm) ? X.elimination_algorithm() : ORDER)); copyto!(F, A32)   # (auto: the library's default)
    Pl = CUDA.device!(() -> (Q = GP.extension().FactorPlan(F; large = 256, graph = false, nstreams = 8); GP.extension().factorize!(Q); Q), devs(1)[1])
    for g in NGPUS
        isfile(gresultfile("gpumulti", "$(name)_$(KIND)_g$(g)$(MSUFFIX)")) && continue
        devices = devs(g); rows = X.row_blocks(n, g)
        blockneed(n, g) > minimum(CUDA.totalmem.(devices)) && (toolarge(name, n, g); continue)
        try
            # ---- whole call ----
            cold!()
            # (the call timed up to its blocks computed and synchronized; their free timed after it, apart: t_free)
            whole() = (cold!(); (wbl, wt) = timed(() -> multi(A32, devices; alg)); (_, wtf) = timed(() -> freeblocks(wbl, devices)); (wt, wtf))
            t_first, _ = whole()
            tw = [whole() for _ in 1:MREPS]; t_tot = first.(tw); t_fr = last.(tw)
            cold!(); X.MGPU_TIMES[] = Dict{Any, Float64}()
            blocks = multi(A32, devices; alg); syncall()
            tl = X.MGPU_TIMES[]; X.MGPU_TIMES[] = nothing
            Gw = gather_rows(blocks, devices, srcs, n); cw = check_cols(Gw, R)
            hw = exact ? hash_rows(blocks, devices, n) : nothing
            isfile(joinpath(EXPORT, "$name.fp")) || write_fp(name, reduce(vcat, [CUDA.device!(() -> fingerprints(B, sg, probes_of(n)), d) for (d, (sg, B)) in zip(devices, blocks)]))
            freeblocks(blocks, devices); blocks = nothing; GC.gc()
            hok = nothing
            if exact
                hok = isnothing(ref_hash) || hw == ref_hash
                isnothing(ref_hash) && (writehash(joinpath(EXPORT, "$name.rowhash"), [parse(UInt64, x; base = 16) for x in hw]); ref_hash = hw)
            end
            # ---- closure alone, into preallocated blocks ----
            out = [CUDA.device!(() -> (B = CuMatrix{Float32}(undef, length(rows[k]), n); fill!(B, NaN32); B), d) for (k, d) in enumerate(devices)]
            MG, t_mg = timed(() -> CUDA.device!(() -> GP.extension().MultiGPUSLU(Pl; devices), devices[1]))
            walls = [zeros(g) for _ in 1:MREPS + 1]
            t_clo = [timed(() -> GP.extension().closure_multigpu!(MG; out, timer = walls[r], labels = :original))[2] for r in 1:MREPS + 1]
            nnan = sum(CUDA.device!(() -> count(isnan, out[k]), d) for (k, d) in enumerate(devices))   # (prefilled before the first)
            cblocks = [(collect(rows[k]), out[k]) for k in 1:g]
            cc = check_cols(gather_rows(cblocks, devices, srcs, n), R)
            free_mg!(MG, devices); MG = nothing
            # ---- new weights, same graph: the symbolic analysis and the plan (Pl) made once, outside the timer; timed: the
            #      numeric refactorization with A2's weights (hybrid, on the first GPU), the structure and copies to every
            #      GPU, the closure into the preallocated blocks. Warm; the first call (prefilled with NaN) checked against
            #      Dijkstra on A2 and excluded from the median. Afterwards the plan is refactorized with A's weights again.
            t_re = Float64[]; rwalls = [zeros(g) for _ in 1:MREPS + 1]; nnan2 = -1; c2 = (ok = false, maxrel = NaN); reuse_err = ""
            try
                foreach(((k, d),) -> CUDA.device!(() -> fill!(out[k], NaN32), d), enumerate(devices)); syncall()
                for r in 1:MREPS + 1
                    M, t = timed(() -> begin
                        copyto!(F, A2_32); CUDA.device!(() -> GP.extension().factorize!(Pl), devices[1])
                        Mg = CUDA.device!(() -> GP.extension().MultiGPUSLU(Pl; devices), devices[1])
                        GP.extension().closure_multigpu!(Mg; out, timer = rwalls[r], labels = :original); Mg
                    end)
                    push!(t_re, t)
                    if r == 1
                        nnan2 = sum(CUDA.device!(() -> count(isnan, out[k]), d) for (k, d) in enumerate(devices))
                        c2 = check_cols(gather_rows(cblocks, devices, srcs, n), R2)
                    end
                    free_mg!(M, devices)
                end
            catch err
                reuse_err = sprint(showerror, err)[1:min(end, 300)]
                @printf "%-15s %d GPUs reuse FAILED: %s\n" name g reuse_err
            end
            copyto!(F, A32); CUDA.device!(() -> GP.extension().factorize!(Pl), devices[1])     # (the plan back on A's weights)
            foreach(((k, d),) -> CUDA.device!(() -> CUDA.unsafe_free!(out[k]), d), enumerate(devices)); out = nothing; GC.gc()
            reuse_ok = isempty(reuse_err) && nnan2 == 0 && c2.ok
            t_c = t_clo[2:end]; w = walls[2:end][argmin(abs.(t_c .- median(t_c)))]          # (the first closure call excluded)
            ok = cw.ok && cc.ok && nnan == 0 && hok !== false
            ok || @printf "%-15s %d GPUs CHECK FAILED: whole call %s (%.2e), closure %s (%.2e), NaN %d, hashes %s\n" name g cw.ok cw.maxrel cc.ok cc.maxrel nnan hok
            gsave("gpumulti", "$(name)_$(KIND)_g$(g)$(MSUFFIX)", Dict("code" => "ours", "graph" => name, "ngpus" => g, "status" => ok ? "ok" : "check failed",
                "n" => n, "m" => nnz(A), "ordering" => MORDER, "auto_choice" => MORDER == "auto" ? auto_choice(A) : "", "rows_per_gpu" => length.(rows),
                "t_total" => median(t_tot), "t" => summ(t_tot), "t_first" => t_first, "t_free" => median(t_fr), "free" => summ(t_fr),
                "scope" => "whole call: A in host memory to every block computed and synchronized; the blocks' free after it, apart (t_free)",
                "timeline" => Dict(string(k isa Tuple ? join(k, "_") : k) => v for (k, v) in tl),
                "t_closure" => median(t_c), "closure" => summ(t_c), "closure_walls" => w, "closure_imbalance" => maximum(w) / (sum(w) / g),
                "t_structure_and_copies" => t_mg,
                "t_reuse" => length(t_re) > 1 ? median(t_re[2:end]) : nothing, "reuse" => length(t_re) > 1 ? summ(t_re[2:end]) : nothing,
                "reuse_first" => isempty(t_re) ? nothing : t_re[1], "reuse_closure_walls" => length(t_re) > 1 ? rwalls[2:end][argmin(abs.(t_re[2:end] .- median(t_re[2:end])))] : nothing,
                "reuse_scope" => "new weights (each arc x U[1, 2], seeded), same graph: numeric refactorization (hybrid) + structure and copies to every GPU + closure into preallocated blocks; symbolic analysis and plan made once, outside the timer; warm, first call excluded",
                "reuse_check" => Dict("ok" => reuse_ok, "nan_left" => nnan2, "dijkstra_maxrel" => c2.maxrel, "error" => reuse_err),
                "threads" => Threads.nthreads(),
                "communication" => "solve structure, factor and operators copied device to device before the closure; none during it",
                "output" => "device, distributed (block g on GPU g, contiguous rows), original labels",
                "check" => Dict("ok" => ok, "rows" => length(srcs), "whole_ok" => cw.ok, "whole_maxrel" => cw.maxrel, "closure_ok" => cc.ok,
                    "closure_maxrel" => cc.maxrel, "nan_left" => nnan, "rtol" => RTOL, "exact" => exact, "rowhash_ok" => hok)))
            @printf "%-15s n=%7d %s x%d t%d | whole %.4f (first %.2f) | closure %.4f walls %s | structure+copies %.4f | reuse %.4f (check %s) | check %s\n" name n KIND g Threads.nthreads() median(t_tot) t_first median(t_c) join([@sprintf("%.3f", x) for x in w], " ") t_mg (length(t_re) > 1 ? median(t_re[2:end]) : NaN) reuse_ok ok
        catch err
            msg = sprint(showerror, err)[1:min(end, 300)]
            @printf "%-15s %d GPUs FAILED: %s\n" name g msg
            gsave("gpumulti", "$(name)_$(KIND)_g$(g)$(MSUFFIX)", Dict("code" => "ours", "graph" => name, "ngpus" => g, "n" => n, "ordering" => MORDER, "status" => "failed: " * msg))
        end
        flush(stdout); GC.gc(); cold!()
    end
    CUDA.device!(() -> X.free_plan!(Pl), devs(1)[1]); Pl = F = nothing
    A = A32 = R = nothing; GC.gc()
end
