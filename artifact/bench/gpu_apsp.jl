# All-pairs shortest paths on one GPU through the library's public API, apsp_gpu!(D, A; alg): Float32, into an n x n
# device matrix allocated and first touched before the timer (t_alloc). Per graph, one result file per variant:
#   ordering   bestfill = BestFill(AMF(), METIS()), the rule of the CPU all-pairs runs (apsp.jl), the main
#              number; amf = AMF alone, recorded alongside
#   factor     hybrid (bottom subtrees on CPU threads, top fronts on the GPU; the library's default) and cpu (the whole
#              numeric factorization on the CPU, factor_gpu = false)
# and for each:
#   t_first        the first call in this process (compiles what this graph needs; excluded)
#   t_warm, t_cold timed calls (conventions in gpu_lib.jl); t_total = median cold call, the headline
#   phases         with_phases on one cold and one warm call: symbolic, numeric setup / CPU / GPU, transfer, solve setup,
#                  inverse (U*), solve (L*), relabel, other; synchronized around every step, so they add up to more
#                  than t_total (phases.total); t_alloc and the reference times are outside all of these
# The result is in the labels of A (D[i, j] = A*[i, j]): the renumbering is inside the call (phase "relabel", 0 when the
# closure writes the columns in place); the CPU rows exclude theirs (t_perm), so t_total - relabel is their counterpart.
# Checks, per variant: the output prefilled with NaN has none left after a call (every entry written); 64 columns
# (apsp.jl's) against Dijkstra and against the CPU's Float64 closure from PATHPACK's CPU factorization (gpu_lib.jl:
# check_cols); with integer weights, the row hashes of the whole result; per-row fingerprints of the whole result
# (gpu_lib.jl: fingerprints), written with the row hashes for the baselines' checks.
# Also writes each graph for the baselines (gpu_lib.jl: export_graph). Usage: julia -t T gpu_apsp.jl GRAPH...
include("gpu_lib.jl")
s = MinPlus()
const KIND = gpukind()
const FASTAPSP_SET = Set(FASTAPSP)
# (BENCH_VARIANTS "o:f o:f ...", e.g. "auto:hybrid auto:cpu amf:hybrid", instead of every ordering with every factor)
const VARIANTS = haskey(ENV, "BENCH_VARIANTS") ? [Tuple(split(v, ':')) for v in split(ENV["BENCH_VARIANTS"])] :
                 [(o, f) for o in split(get(ENV, "BENCH_ORDERINGS", "bestfill amf")) for f in split(get(ENV, "BENCH_FACTORS", "hybrid cpu"))]
vname(o, f) = "ours-$o" * (f == "cpu" ? "-cpufactor" : "")
get(ENV, "BENCH_EXPORT_ONLY", "0") == "1" || let A = SparseMatrixCSC{Float32, Int}(fromgraph(Graphs.grid([12, 12]))); n = size(A, 1)     # compile the common paths
    D = CuMatrix{Float32}(undef, n, n)
    for (o, f) in VARIANTS; closure!(D, A; alg = ORDERINGS[o], factor_gpu = f == "hybrid"); phases_of(() -> closure!(D, A; alg = ORDERINGS[o], factor_gpu = f == "hybrid")); end
    rowhashes(D, 1:n); CUDA.unsafe_free!(D); GC.gc()
end
for name in ARGS
    get(ENV, "BENCH_EXPORT_ONLY", "0") == "1" || !all(((o, f),) -> isfile(gresultfile("gpuapsp", "$(name)_$(KIND)_$(vname(o, f))")), VARIANTS) || continue
    A = graph(name); n = size(A, 1); A32 = SparseMatrixCSC{Float32, Int}(A)
    export_graph(name, A; general = true)
    get(ENV, "BENCH_EXPORT_ONLY", "0") == "1" && continue        # (only the graph files for the baselines)
    if 4.0 * n^2 * parse(Float64, get(ENV, "BENCH_MEM_MARGIN", "1.1")) > CUDA.totalmem(CUDA.device())   # (margin 1.0: let the library's own memory check decide)
        @printf "%-15s n=%d: the closure needs %.1f GB, more than this GPU (%s) holds: skipped\n" name n 4.0 * n^2 / 1e9 CUDA.name(CUDA.device())
        for (o, f) in VARIANTS
            gsave("gpuapsp", "$(name)_$(KIND)_$(vname(o, f))", Dict("graph" => name, "code" => vname(o, f), "n" => n, "m" => nnz(A), "status" => "too large for one GPU"))
        end
        continue
    end
    cols = sample_vertices(n)
    t_ref = @elapsed begin R = ref_dijkstra(A, cols); Rf, st = ref_cpu_factor(A, cols) end      # CPU, Float64
    exact = integral(A) && n * maximum(nonzeros(A); init = 0.0) < 2^24
    cold!()
    D, t_map, t_touch = alloc_output(Float32, n, n)
    for (o, f) in VARIANTS
        code = vname(o, f); alg = ORDERINGS[o]; fg = f == "hybrid"
        isfile(gresultfile("gpuapsp", "$(name)_$(KIND)_$(code)")) && continue
        call() = closure!(D, A32; alg, factor_gpu = fg)
        local t_first, t_warm, t_cold, ph_cold, ph_warm, nnan, c_dij, c_cpu, hc, ok
        try
            _, t_first = timed(call)
            t_warm = [timed(call)[2] for _ in 1:WREPS]
            t_cold = [(cold!(); timed(call)[2]) for _ in 1:REPS]
            cold!(); ph_cold = phases_of(call); ph_warm = phases_of(call)
            # ---- the checked call ----
            fill!(D, NaN32); syncall(); call(); syncall()
            nnan = count(isnan, D)
            G = Array(D[:, cols]); c_dij = check_cols(G, R); c_cpu = check_cols(G, Rf)
            (o, f) == VARIANTS[1] && write_fp(name, fingerprints(D, 1:n, probes_of(n)))      # (for the baselines' checks)
            hc = nothing
            if exact
                h = rowhashes(D, 1:n)
                (o, f) == VARIANTS[1] && writehash(joinpath(EXPORT, "$name.rowhash"), h)
                hc = string(foldl((a, (i, x)) -> a + x * (UInt64(i) * 0xD6E8FEB86659FD93 + 1), enumerate(h); init = UInt64(0)); base = 16, pad = 16)
            end
            ok = nnan == 0 && c_dij.ok && c_cpu.ok
        catch err
            msg = sprint(showerror, err)[1:min(end, 300)]
            @printf "%-15s %s FAILED: %s\n" name code msg
            gsave("gpuapsp", "$(name)_$(KIND)_$(code)", Dict("graph" => name, "code" => code, "n" => n, "m" => nnz(A), "status" => "failed: " * msg)); continue
        end
        ok || @printf "%-15s %s CHECK FAILED: NaN left %d, vs Dijkstra %s (max rel %.2e), vs CPU closure %s (max rel %.2e)\n" name code nnan c_dij.ok c_dij.maxrel c_cpu.ok c_cpu.maxrel
        gsave("gpuapsp", "$(name)_$(KIND)_$(code)", Dict("graph" => name, "code" => code, "status" => ok ? "ok" : "check failed", "n" => n, "m" => nnz(A),
            "ordering" => ordering_label(o), "auto_choice" => o == "auto" ? auto_choice(A) : "", "factor" => f, "ncc" => st.ncc,
            "fill" => st.nnzLU / nnz(A), "nnzLU" => st.nnzLU, "maxfront" => st.maxfront, "muladds" => st.muladds,       # (of the CPU's BestFill factor)
            "eltype" => "Float32", "output" => "device, preallocated (apsp_gpu!); original labels",
            "t_total" => median(t_cold), "t_cold" => summ(t_cold), "t_warm" => summ(t_warm), "t_first" => t_first,
            "t_alloc" => t_map + t_touch, "t_alloc_map" => t_map, "t_alloc_touch" => t_touch,
            "phases" => ph_cold, "phases_warm" => ph_warm,
            "check" => Dict("ok" => ok, "nan_left" => nnan, "rtol" => RTOL, "columns" => length(cols), "dijkstra_ok" => c_dij.ok, "dijkstra_maxrel" => c_dij.maxrel,
                "cpu_closure_ok" => c_cpu.ok, "cpu_closure_maxrel" => c_cpu.maxrel, "exact" => exact, "rowhash" => hc),
            "t_reference" => t_ref))
        p = ph_cold
        @printf "%-15s n=%7d %s %-24s | total %.4f (cold med; warm %.4f; first %.2f) alloc %.4f | sym %.4f numset %.4f cpu %.4f gpu %.4f xfer %.4f setup %.4f inv %.4f solve %.4f relabel %.4f other %.4f (sum %.4f) | check %s (%.1e)\n" name n KIND code median(t_cold) median(t_warm) t_first t_map + t_touch p["symbolic"] p["numeric_setup"] p["numeric_cpu"] p["numeric_gpu"] p["transfer"] p["solve_setup"] p["inverse"] p["solve"] p["relabel"] p["other"] p["total"] ok max(c_dij.maxrel, c_cpu.maxrel)
        flush(stdout)
    end
    CUDA.unsafe_free!(D); D = nothing; GC.gc(); CUDA.reclaim()
    A = A32 = R = Rf = nothing; GC.gc()
end
