# The GPU baselines' runs (hpc/gpu_baselines.sh), checked and turned into result files. Usage:
#   julia gpu_baselines.jl rome GRAPH...       -> gpuapsp_<graph>_<gpu>_rome-<build>  rows of gpu_apsp.csv
#   julia gpu_baselines.jl fastapsp GRAPH...   -> gpumulti_<graph>_<gpu>_g<count>_fastapsp rows of gpu_multigpu.csv
#   julia gpu_baselines.jl others GRAPH...     -> gpuapsp_<graph>_<gpu>_<code> rows (cuGraph, Gunrock, ADDS, FW, ECL-APSP, Fast-APSP)
# A run that failed, or whose result does not pass its check, is reported as a failure (status), never as a time.
#
# ROME: timers tree (Scotch ordering, elimination tree, supernodes), alloc (cudaMallocPitch of the n x n matrix, the graph
# uploaded, the matrix initialized: infinity, then the arcs; the metadata buffers), computing. Three scopes:
#   t_rome   = tree + computing                  ROME's own time (as its authors report it); leaves out the initialization of the matrix
#   t_match  = tree + alloc - mallocPitch + computing   our rule: the output allocated before the timer, everything else in
#                                                (mallocPitch from the instrumented build's split of alloc, same graph)
#   t_full   = tree + alloc + computing          including the allocation of the output
# The result stays in device memory (ROME_DEVICE_ONLY), as ours: neither copies it to the host. Check: the 64 sampled rows
# (run 1 of each build) against Dijkstra in Float64, relative tolerance RTOL (gpu_lib.jl: check_cols).
# Fast-APSP: canonical = conversion from our CSR + METIS + preconditioning + solve, the result in host memory (its design:
# it allocates the n x n result and a path matrix inside the timer, and copies to the host as it goes). Check: its row
# hashes against ours (every row; integer weights) and the run's status.
include("gpu_lib.jl")
const KIND = gpukind()
mode = ARGS[1]; graphs_arg = ARGS[2:end]
# (recheck: every mode, again, over the task folders in BENCH_RECHECK_DIRS, graphs from the logs found there)
secs(txt, pat) = (m = match(pat, txt); m === nothing ? nothing : parse(Float64, m[1]))
const RE_TREE = r"Time elapsed of tree build: ([\d.]+) seconds"
const RE_ALLOC = r"Time elapsed for GPU memory allocation and initialization: ([\d.]+) seconds"
const RE_COMP = r"Time elapsed of computing: ([\d.]+) seconds"
const RE_PITCH = r"\[INSTR\] alloc_init: mallocPitch\(n\^2\) ([\d.]+)"
const RE_WALL = r"PROCESS_WALL ([\d.]+) rc=(-?\d+)"
const RE_WORK = r"ROME work: (\d+), FW work (\d+)"
function result_line(txt)
    l = findfirst(x -> startswith(x, "RESULT "), split(txt, '\n'))
    return l === nothing ? Dict{String, Any}() : (try JSON.parse(split(txt, '\n')[l][8:end]) catch; Dict{String, Any}("status" => "bad result line") end)
end
function readdump(path, n)
    rows = Dict{Int, Vector{Float64}}()
    isfile(path) || return rows
    for l in eachline(path)
        t = split(l); length(t) == n + 1 || continue
        rows[parse(Int, t[1])] = [x == "Inf" ? Inf : parse(Float64, x) for x in t[2:end]]
    end
    return rows
end

function parse_mode(mode, OUTD, graphs)
if mode == "rome"
    for name in graphs
        A = graph(name); n = size(A, 1); srcs = sample_vertices(n)
        R = ref_dijkstra(sparse(transpose(A)), srcs)                   # column t = row srcs[t] of A*
        local d = joinpath(OUTD, "rome")
        il = joinpath(d, "$name.instr.r1.log"); itxt = isfile(il) ? read(il, String) : ""
        pitch = secs(itxt, RE_PITCH)
        for v in split(get(ENV, "ROME_VARIANTS", "default leaf400 leaf800"))
            logs = sort(filter(f -> startswith(f, "$name.$v.r") && endswith(f, ".log"), isdir(d) ? readdir(d) : String[]))
            isempty(logs) && continue
            runs = Dict{String, Any}[]
            for f in logs
                txt = read(joinpath(d, f), String); w = match(RE_WALL, txt)
                t, a, c = secs(txt, RE_TREE), secs(txt, RE_ALLOC), secs(txt, RE_COMP)
                ok = w !== nothing && w[2] == "0" && t !== nothing && a !== nothing && c !== nothing
                wk = match(RE_WORK, txt)
                push!(runs, Dict("log" => f, "ok" => ok, "rc" => w === nothing ? nothing : parse(Int, w[2]), "tree" => t, "alloc" => a, "computing" => c,
                    "process_wall" => w === nothing ? nothing : parse(Float64, w[1]), "rome_work" => wk === nothing ? nothing : parse(Float64, wk[1]),
                    "error" => ok ? nothing : join(last(filter(!isempty, split(txt, '\n')), 3), " | ")))
            end
            good = filter(r -> r["ok"], runs)
            dump = readdump(joinpath(d, "$name.$v.dump"), n)
            G = fill(NaN32, n, length(srcs)); have = 0
            for (t, v0) in enumerate(srcs); haskey(dump, v0) && (G[:, t] = dump[v0]; have += 1); end
            c = have == length(srcs) ? check_cols(G, R) : (ok = false, pattern = false, maxrel = NaN)
            status = isempty(good) ? "failed: " * something(runs[end]["error"], "no time") : !c.ok ? (have < length(srcs) ? "check failed: rows not dumped" : "check failed") : length(good) < length(runs) ? "ok (some runs failed)" : "ok"
            r = Dict{String, Any}("graph" => name, "code" => "rome-$v", "status" => status, "n" => n, "m" => nnz(A), "runs" => runs, "nruns_ok" => length(good),
                "check" => Dict("ok" => c.ok, "rows" => have, "maxrel" => c.maxrel, "rtol" => RTOL, "pattern" => c.pattern),
                "t_mallocpitch" => pitch, "output" => "device (ROME_DEVICE_ONLY), ROME's own order (its elimination order); not relabelled",
                "threads" => get(ENV, "GPU_CPUS", ""), "build" => "rome_$v")
            if !isempty(good) && c.ok
                tr = [g["tree"] for g in good]; al = [g["alloc"] for g in good]; co = [g["computing"] for g in good]
                r["t_tree"] = summ(tr); r["t_alloc"] = summ(al); r["t_computing"] = summ(co)
                r["t_rome"] = summ(tr .+ co); r["t_full"] = summ(tr .+ al .+ co)
                pitch === nothing || (r["t_match"] = summ(tr .+ al .- pitch .+ co); r["t_total"] = median(tr .+ al .- pitch .+ co))
            end
            gsave("gpuapsp", "$(name)_$(KIND)_rome-$v", r; dir = OUTD)
            @printf "%-15s rome %-8s %s | tree+comp %s  matched %s  full %s | check %s (%d rows, max rel %.1e)\n" name v status get(r, "t_rome", Dict("median" => NaN))["median"] get(r, "t_match", Dict("median" => NaN))["median"] get(r, "t_full", Dict("median" => NaN))["median"] c.ok have c.maxrel
        end
    end
elseif mode == "fastapsp"
    fd = joinpath(OUTD, "fastapsp")
    for name in graphs, c in 1:8
        logs = isdir(fd) ? sort(filter(f -> startswith(f, "$name.g$c.k") && endswith(f, ".log"), readdir(fd))) : String[]
        isempty(logs) && continue
        ours = isfile(joinpath(EXPORT, "$name.fp")) ? read_fp(joinpath(EXPORT, "$name.fp")) : nothing
        exact = isfile(joinpath(EXPORT, "$name.rowhash"))
        runs = Dict{String, Any}[]
        for f in logs
            txt = read(joinpath(fd, f), String); w = match(RE_WALL, txt)
            res = result_line(txt)
            k = parse(Int, match(r"\.k(\d+)\.", f)[1]); tag = match(r"\.k\d+\.(\w+)\.log$", f)[1]
            ok = w !== nothing && w[2] == "0" && get(res, "status", "") == "ok" && get(res, "canonical", nothing) isa Number
            fpf = filter(x -> startswith(x, replace(f, ".log" => ".fp")), readdir(fd))
            fpc = (isempty(fpf) || ours === nothing) ? nothing : compare_fp(reduce(vcat, [readlines(joinpath(fd, x)) for x in fpf]), ours)
            push!(runs, Dict("log" => f, "K" => k, "tag" => tag, "ok" => ok, "canonical" => get(res, "canonical", nothing), "inproc" => get(res, "inproc", nothing),
                "gpu_compute" => get(res, "gpu_compute", nothing), "phases" => get(res, "phases", nothing), "rowhash_ok" => get(res, "rowhash_ok", nothing),
                "rows_hashed" => get(res, "rows_hashed", nothing), "checksum" => get(res, "checksum", nothing),
                "fp" => fpc === nothing ? nothing : Dict("rows" => fpc.rows, "bad" => fpc.bad, "maxrel" => fpc.maxrel, "unknown" => fpc.unknown, "ok" => fpc.ok),
                "maxrss_kb" => (m = match(r"MAXRSS_KB (\d+)", txt); m === nothing ? nothing : parse(Int, m[1])),
                "rc" => w === nothing ? nothing : parse(Int, w[2]), "error" => ok ? nothing : join(last(filter(!isempty, split(txt, '\n')), 3), " | ")))
        end
        timed_runs = filter(r -> r["tag"] != "sweep", runs)
        fpok = [r["fp"]["ok"] for r in timed_runs if r["fp"] !== nothing]
        verified = exact ? all(r -> r["rowhash_ok"] === true, filter(r -> r["ok"], timed_runs)) : (!isempty(fpok) && all(fpok))
        good = filter(r -> r["ok"], timed_runs)
        bestK = isempty(timed_runs) ? nothing : timed_runs[1]["K"]
        status = isempty(timed_runs) ? "failed: " * something(runs[end]["error"], "no run") : isempty(good) ? "failed: " * something(timed_runs[end]["error"], "?") :
                 !verified ? (exact ? "check failed (row hashes)" : isempty(fpok) ? "unverified (no fingerprints)" : "check failed (fingerprints)") : "ok"
        r = Dict{String, Any}("graph" => name, "code" => "fastapsp", "ngpus" => c, "status" => status, "K" => bestK, "runs" => runs, "nruns_ok" => length(good),
            "check" => Dict("method" => exact ? "row hashes (all rows)" : "fingerprints (all rows, run 1)", "ok" => verified, "rtol" => RTOL),
            "output" => "host (Fast-APSP's design), its own row order; a path matrix computed too",
            "K_floor_int32" => (nn = (ln = filter(l -> !startswith(l, '%'), collect(Iterators.take(eachline(joinpath(EXPORT, "$name.general.mtx")), 200))); isempty(ln) ? nothing : parse(Int, split(ln[1])[1]));
                                nn === nothing ? nothing : nn^2 / 2^31),
            "K_note" => "Fast-APSP's kernels index with 32-bit integers, so it needs at least n^2/2^31 partitions (K); K swept over count x 2^j above that floor",
            "scope" => "canonical: conversion from CSR + METIS + preconditioning + solve; output allocation inside the timer")
        if status == "ok"
            r["t_total"] = median([g["canonical"] for g in good]); r["t"] = summ([Float64(g["canonical"]) for g in good])
            r["t_gpu_compute"] = summ([Float64(g["gpu_compute"]) for g in good])
        end
        gsave("gpumulti", "$(name)_$(KIND)_g$(c)_fastapsp", r; dir = OUTD)
        @printf "%-15s fastapsp %d GPUs K=%s %s | canonical %s\n" name c something(bestK, "-") status get(r, "t_total", NaN)
    end
elseif mode == "others"
    od = joinpath(OUTD, "others")
    for name in graphs
        ours = isfile(joinpath(EXPORT, "$name.fp")) ? read_fp(joinpath(EXPORT, "$name.fp")) : nothing
        exact = isfile(joinpath(EXPORT, "$name.rowhash"))
        for code in split(get(ENV, "OTHERS", "cugraph gunrock adds fw ecl fastapsp1"))
            logs = isdir(od) ? sort(filter(f -> startswith(f, "$name.$code.r") && endswith(f, ".log"), readdir(od))) : String[]
            isempty(logs) && continue
            runs = Dict{String, Any}[]
            for f in logs
                txt = read(joinpath(od, f), String); w = match(RE_WALL, txt); res = result_line(txt)
                startswith(txt, "others ") && (push!(runs, Dict("log" => f, "ok" => false, "skipped" => strip(txt))); continue)
                # (the drivers' own check of sampled rows, a CPU Dijkstra in int64 arithmetic, is meaningful only for integer
                #  weights: with real weights it reports row_mismatch whatever the rows; those are checked by fingerprints)
                st = get(res, "status", "")
                ok = w !== nothing && w[2] == "0" && (st == "ok" || (!exact && st == "row_mismatch"))
                fpf = joinpath(od, replace(f, ".log" => ".fp"))
                fpc = (isfile(fpf) && ours !== nothing) ? compare_fp(readlines(fpf), ours) : nothing
                t = code == "fastapsp1" ? get(res, "canonical", nothing) : get(res, "inproc", nothing)
                push!(runs, Dict("log" => f, "ok" => ok, "time" => t, "inproc" => get(res, "inproc", nothing), "gpu_compute" => get(res, "gpu_compute", nothing),
                    "e2e_inproc" => get(res, "e2e_inproc", nothing), "extrapolated" => get(res, "extrapolated", false), "sources_run" => get(res, "sources_run", nothing),
                    "rows_checked" => get(res, "rows_checked", nothing), "rows_bad" => get(res, "rows_bad", nothing), "rowhash_ok" => get(res, "rowhash_ok", nothing),
                    "rows_hashed" => get(res, "rows_hashed", nothing), "phases" => get(res, "phases", nothing), "driver_status" => get(res, "status", nothing),
                    "fp" => fpc === nothing ? nothing : Dict("rows" => fpc.rows, "bad" => fpc.bad, "maxrel" => fpc.maxrel, "unknown" => fpc.unknown, "ok" => fpc.ok),
                    "rc" => w === nothing ? nothing : parse(Int, w[2]), "error" => ok ? nothing : join(last(filter(!isempty, split(txt, '\n')), 3), " | ")))
            end
            skipped = all(r -> haskey(r, "skipped"), runs)
            good = filter(r -> r["ok"] && r["time"] isa Number && r["time"] > 0, runs)
            fps = [r["fp"] for r in runs if get(r, "fp", nothing) !== nothing]
            hashok = [r["rowhash_ok"] for r in good if r["rowhash_ok"] !== nothing]
            # (row hashes when the run had ours to compare with; runs from before ours wrote them rest on the fingerprints)
            hashed = exact && !isempty(hashok)
            verified = !isempty(fps) && all(x -> x["ok"], fps) && (!hashed || all(==(true), hashok))
            killed = any(r -> get(r, "rc", nothing) == 137, runs)   # (a later run may be cut off by the job's end: no rc)
            status = skipped ? "not run: " * runs[1]["skipped"] : isempty(good) ?
                     (killed ? "failed: timeout (killed at $(get(ENV, "OTHERS_TIMEOUT", "900")) s)" : "failed: " * something(runs[end]["error"], "?")) :
                     !verified ? (isempty(fps) ? "unverified (no fingerprints)" : "check failed") : "ok"
            r = Dict{String, Any}("graph" => name, "code" => code, "status" => status, "runs" => runs, "nruns_ok" => length(good),
                "extrapolated" => any(r -> r["extrapolated"] == true, good),
                "check" => Dict("ok" => verified, "method" => "fingerprints of every produced row (run 1)" * (hashed ? " and row hashes" : ""), "rtol" => RTOL),
                "scope" => code == "fastapsp1" ? "canonical: conversion + METIS + preconditioning + solve, result in host memory" :
                           "inproc: graph upload, allocation, all sources (extrapolated from a sample when marked), result on the device")
            status == "ok" && (r["t_total"] = median([Float64(g["time"]) for g in good]); r["t"] = summ([Float64(g["time"]) for g in good]))
            gsave("gpuapsp", "$(name)_$(KIND)_$(code)", r; dir = OUTD)
            @printf "%-15s %-9s %s | %s%s\n" name code status get(r, "t_total", NaN) r["extrapolated"] ? " (extrapolated)" : ""
        end
    end
end
end   # parse_mode

logs_graphs(dir, re) = isdir(dir) ? sort(unique([m[1] for m in (match(re, f) for f in readdir(dir)) if m !== nothing])) : String[]
if mode == "recheck"
    for d in split(get(ENV, "BENCH_RECHECK_DIRS", RESULTS), ':'; keepempty = false)
        isdir(d) || error("recheck: no folder $d (from $(pwd()))")
        g1 = logs_graphs(joinpath(d, "rome"), r"^(.+)\.(default|leaf400|leaf800|instr)\.r\d+\.log$")
        g2 = logs_graphs(joinpath(d, "others"), r"^(.+)\.(cugraph|gunrock|adds|fw|ecl|fastapsp1)\.r\d+\.log$")
        g3 = logs_graphs(joinpath(d, "fastapsp"), r"^(.+)\.g\d+\.k\d+\.\w+\.log$")
        isempty(g1) || parse_mode("rome", d, g1)
        isempty(g2) || withenv(() -> parse_mode("others", d, g2), "OTHERS" => nothing)
        isempty(g3) || parse_mode("fastapsp", d, g3)
    end
else
    parse_mode(mode, RESULTS, graphs_arg)
end