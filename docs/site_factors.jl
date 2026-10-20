# PATHPACK.jl's factorizations for the Paris and bunny demos of the website.
#
#   julia --project=docs docs/site_factors.jl [refdir]
#
# For each demo, build exactly the graph the browser builds from its data file (docs/src/public/data/<name>.bin.gz),
# factorize it once with `mlu(MinPlus(), A)` in Float32, and write what one query `rmul!(B, F)` with a single row B
# reads to docs/src/public/data/<name>-lu.bin.gz. In the browser, docs/src/components/lib/pathpack.ts runs that query
# (PATHPACK's supernodal triangular solves, ported line for line) for the clicked source.
#
# The file format is the one of the other demo files (docs/src/components/lib/bin.ts): uint32 header length, JSON
# header {meta, arrays: name => [dtype, offset, count]}, arrays aligned to 4 bytes, all gzip-compressed.
#
# With `refdir`, also write reference rows computed by PATHPACK's own `rmul!` (and the graph's edges) to
# refdir/<name>-ref.bin.gz, for docs/site_factors_test.mjs, which checks the browser code against them.
using PATHPACK, SparseArrays, LinearAlgebra, Random
using PATHPACK.CPU: MinPlus, mlu, ChordalSLU, isintegral
using CliqueTrees.Multifrontal: pointers, targets, eltypedegree, nv, ne

const DATA = joinpath(@__DIR__, "src", "public", "data")

# ===== read a demo data file =====

function readbin(path)
    raw = read(pipeline(`gzip -dc $path`))
    hl = Int(reinterpret(UInt32, raw[1:4])[1])
    head = String(raw[5:4 + hl])
    base = 4 + hl
    types = Dict("uint8" => UInt8, "uint16" => UInt16, "uint32" => UInt32, "int16" => Int16, "int32" => Int32, "float32" => Float32)
    arrays = Dict{String, Vector}()
    for m in eachmatch(r"\"(\w+)\":\[\"(\w+)\",(\d+),(\d+)\]", head)
        T = types[m[2]]; off = parse(Int, m[3]); cnt = parse(Int, m[4])
        arrays[m[1]] = collect(reinterpret(T, raw[base + off + 1:base + off + cnt * sizeof(T)]))
    end
    return head, arrays
end

# the first occurrence of "key":value in the header (the meta numbers come before any string in it)
metanum(head, key) = parse(Float64, match(Regex("\"$key\":(-?[0-9.eE+-]+)"), head)[1])
metavec(head, key) = parse.(Float64, split(match(Regex("\"$key\":\\[([^\\]]*)\\]"), head)[1], ','))

# ===== the graphs, as the components build them =====

# ParisLive.vue: w[e] = a.wm[e] / 1000 stored in a Float32Array; csr(n, ei, ej, w) makes every street an arc both ways.
function parisgraph()
    head, a = readbin(joinpath(DATA, "paris.bin.gz"))
    n = Int(metanum(head, "n")); m = Int(metanum(head, "m"))
    I = Int.(a["ei"]) .+ 1
    J = Int.(a["ej"]) .+ 1
    W = Float32[Float32(a["wm"][e] / 1000) for e in 1:m]
    @assert length(I) == length(J) == m
    return n, I, J, W
end

# MeshPaint.vue: P[i] = meta.lo[i % 3] + a.pos[i] / meta.scale in a Float32Array; each edge of each triangle once
# (in the order the triangles list them), weighted by Math.hypot of the Float32 coordinate differences, stored in
# csr's Float32Array.
function bunnygraph()
    head, a = readbin(joinpath(DATA, "bunny.bin.gz"))
    n = Int(metanum(head, "n")); nf = Int(metanum(head, "f"))
    lo = metavec(head, "lo"); scale = metanum(head, "scale")
    pos = a["pos"]; tri = a["tri"]
    P = Float32[Float32(lo[(i - 1) % 3 + 1] + pos[i] / scale) for i in 1:3n]
    seen = Set{Int}(); I = Int[]; J = Int[]; W = Float32[]
    for f in 0:nf - 1
        i, j, k = Int(tri[3f + 1]), Int(tri[3f + 2]), Int(tri[3f + 3])
        for (p, q) in ((i, j), (j, k), (k, i))
            key = min(p, q) * 65536 + max(p, q)
            key in seen && continue
            push!(seen, key)
            dx = Float64(P[3p + 1]) - Float64(P[3q + 1])
            dy = Float64(P[3p + 2]) - Float64(P[3q + 2])
            dz = Float64(P[3p + 3]) - Float64(P[3q + 3])
            push!(I, p + 1); push!(J, q + 1); push!(W, Float32(jshypot(dx, dy, dz)))
        end
    end
    return n, I, J, W
end

# Math.hypot as V8 computes it (src/builtins/math.tq, MathHypot): scale by the largest magnitude, add the squares
# with Kahan's compensated summation, and scale back. The site's weights are Float32(Math.hypot(...)).
function jshypot(xs::Float64...)
    mx = 0.0
    for x in xs
        mx = max(mx, abs(x))
    end
    if mx == 0
        return 0.0
    end
    sum = 0.0; comp = 0.0
    for x in xs
        r = abs(x) / mx
        s = r * r - comp
        prel = sum + s
        comp = (prel - sum) - s
        sum = prel
    end
    return sqrt(sum) * mx
end

# the browser's graph as a matrix: arcs both ways, parallel arcs combined by ⊕ = min (as Dijkstra sees them)
function matrix(n, I, J, W)
    A = sparse(vcat(I, J), vcat(J, I), vcat(W, W), n, n, min)
    return A
end

# ===== the factor file =====

function smallest(v)
    m = isempty(v) ? 0 : maximum(v)
    @assert isempty(v) || minimum(v) >= 0
    return m <= typemax(UInt8) ? UInt8.(v) : m <= typemax(UInt16) ? UInt16.(v) : UInt32.(v)
end

dtype(::Type{UInt8}) = "uint8"
dtype(::Type{UInt16}) = "uint16"
dtype(::Type{UInt32}) = "uint32"
dtype(::Type{Float32}) = "float32"

function writebin(path, meta::String, arrays::Vector{<:Pair})
    off = 0; blobs = UInt8[]; parts = String[]
    for (k, v) in arrays
        push!(parts, "\"$k\":[\"$(dtype(eltype(v)))\",$off,$(length(v))]")
        b = collect(reinterpret(UInt8, v)); append!(b, zeros(UInt8, mod(-length(b), 4)))
        append!(blobs, b); off += length(b)
    end
    h = "{\"meta\":$meta,\"arrays\":{$(join(parts, ','))}}"
    h *= " "^mod(-sizeof(h), 4)
    raw = vcat(collect(reinterpret(UInt8, [UInt32(sizeof(h))])), Vector{UInt8}(codeunits(h)), blobs)
    tmp = path * ".raw"
    write(tmp, raw)
    run(pipeline(`gzip -9 -n -c $tmp`, stdout = path))
    rm(tmp)
    println(basename(path), ": ", round(filesize(path) / 1e3; digits = 1), " kB")
end

function factorfile(name, n, I, J, W, refdir)
    A = matrix(n, I, J, W)
    F = mlu(MinPlus(), A)
    S = F.S.S                                    # CliqueTrees ChordalSymbolic: res, sep, Dptr, Lptr
    @assert F isa ChordalSLU{MinPlus, Float32}
    @assert !isintegral(MinPlus())  # so the solves apply sstar to the diagonal of U

    nf = nv(S.res)
    Rptr = pointers(S.res)
    Sptr = pointers(S.sep); Stgt = targets(S.sep)
    #
    # The solves address the dense blocks through Dptr and Lptr, which are the running sums of nn² and nn⋅na: the
    # browser recomputes them from Rptr and Sptr.
    #
    Dp = 1; Lp = 1
    for f in 1:nf
        nn = eltypedegree(S.res, f); na = eltypedegree(S.sep, f)
        @assert S.Dptr[f] == Dp && S.Lptr[f] == Lp
        Dp += nn * nn; Lp += nn * na
    end
    @assert S.Dptr[nf + 1] == Dp && S.Lptr[nf + 1] == Lp
    #
    # rmul!(B, F) solves with U (diag :N, forward) and then with L (diag :U, backward). From the diagonal blocks, the
    # U solve reads the upper triangle and the diagonal of U.Dval, the L solve the strict lower triangle of L.Dval:
    # D holds exactly those entries, in the same layout.
    #
    D = similar(F.UDval)
    for f in 1:nf
        nn = eltypedegree(S.res, f); p = S.Dptr[f]
        for j in 1:nn, i in 1:nn
            o = p + (i - 1) + (j - 1) * nn
            D[o] = i <= j ? F.UDval[o] : F.LDval[o]
        end
    end

    nB = Int(F.S.nBptr)
    maxnn = maximum(eltypedegree(S.res, f) for f in 1:nf)
    maxna = maximum(eltypedegree(S.sep, f) for f in 1:nf)
    nvals = length(D) + length(F.ULval) + length(F.LLval) + length(F.Nval)
    println("$name: n = $n, nnz(A) = $(nnz(A)), fronts = $nf, components = $nB, largest front $maxnn + $maxna, ",
        "values: D $(length(D)) + U $(length(F.ULval)) + L $(length(F.LLval)) + N $(length(F.Nval)) = $nvals")

    meta = "{\"name\":\"$name\",\"n\":$n,\"nf\":$nf,\"nB\":$nB,\"semiring\":\"MinPlus\",\"eltype\":\"Float32\"," *
        "\"note\":\"PATHPACK.jl mlu(MinPlus(), A) of the demo graph; read by lib/pathpack.ts\"}"
    writebin(joinpath(DATA, "$name-lu.bin.gz"), meta, [
        "cinvp" => smallest(F.cinvp), "rperm" => smallest(F.rperm),
        "Rptr" => smallest(Rptr[1:nf + 1]), "Sptr" => smallest(Sptr[1:nf + 1]), "Stgt" => smallest(Stgt[1:Sptr[nf + 1] - 1]),
        "Bptr" => smallest(F.S.Bptr[1:nB + 1]), "Fptr" => smallest(F.S.Fptr[1:nB + 1]),
        "Nptr" => smallest(pointers(F.S.N)[1:n + 1]), "Ntgt" => smallest(targets(F.S.N)[1:ne(F.S.N)]),
        "Nval" => Vector{Float32}(F.Nval), "D" => Vector{Float32}(D),
        "UL" => Vector{Float32}(F.ULval), "LL" => Vector{Float32}(F.LLval),
    ])

    if !isnothing(refdir)
        # reference rows: rmul!(B, F) with B = fill(Inf, 1, n), B[s] = 0, one source at a time
        rng = Xoshiro(2026)
        srcs = unique(vcat([1, n, 2, n - 1], rand(rng, 1:n, 400)))[1:250]
        rows = Float32[]
        t = @elapsed for s in srcs
            B = fill(Inf32, 1, n); B[s] = 0f0
            rmul!(B, F)
            append!(rows, vec(B))
        end
        println("$name: $(length(srcs)) reference rows with rmul!, $(round(1e3 * t / length(srcs); digits = 3)) ms per row")
        meta = "{\"name\":\"$name\",\"n\":$n,\"m\":$(length(I)),\"k\":$(length(srcs))}"
        writebin(joinpath(refdir, "$name-ref.bin.gz"), meta, [
            "src" => UInt32.(srcs .- 1), "rows" => rows,
            "I" => UInt32.(I .- 1), "J" => UInt32.(J .- 1), "W" => W,
        ])
    end
    return F
end

refdir = isempty(ARGS) ? nothing : ARGS[1]
!isnothing(refdir) && mkpath(refdir)
factorfile("paris", parisgraph()..., refdir)
factorfile("bunny", bunnygraph()..., refdir)
