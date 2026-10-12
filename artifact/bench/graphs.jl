# The graphs: three sets, each with its weight convention. A[i, j] is the weight of arc i -> j.
#   APSP:    the union of the SuperFW (Sao et al., PPoPP'20, Table 3) and ROME (Luo et al., PPoPP'26, Table 2) test sets.
#            SuiteSparse matrices carry SuperFW's convention: weight |a_ij| (pattern matrices: 1), symmetric.
#            Graphs without values (SNAP, synthetic) get uniform weights in [1, 100] with a fixed seed.
#   SSSP:    the eleven DIMACS road networks (travel times), a few all-pairs graphs, and two high-fill graphs.
#   SCALING: SuperFW's strong-scaling graphs (their Fig. 7).
#   FASTAPSP: Fast-APSP's test graphs (Yang et al., ICS'23), for the multi-GPU runs: SuiteSparse (DIMACS10, Gleich) and
#            web-sk-2005 (Network Repository), all pattern matrices, so with the SuiteSparse convention: unit weights.
#   LARGE1:  larger graphs on one GPU (Fast-APSP's, and DIMACS10 delaunay_n17 and rgg_n_2_17_s0, unit weights).
#   LARGE2:  one GPU near its memory limit (SuiteSparse 144, wave [unit weights], Dubcova3, c-73, cont-300, thermomech_dM).
#   HUGE:    delaunay_n19 on 7-8 GPUs (the largest closure that fits on one 8 x B200 node).
#   ROADS:   DIMACS road networks whose n x n output exceeds one GPU (travel times), for several GPUs.
using CodecZlib, Graphs

const APSP = ["USpowerGrid", "delaunay_n13", "G67", "c-42", "CurlCurl_0", "fe_sphere", "nd6k", "delaunay_n14",
              "OPF_6000", "EB_8192_256", "EB_16384_64", "rgg2d_14", "rgg3d_14", "hypercube_14",
              "lpl1", "email-Enron", "t60k", "wing", "delaunay_n16", "c-69", "oilpan", "finan512", "fe_tooth",
              "onera_dual", "net4-1", "luxembourg_osm"]
const DIMACS = ["NY", "BAY", "COL", "FLA", "NW", "NE", "CAL", "E", "W", "CTR", "USA"]
const SSSP = vcat(["USA-t-" * r for r in DIMACS], ["delaunay_n16", "wing", "c-69"], ["email-Enron", "soc-Epinions1"])
const SCALING = ["finan512", "net4-1", "email-Enron", "wing"]
const FASTAPSP = ["598a", "luxembourg_osm", "web-sk-2005", "usroads-48", "fe_ocean"]
const LARGE1 = ["598a", "luxembourg_osm", "web-sk-2005", "usroads-48", "fe_ocean", "delaunay_n17", "rgg_n_2_17_s0"]   # one GPU, n ~ 100k-150k
const LARGE2 = ["144", "Dubcova3", "wave", "c-73", "cont-300", "thermomech_dM"]   # one GPU near its limit, n ~ 145k-205k
const HUGE = ["delaunay_n19"]                   # several GPUs only: n = 524,288 (1.1 TB of Float32 distances), 7-8 B200s
const ROADS = ["USA-t-NY", "USA-t-BAY", "USA-t-COL"]                                                              # more than one GPU holds

# ---------------- readers ----------------
function read_mtx_values(name)                  # SuperFW's convention: |a_ij| (pattern: 1; explicit zeros: 1), symmetric, min over duplicates
    path = joinpath(DATA, "mtx", name, name * ".mtx")
    W = Dict{Tuple{Int, Int}, Float64}(); n = 0
    open(path) do io
        pattern = occursin("pattern", lowercase(readline(io)))
        line = readline(io); while startswith(line, "%"); line = readline(io); end
        n = parse(Int, split(line)[1])
        for l in eachline(io)
            isempty(l) && continue
            t = split(l); i = parse(Int, t[1]); j = parse(Int, t[2]); i == j && continue
            w = pattern ? 1.0 : abs(parse(Float64, t[3])); w == 0 && (w = 1.0)
            key = (min(i, j), max(i, j)); W[key] = min(get(W, key, Inf), w)
        end
    end
    I = Int[]; J = Int[]; V = Float64[]
    for ((i, j), w) in W; push!(I, i, j); push!(J, j, i); push!(V, w, w); end
    return sparse(I, J, V, n, n)
end
function read_snap(name)                        # SNAP edge list -> (I, J, n) with contiguous ids, no self-loops
    I = Int[]; J = Int[]; ids = Dict{Int, Int}(); id(x) = get!(ids, x, length(ids) + 1)
    open(joinpath(DATA, name * ".txt.gz")) do io
        for l in eachline(GzipDecompressorStream(io))
            (isempty(l) || l[1] == '#') && continue
            a, b = split(l); u = id(parse(Int, a)); v = id(parse(Int, b)); u == v || (push!(I, u); push!(J, v))
        end
    end
    return I, J, length(ids)
end
function read_gr(region)                        # DIMACS travel times; duplicate arcs: the minimum
    I = Int[]; J = Int[]; W = Float64[]; n = 0
    for l in eachline(joinpath(DATA, "gr", "USA-road-t_$(region).gr"))
        isempty(l) && continue
        if l[1] == 'a'; t = split(l); push!(I, parse(Int, t[2])); push!(J, parse(Int, t[3])); push!(W, parse(Float64, t[4]))
        elseif l[1] == 'p'; n = parse(Int, split(l)[3]); end
    end
    keep = I .!= J
    return sparse(I[keep], J[keep], W[keep], n, n, min)
end
function weighted(I, J, n; undirected, seed = 1)   # uniform weights in [1, 100]; undirected: symmetric pattern and weights
    rng = MersenneTwister(seed)
    if undirected
        r, c, _ = findnz(tril(sparse(vcat(I, J), vcat(J, I), true, n, n, |), -1)); w = 1 .+ 99 .* rand(rng, length(r))
        return sparse(vcat(r, c), vcat(c, r), vcat(w, w), n, n)
    end
    r, c, _ = findnz(sparse(I, J, true, n, n, |)); return sparse(r, c, 1 .+ 99 .* rand(rng, length(r)), n, n)
end
function lcc(A)                                 # largest weakly connected component
    cc = weakly_connected_components(SimpleDiGraph(A .!= 0)); c = sort(cc[argmax(length.(cc))]); return A[c, c]
end
fromgraph(g; seed = 1) = weighted([src(e) for e in edges(g)], [dst(e) for e in edges(g)], nv(g); undirected = true, seed)
function rgg_exact(n, deg, dim; seed = 1)       # random geometric graph with exact average degree: the n*deg/2 closest pairs
    rng = MersenneTwister(seed); X = rand(rng, dim, n); vol = dim == 2 ? pi : 4pi / 3; m = round(Int, n * deg / 2)
    for grow in (1.35, 1.7, 2.2)
        r = grow * (deg / (vol * n))^(1 / dim); r2 = r^2; nb = max(1, floor(Int, 1 / r)); cell(v) = min(nb, floor(Int, v * nb) + 1)
        buckets = Dict{NTuple{dim, Int}, Vector{Int32}}()
        for i in 1:n; push!(get!(buckets, ntuple(k -> cell(X[k, i]), dim), Int32[]), i); end
        D = Float64[]; I = Int32[]; J = Int32[]; offs = collect(Iterators.product(ntuple(_ -> -1:1, dim)...))
        for i in 1:n, o in offs
            b = get(buckets, ntuple(k -> cell(X[k, i]), dim) .+ o, nothing); b === nothing && continue
            for j in b
                j > i || continue
                d = 0.0; for k in 1:dim; d += (X[k, i] - X[k, j])^2; end
                d <= r2 && (push!(D, d); push!(I, i); push!(J, j))
            end
        end
        length(D) >= m || continue
        keep = partialsortperm(D, 1:m); return weighted(Int.(I[keep]), Int.(J[keep]), n; undirected = true, seed)
    end
    error("rgg_exact: could not reach average degree $deg")
end

const SYNTHETIC = Dict(                         # SuperFW's synthetic graphs, generated to their Table 3 parameters
    "rgg2d_14" => () -> rgg_exact(2^14, 128.17, 2), "rgg3d_14" => () -> rgg_exact(2^14, 910.0, 3),
    "EB_8192_256" => () -> fromgraph(barabasi_albert(8192, 130; seed = 1)), "EB_16384_64" => () -> fromgraph(barabasi_albert(16384, 32; seed = 1)),
    "hypercube_14" => () -> fromgraph(Graphs.grid(fill(2, 14))))
const SNAP_DIRECTED = Set(["soc-Epinions1", "wiki-Vote", "p2p-Gnutella31", "cit-HepPh", "email-EuAll"])

function graph(name)
    startswith(name, "USA-t-") && return read_gr(name[7:end])
    haskey(SYNTHETIC, name) && return SYNTHETIC[name]()
    isfile(joinpath(DATA, "mtx", name, name * ".mtx")) && return read_mtx_values(name)
    if isfile(joinpath(DATA, name * ".txt.gz"))
        name in SNAP_DIRECTED && return weighted(read_snap(name)...; undirected = false)
        return lcc(weighted(read_snap(name)...; undirected = true))
    end
    error("graph $name: no data under $DATA (run hpc/setup.sh)")
end
