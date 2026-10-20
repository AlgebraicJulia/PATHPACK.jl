# Explorable code cells: for each cell, run the code with PATHPACK.jl for every combination of its parameters and
# save what Julia prints, so the page can swap outputs instantly. Run by make.jl before makedocs; writes
# docs/src/public/explore/<id>.json, read by the <CodeExplorer id="..."/> component.
#
# A cell: `code` with `{name}` slots; `params` = name => (label, values); `run(values...)` returns the value to show.
using PATHPACK, SparseArrays, LinearAlgebra, Random
using PATHPACK.CPU: MinPlus, MaxMin, MinMax, MaxPlus, MaxProd, PlusProd, mstar, mlu

const OUT = joinpath(@__DIR__, "src", "public", "explore")
mkpath(OUT)

shown(x) = sprint((io, v) -> show(IOContext(io, :limit => true, :displaysize => (24, 80)), MIME"text/plain"(), v), x)
jstr(s) = "\"" * escape_string(s) * "\""

function write_cell(id; code, params, run, note = "")
    names = first.(params)
    keys_ = String[]; outs = String[]
    for combo in Iterators.product((last(last(p)) for p in params)...)
        push!(keys_, join(string.(combo), "|")); push!(outs, shown(run(combo...)))
    end
    open(joinpath(OUT, "$id.json"), "w") do io
        print(io, "{\"code\":", jstr(code), ",\"note\":", jstr(note), ",\"params\":[")
        join(io, ("{\"name\":$(jstr(string(n))),\"label\":$(jstr(first(last(p)))),\"values\":[" *
                  join((jstr(string(v)) for v in last(last(p))), ",") * "]}" for (n, p) in zip(names, params)), ",")
        print(io, "],\"outputs\":{")
        join(io, ("$(jstr(k)):$(jstr(o))" for (k, o) in zip(keys_, outs)), ",")
        print(io, "}}")
    end
    @info "explore: $id ($(length(keys_)) outputs)"
end

# ---- the example graph of Get Started: one call, the semiring decides the question ----
const I0 = [1, 1, 2, 3, 4, 4]; const J0 = [2, 3, 4, 2, 1, 5]; const V0 = [3.0, 8.0, 1.0, 4.0, 2.0, 7.0]
const A0 = sparse(I0, J0, V0, 5, 5)
const SEMI = Dict("MinPlus" => MinPlus(), "MaxMin" => MaxMin(), "MinMax" => MinMax())
write_cell("semirings";
    code = """
    using PATHPACK, SparseArrays
    using PATHPACK.CPU: MinPlus, MaxMin, MinMax, mstar

    A = sparse([1, 1, 2, 3, 4, 4], [2, 3, 4, 2, 1, 5],
               [3.0, 8.0, 1.0, 4.0, 2.0, 7.0], 5, 5)
    mstar({semiring}(), A)""",
    params = [:semiring => ("semiring", ["MinPlus", "MaxMin", "MinMax"])],
    run = s -> mstar(SEMI[s], A0),
    note = "MinPlus: shortest paths. MaxMin: the largest capacity of a single path. MinMax: the smallest largest weight on a path.")

# ---- one row from a factorization ----
const F0 = mlu(MinPlus(), A0)
write_cell("rows";
    code = """
    F = mlu(MinPlus(), A)            # factorize once
    B = fill(Inf, 1, 5); B[{source}] = 0.0
    rmul!(B, F)                      # the distances from that vertex""",
    params = [:source => ("source", collect(1:5))],
    run = s -> (B = fill(Inf, 1, 5); B[s] = 0.0; rmul!(B, F0)))

# ---- a city grid: size and the question asked of all its distances ----
function grid(k; seed = 1)
    Random.seed!(seed); id(i, j) = (j - 1) * k + i; I = Int[]; J = Int[]; V = Float64[]
    for j in 1:k, i in 1:k, (di, dj) in ((1, 0), (0, 1))
        (i + di <= k && j + dj <= k) || continue
        t = rand(1.0:5.0); a, b = id(i, j), id(i + di, j + dj)
        push!(I, a, b); push!(J, b, a); push!(V, t, t)
    end
    return sparse(I, J, V, k * k, k * k), id
end
const GRIDS = Dict(k => grid(k) for k in (10, 20, 40, 80))
const DGRID = Dict(k => mstar(MinPlus(), GRIDS[k][1]) for k in keys(GRIDS))
write_cell("grid";
    code = """
    k = {k}
    A, id = city_grid(k)                 # a k × k street grid, 1–5 minutes per street
    D = mstar(MinPlus(), A)              # every travel time
    {question}""",
    params = [:k => ("grid size", [10, 20, 40, 80]),
              :question => ("question", ["D[id(1, 1), id(k, k)]", "maximum(D)", "size(D)",
                                         "CartesianIndices((k, k))[argmin(vec(sum(D; dims = 2)))]"])],
    run = (k, q) -> begin
        D = DGRID[k]; id = GRIDS[k][2]
        q == "D[id(1, 1), id(k, k)]" ? D[id(1, 1), id(k, k)] :
        q == "maximum(D)" ? maximum(D) :
        q == "size(D)" ? size(D) : CartesianIndices((k, k))[argmin(vec(sum(D; dims = 2)))]
    end,
    note = "city_grid is the grid built at the top of this page. The last question finds the most central intersection.")
