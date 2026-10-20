# # Factorize Once, Query Many Times
#
# `mstar` computes all ``n^2`` entries of the star. Often only some of them are needed: the distances from a few
# depots, to a few hospitals, or from whichever node a user clicks next. PATHPACK.jl splits the work in two. `mlu`
# factorizes the star once, ``A^* = U^* L^*``; then each query is a pair of triangular solves with the factors, which
# costs far less than the factorization.

# ## A Directed Street Grid
#
# A 60 × 60 grid of streets, 3,600 intersections. Going one way along a street can take longer than the other (a hill,
# a busy lane), so the graph is directed: distances *from* a place differ from distances *to* it.

using PATHPACK, SparseArrays, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: MinPlus, mstar, mlu
using LinearAlgebra: lmul!, rmul!

Random.seed!(2)
k = 60; n = k * k
id(i, j) = (j - 1) * k + i
src = Int[]; dst = Int[]; minutes = Float64[]
for j in 1:k, i in 1:k, (a, b) in ((1, 0), (0, 1))
    (i + a <= k && j + b <= k) || continue
    u, v = id(i, j), id(i + a, j + b)
    push!(src, u, v); push!(dst, v, u); push!(minutes, rand(1:5), rand(1:5))
end
A = sparse(src, dst, minutes, n, n);

# ## The Factorization
#
# `mlu` takes the same arguments as `mstar` and returns the factorization instead of the star.

F = mlu(MinPlus(), A);

# For checking the queries below, here is the whole star too:

D = mstar(MinPlus(), A);

# ## Distances From Some Sources: Rows
#
# `rmul!(B, F)` overwrites `B` with ``B A^*``. Give `B` one row per source, holding the semiring's one (`0`) at the
# source and its zero (`Inf`) everywhere else, and each row becomes that source's distances to every node.

sources = [id(1, 1), id(30, 30), id(60, 1)]
B = fill(Inf, length(sources), n)
for (r, s) in enumerate(sources)
    B[r, s] = 0.0
end
rmul!(B, F)
B == D[sources, :]

#-

m(v) = reshape(Figs.unit(v), k, k)  #hide
figure(panel("row 1 of B", gridmap(m(B[1, :]); marks = [(1, 1)])), panel("row 2", gridmap(m(B[2, :]); marks = [(30, 30)])),  #hide
    panel("row 3", gridmap(m(B[3, :]); marks = [(60, 1)])); legend = legend("near", "far"),  #hide
    caption = "The three rows rmul! computed, one per circled source, from one factorization.")  #hide

# ## Distances To Some Targets: Columns
#
# `lmul!(F, X)` overwrites `X` with ``A^* X``. With one column per target, each column becomes the distances from
# every node to that target.

targets = [id(60, 60), id(15, 45)]
X = fill(Inf, n, length(targets))
for (c, t) in enumerate(targets)
    X[t, c] = 0.0
end
lmul!(F, X)
X == D[:, targets]

# Rows and columns differ here, since the graph is directed:

D[id(1, 1), id(60, 60)], D[id(60, 60), id(1, 1)]

# ## Many Queries
#
# The factorization is reused across calls. Timing on this small grid, on the threads this page was built with (after
# a first call of each, so that compilation is not counted):

function rows(S)
    B = fill(Inf, length(S), n)
    for (r, s) in enumerate(S)
        B[r, s] = 0.0
    end
    return rmul!(B, F)
end
rows(1:1); rows(1:100); mlu(MinPlus(), A); mstar(MinPlus(), A);

t_factor = @elapsed mlu(MinPlus(), A)
t_single = @elapsed for s in 1:100
    rows(s:s)
end
t_block = @elapsed rows(1:100)
t_star = @elapsed mstar(MinPlus(), A)
ms(t) = round(1000t; digits = 2)
(factorization = ms(t_factor), one_row = ms(t_single / 100),
    hundred_rows_one_call = ms(t_block), whole_star = ms(t_star))

#-

figure(panel("milliseconds, measured while this page was built", bars(["factorization", "one row", "100 rows, one call", "the whole star"], [ms(t_factor), ms(t_single / 100), ms(t_block), ms(t_star)]; fmt = x -> string(x, " ms"), hi = [false, true, false, false])))  #hide

# The times are in milliseconds, measured on the machine that built this page, so they vary from build to build. The
# factorization is paid once; each row afterwards costs a small fraction of it, and many rows in one call share the
# work of the solve.
#
# ## New Weights, Same Graph
#
# When the weights change but the streets do not (travel times at another hour), the ordering and the symbolic
# analysis of the graph can be kept. `ChordalSLU` holds them; `copyto!` loads new weights and `lu!` factorizes.

using PATHPACK.CPU: ChordalSLU
using LinearAlgebra: lu!

G = ChordalSLU(MinPlus(), A)            # the symbolic part, once
rush = copy(A); nonzeros(rush) .*= 2    # every street twice as slow
copyto!(G, rush); lu!(G)
B = fill(Inf, 1, n); B[1, id(1, 1)] = 0.0
rmul!(B, G) == 2 .* D[id(1, 1):id(1, 1), :]

# ## Threads
#
# Every one of these functions takes the keyword `nt`, the number of threads, and uses all of Julia's threads by
# default. For example, `mlu(MinPlus(), A; nt = 4)` or `rmul!(B, F; nt = 1)`. Start Julia with `--threads auto` to
# use every core.
