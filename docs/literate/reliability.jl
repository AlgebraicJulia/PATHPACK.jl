# # Most Reliable Routes
#
# Every link of a communication network works with some probability. If links fail independently, a route works with
# the product of its links' probabilities, and the most reliable route between two nodes is the one with the largest
# product. In the max-times semiring (``\oplus = \max``, ``\otimes = \times``) the star of the matrix of link
# probabilities holds the reliability of the most reliable route between every pair of nodes.

# ## A Network of Unreliable Links
#
# 300 nodes scattered in the unit square, each linked to the nodes within distance 0.12. Longer links fail more often:
# a link works with probability between 0.90 and 0.999, lower the longer it is.

using PATHPACK, SparseArrays, LinearAlgebra, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: MaxProd, MinPlus, mstar

Random.seed!(7)
n = 300
xy = rand(n, 2)
src = Int[]; dst = Int[]; p = Float64[]
for i in 1:n, j in (i + 1):n
    d = norm(xy[i, :] - xy[j, :])
    if d < 0.12
        q = 0.999 - 0.099 * d / 0.12    # 0.999 for a short link, 0.90 at the cutoff
        push!(src, i, j); push!(dst, j, i); push!(p, q, q)
    end
end
P = sparse(src, dst, p, n, n)
nnz(P) ÷ 2    # links

# ## Every Most Reliable Route
#
# `R[i, j]` is the probability that the most reliable route from `i` to `j` works: `1` on the diagonal (the
# semiring's one) and `0` when no route exists (its zero).

R = mstar(MaxProd(), P)
extrema(R)

# ## The Same Answer as Shortest Paths
#
# Taking logarithms turns products into sums: maximizing ``\prod p_e`` is minimizing ``\sum -\log p_e``. So the most
# reliable routes are the shortest paths for the weights ``-\log p``, and ``R = e^{-D}``.

W = copy(P); nonzeros(W) .= -log.(nonzeros(W))
D = mstar(MinPlus(), W)
maximum(abs.(R .- exp.(-D)))

# Both stars also pick the same route. Walk from `s` to `t`, always taking a link that keeps the best value to `t`:

function route(best, A, s, t; mul, tol = 1e-12)
    path = [s]
    while path[end] != t
        u = path[end]
        cols = rowvals(A)
        v = first(cols[q] for q in nzrange(A, u)
            if isapprox(mul(nonzeros(A)[q], best[cols[q], t]), best[u, t]; rtol = tol))
        push!(path, v)
    end
    return path
end

s, t = argmin(xy[:, 1] + xy[:, 2]), argmax(xy[:, 1] + xy[:, 2])    # opposite corners
reliable = route(R, sparse(P'), s, t; mul = *)
shortest = route(D, sparse(W'), s, t; mul = +)
reliable == shortest, length(reliable) - 1, R[s, t]

#-

figure(panel("R[s, :]", network(xy, src, dst; val = 1 .- Figs.unit(R[s, :]; log = true), path = reliable, mark = [s, t]); sub = "most reliable route from the circled corner"),  #hide
    legend = legend("most reliable", "least"), caption = "Every node colored by the reliability of its best route from s, one row of the max-times star; the line is the most reliable route from s to t.")  #hide

# (The transpose makes `nzrange` run over the links out of `u`; the network is undirected, so it is the same matrix.)
#
# ## Why Use MaxProd Directly
#
# The two give the same routes, so either works. Working with probabilities avoids the logarithm and its inverse, and
# the result is in the units people ask about. It is also the natural form when the star feeds other products of
# probabilities, for example to combine the reliability of the best route to a relay with that from the relay onward.
#
# On large networks, the same call runs on the GPU for undirected graphs, with `Float32` or `Float64` probabilities:
#
# ```julia
# using CUDA
# using PATHPACK.GPU: apsp_gpu
#
# R = apsp_gpu(P; semiring = MaxProd())
# ```
