# # Isomap and Manifold Learning
#
# Isomap finds a flat map of data that lies on a curved surface. It links each point to its nearest neighbours,
# measures distances along that graph instead of straight through space, and then finds coordinates whose
# Euclidean distances match the graph distances. The middle step needs the shortest-path distance between every pair
# of points. It is usually the slowest step, and it is exactly what PATHPACK.jl computes.

#md # ```@raw html
#md # <div class="pp-t-hero">
#md #   <div class="pp-t-clip"><SwissUnroll /></div>
#md #   <div class="pp-t-facts">
#md #     <div class="pp-t-fact"><b>105×</b><span>faster than scikit-learn's Isomap shortest-path step, on a 10,000-point swiss roll</span></div>
#md #     <div class="pp-t-fact"><b>0.21 s</b><span>for all 100 million distances on 16 CPU threads; NetworKit takes 1.27 s on the same threads</span></div>
#md #     <div class="pp-t-fact"><b>22.1 ms</b><span>on one NVIDIA B200</span></div>
#md #   </div>
#md # </div>
#md # <p class="pp-caption">10,000 points on a swiss roll, colored by geodesic distance from one end, moving to their Isomap coordinates: PATHPACK.jl computes every geodesic of the nearest-neighbour graph, and classical MDS lays them flat.</p>
#md # ```

# ## A Swiss Roll
#
# The swiss roll is a rectangle of paper rolled into a spiral. Sample 2,000 points on it, as scikit-learn's
# `make_swiss_roll` does: an angle ``t`` between ``1.5\pi`` and ``4.5\pi`` sets the position along the spiral, and a
# height ``h`` between 0 and 21 the position across it.

using PATHPACK, SparseArrays, LinearAlgebra, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: MinPlus, mstar

Random.seed!(4)
n = 2000
t = 1.5π .+ 3π .* rand(n)
h = 21 .* rand(n)
X = [t .* cos.(t) h t .* sin.(t)];    # n points in 3D

# ## The Nearest-Neighbour Graph
#
# Link each point to its 10 nearest neighbours, with the Euclidean distance as the edge length, in both directions.
# Brute force is fine at this size: `E` holds the straight-line distances, and each column's smallest entry is the
# point itself, which is skipped.

E = [norm(X[i, :] - X[j, :]) for i in 1:n, j in 1:n]
k = 10
src = Int[]; dst = Int[]; len = Float64[]
for i in 1:n
    for j in partialsortperm(view(E, :, i), 2:(k + 1))
        push!(src, i, j); push!(dst, j, i); push!(len, E[i, j], E[i, j])
    end
end
A = sparse(src, dst, len, n, n, min)
nnz(A)

# ## Every Geodesic Distance
#
# One call gives the shortest-path distance between all ``n^2`` pairs of points:

G = mstar(MinPlus(), A)
all(isfinite, G)    # the graph is connected

# The roll is a rolled-up rectangle, so the true distance along the sheet is known. Unroll the spiral by its arc
# length ``s(t)``, and the distance between two points is the plane distance between ``(s, h)`` coordinates.

arc(t) = (t * sqrt(1 + t^2) + asinh(t)) / 2
s = arc.(t)
truth = [hypot(s[i] - s[j], h[i] - h[j]) for i in 1:n, j in 1:n]
off = [i != j for i in 1:n, j in 1:n]
graph_ratio = sum(G[off] ./ truth[off]) / count(off)
straight_ratio = sum(E[off] ./ truth[off]) / count(off)
graph_ratio, straight_ratio

# On average the graph distances are within a few percent of the distances along the sheet. Straight-line
# distances cut across the layers of the roll and come out about a third too short.

# ## Unrolling: Classical MDS
#
# The last step of Isomap looks for points in the plane whose distances match ``G``. Classical multidimensional scaling
# does it with one eigendecomposition: double-center the squared distances and keep the two leading eigenvectors.
# scikit-learn's Isomap does the same, as kernel PCA on ``-\tfrac12 G^2``.

C = I - fill(1 / n, n, n)
K = -0.5 .* (C * (G .^ 2) * C)
λ, V = eigen(Symmetric(K), (n - 1):n)
Y = V[:, [2, 1]] .* sqrt.(λ[[2, 1]])';    # n × 2 coordinates

# The two coordinates should recover the two directions of the sheet: position along the roll and height. The
# correlation tells (its sign is arbitrary, since eigenvectors are):

centered(a) = a .- sum(a) / length(a)
corr(a, b) = dot(centered(a), centered(b)) / (norm(centered(a)) * norm(centered(b)))
abs(corr(Y[:, 1], s)), abs(corr(Y[:, 2], h))

# The same scaling with straight-line distances does not unroll anything:

Kₑ = -0.5 .* (C * (E .^ 2) * C)
λₑ, Vₑ = eigen(Symmetric(Kₑ), (n - 1):n)
abs(corr(Vₑ[:, 2], s)), abs(corr(Vₑ[:, 1], h))

#-

figure(panel("Isomap", scatter(Y[:, 1], Y[:, 2], Figs.unit(s)); sub = "MDS of the graph distances G"),  #hide
    panel("straight-line MDS", scatter(Vₑ[:, 2] .* sqrt(λₑ[2]), Vₑ[:, 1] .* sqrt(λₑ[1]), Figs.unit(s)); sub = "MDS of the distances E through space");  #hide
    legend = legend("one end of the sheet", "the other"),  #hide
    caption = "The 2,000 points, colored by their true position along the sheet. With graph distances the roll lies flat and the colors run in order; with straight-line distances the layers fold onto each other.")  #hide

# ## Larger Data, and GPUs
#
# The distance matrix has ``n^2`` entries: 100 million for 10,000 points, 800 MB in `Float64`. With the CUDA extension
# it can be computed on the GPU and stay there for the eigendecomposition. The kNN graph is undirected, as the GPU
# solver needs.
#
# ```julia
# using CUDA
# using PATHPACK.GPU: apsp_gpu
#
# G = apsp_gpu(Float32.(A))    # a CuMatrix{Float32}
# ```
#
# ## Measured
#
# The shortest-path step of Isomap on a 10,000-point swiss roll, 10 nearest neighbours: all 100 million distances.
# CPU: one 32-core AMD EPYC 7702 node. GPU: one NVIDIA B200. All codes return the same distances.
#
# | Code | Hardware | Time | vs PATHPACK.jl |
# |:---|:---|---:|---:|
# | **PATHPACK.jl** | 16 CPU threads | **0.206 s** | |
# | NetworKit | 16 CPU threads | 1.27 s | 6.2× slower |
# | SciPy | 16 processes | 1.68 s | 8.1× slower |
# | SciPy Dijkstra | 1 core | 21.8 s | 105× slower |
# | scikit-learn Isomap, path step | 1 core | 21.8 s | 105× slower |
# | igraph | 1 core | 31.6 s | 153× slower |
# | **PATHPACK.jl** | one B200 | **22.1 ms** | |
# | ROME | one B200 | 25.0 ms | 1.1× slower |
# | cuGraph | one B200 | 285.7 ms | 12.9× slower |
#
# The GPU rows are compared with PATHPACK.jl on the same GPU.
