# # Bandwidth and Widest Paths
#
# A route through a network is only as good as its weakest link. The bandwidth of a route is the smallest capacity
# along it, and the widest path between two nodes is the route with the largest bandwidth. In the max-min semiring
# (``\oplus = \max``, ``\otimes = \min``) the star of the capacity matrix holds the widest-path bandwidth between every
# pair of nodes. Its mirror, the min-max semiring, gives bottleneck paths: the route whose worst step is mildest.

# ## A Network With Link Capacities
#
# 250 routers in the unit square, each linked to the routers within distance 0.13. Each link has a capacity of 1,
# 10, 40 or 100 Gbit/s; fast links are rarer.

using PATHPACK, SparseArrays, LinearAlgebra, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: MaxMin, MinMax, mstar

Random.seed!(11)
n = 250
xy = rand(n, 2)
src = Int[]; dst = Int[]; cap = Float64[]
for i in 1:n, j in (i + 1):n
    if norm(xy[i, :] - xy[j, :]) < 0.13
        c = rand([1.0, 1.0, 1.0, 10.0, 10.0, 40.0, 100.0])
        push!(src, i, j); push!(dst, j, i); push!(cap, c, c)
    end
end
C = sparse(src, dst, cap, n, n)
nnz(C) ÷ 2    # links

# ## Every Widest Path

W = mstar(MaxMin(), C)
W[1:6, 1:6]

# The diagonal is `Inf`, the semiring's one: a node reaches itself through no link at all. An entry is `-Inf`, the
# zero, when no route exists. Here is how the bandwidths between all pairs of distinct routers are spread:

offdiag = W[.!I(n)]
[c => count(==(c), offdiag) for c in (1.0, 10.0, 40.0, 100.0)]

#-

r0 = argmax([length(unique(W[i, :])) * 1000 - count(==(1.0), W[i, :]) for i in 1:n])  #hide
figure(panel("W[$r0, :]", network(xy, src, dst; val = 1 .- Figs.unit(W[r0, :]; log = true), lw = map(c -> c == 1 ? 0.5 : c == 10 ? 1.1 : c == 40 ? 2 : 3.2, cap), mark = [r0]); sub = "bandwidth from the circled router"),  #hide
    legend = legend("100 Gbit/s", "1 Gbit/s"), caption = "Link width shows capacity. Each router is colored by the widest-path bandwidth from the circled router: a row of the max-min star.")  #hide

# ## A Check by Thresholds
#
# The widest path has a simple characterization: two routers are joined by a route of bandwidth at least ``c`` exactly
# when they are connected using only the links of capacity at least ``c``. So for each capacity level, the connected
# components of the network's fast links must match the pairs with `W .>= c`.

function components(n, src, dst, keep)
    root = collect(1:n)
    find(i) = root[i] == i ? i : (root[i] = find(root[i]))
    for (u, v, k) in zip(src, dst, keep)
        k && (root[find(u)] = find(v))
    end
    return [find(i) for i in 1:n]
end

all(1:4) do level
    c = (1.0, 10.0, 40.0, 100.0)[level]
    comp = components(n, src, dst, cap .>= c)
    (W .>= c) == (comp .== comp')
end

# The routes themselves follow a maximum spanning tree of the network: the widest path between two nodes is their
# path in such a tree. The star gives the bandwidth of every pair without building the tree.

# ## Bottleneck Paths: MinMax
#
# Swap the operations and the question becomes the mildest worst step. Take a terrain of 40 × 40 cells with random,
# smoothed heights. A step between neighbouring cells reaches the higher of the two heights, and the min-max star
# holds, for every pair of cells, the lowest height a walker must climb to on the way. It is also the water level at
# which a flood first connects the two cells.

k = 40; m = k * k
cell(i, j) = (j - 1) * k + i
blur(h) = [sum(h[clamp(i + a, 1, k), clamp(j + b, 1, k)] for a in -1:1, b in -1:1) / 9
    for i in 1:k, j in 1:k]
h = blur(blur(blur(blur(randn(k, k)))))    # noise smoothed into hills and valleys
h = h ./ maximum(abs.(h))
src = Int[]; dst = Int[]; top = Float64[]
for j in 1:k, i in 1:k, (a, b) in ((1, 0), (0, 1))
    (i + a <= k && j + b <= k) || continue
    u, v = cell(i, j), cell(i + a, j + b)
    t = max(h[i, j], h[i + a, j + b])
    push!(src, u, v); push!(dst, v, u); push!(top, t, t)
end
T = sparse(src, dst, top, m, m)
B = mstar(MinMax(), T)
B[cell(1, 1), cell(k, k)]    # the lowest pass between two corners

#-

figure(panel("the terrain", gridmap(Figs.unit(h); marks = [(1, 1), (k, k)]); sub = "height, low to high"),  #hide
    panel("B[cell(1, 1), :]", gridmap(reshape(Figs.unit(B[cell(1, 1), :]), k, k); marks = [(1, 1), (k, k)]); sub = "the highest point on the lowest route");  #hide
    legend = legend("low", "high"), caption = "Each cell is colored by the lowest height a route from the top-left corner must climb to reach it: a row of the min-max star. Valleys connected by low passes share a color.")  #hide

# The same threshold check applies: two cells are connected by a walk that never rises above ``\ell`` exactly when
# they are connected among the steps of height at most ``\ell``.

all(rand(unique(top), 20)) do ℓ
    comp = components(m, src, dst, top .<= ℓ)
    offd = .!I(m)
    ((B .<= ℓ) .& offd) == ((comp .== comp') .& offd)
end

# (The diagonal is left out: it holds `-Inf`, the semiring's one, since a cell reaches itself without a step.)
#
# ## Integers and GPUs
#
# Capacities are often integers. The max-min semiring needs no arithmetic beyond comparisons, so the GPU solver also
# takes `Int32` weights for it:
#
# ```julia
# using CUDA
# using PATHPACK.GPU: apsp_gpu
#
# W = apsp_gpu(Int32.(C); semiring = MaxMin())
# ```
