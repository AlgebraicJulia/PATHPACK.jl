# # One Graph, Many Questions
#
# The star of a matrix depends on the semiring it is computed in. On one graph with one set of weights, a different
# semiring answers a different question, and PATHPACK.jl computes each one with the same call and the same solver.
#
# Below, one landscape: north-central Florida as a grid of 67,411 land-cover cells, each step weighted by how hard
# that ground is for a forest animal to cross. Three semirings give three different maps, each the row of one forest
# cell in the all-pairs answer.

#md # ```@raw html
#md # <LandscapeLive />
#md # ```

# ## A Small Landscape
#
# A 30 × 30 grid stands in for the map: each cell has a resistance between 1 (open forest) and 20 (a road or a town),
# and a step between neighbours costs the mean of the two resistances.

using PATHPACK, SparseArrays, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: MinPlus, MinMax, MaxMin, PlusProd, mstar

Random.seed!(2)
k = 30; n = k * k
id(i, j) = (j - 1) * k + i
r = rand([1.0, 1.0, 1.0, 2.0, 5.0, 20.0], k, k)
I = Int[]; J = Int[]; C = Float64[]
for j in 1:k, i in 1:k
    for (di, dj) in ((1, 0), (0, 1))
        (i + di <= k && j + dj <= k) || continue
        c = (r[i, j] + r[i + di, j + dj]) / 2
        a, b = id(i, j), id(i + di, j + dj)
        push!(I, a, b); push!(J, b, a); push!(C, c, c)
    end
end
A = sparse(I, J, C, n, n)
s, t = id(1, 1), id(k, k);    # from one corner to the other

# ## The Cheapest Route: MinPlus
#
# With ⊕ = min and ⊗ = +, a path's weight is its total cost and the star keeps the cheapest: the least-cost distance
# of landscape ecology.

D = mstar(MinPlus(), A)
D[s, t]

# ## The Worst Barrier: MinMax
#
# With ⊕ = min and ⊗ = max, a path's weight is its single worst step, and the star keeps the path whose worst step is
# mildest. It tells whether two places are connected without ever crossing a step above some cost: the barriers
# that split a landscape.

B = mstar(MinMax(), A)
B[s, t]

# ## The Widest Route: MaxMin
#
# Flip both operations and the weights become capacities: with ⊕ = max and ⊗ = min the star is the most capacity a
# single route can carry, as for the bandwidth of a network or the heaviest truck a road network allows. Here, with
# ease of crossing (one over the cost) as the capacity:

E = copy(A); nonzeros(E) .= 1 ./ nonzeros(E)
W = mstar(MaxMin(), E)
W[s, t] == 1 / B[s, t]    # the widest route in ease is the mildest route in cost

# ## Where a Wandering Animal Goes: PlusProd
#
# With ⊕ = + and ⊗ = ×, the star sums over every walk instead of picking one. Take the random walk that steps to a
# neighbour with probability proportional to how easy it is to cross, and survives each step with probability β:
# the star `(I - βP)⁻¹` holds the expected number of visits to every cell from every start.

β = 0.999
P = spdiagm(β ./ vec(sum(E; dims = 2))) * E
V = mstar(PlusProd(), P)
V[s, t], V[s, s]

#-

m(v) = reshape(v, k, k)  #hide
figure(panel("the landscape", gridmap(m(Figs.unit(vec(r); log = true)); marks = [(1, 1)]); sub = "resistance, low to high"),  #hide
    panel("MinPlus", gridmap(m(Figs.unit(D[s, :])); marks = [(1, 1)]); sub = "cheapest route"),  #hide
    panel("MinMax", gridmap(m(Figs.unit(B[s, :])); marks = [(1, 1)]); sub = "worst step on the mildest route"),  #hide
    panel("PlusProd", gridmap(m(1 .- Figs.unit(V[s, :]; log = true)); marks = [(1, 1)]); sub = "expected visits, log scale");  #hide
    cols = 2, legend = legend("near, mild, often", "far, harsh, rarely"),  #hide
    caption = "Row s of each star, from the circled corner cell: one graph, one set of weights, three questions.")  #hide

# The same sum over walks, with other weights, gives exact personalized PageRank for graph neural networks and walk
# counts in a network.
#
# ## Every Semiring
#
# | Semiring | ⊕, ⊗ | The star answers |
# |---|---|---|
# | `MinPlus` | min, + | shortest paths |
# | `MaxPlus` | max, + | longest paths (on graphs without positive cycles) |
# | `MaxMin` | max, min | widest paths |
# | `MinMax` | min, max | bottleneck paths |
# | `MaxProd` | max, × | most reliable paths, with probabilities as weights |
# | `PlusProd` | +, × | sums over all walks |
# | `OrAnd` | ∪, ∩ on sets of bits | reachability |
#
# All of them are exported by `PATHPACK.CPU`; the GPU solver takes a semiring through `apsp_gpu(A; semiring = ...)`.
