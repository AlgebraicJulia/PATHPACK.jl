# # Road Networks
#
# A routing table, an accessibility map or a logistics model needs the travel distance between every pair of places.
# On a road network that is the all-pairs shortest-path problem: with ``n`` intersections, ``n^2`` distances.
# PATHPACK.jl computes all of them at once, as the star ``A^*`` of the network's weighted adjacency matrix in the
# min-plus semiring.

#md # ```@raw html
#md # <div class="pp-t-wide"><AppVisual name="france" ring label="France's road network, colored by road distance from Paris" /></div>
#md # <p class="pp-caption">France's roads, 297,144 intersections, colored by road distance from Paris. PATHPACK.jl computes all 88 billion pairs on eight NVIDIA B200s in 0.34 s.</p>
#md # ```

# ## A Small Road Grid
#
# A city grid stands in for a real network here: intersections on a 40 × 40 lattice, each street with a travel time
# between 1 and 5 minutes, in both directions.

using PATHPACK, SparseArrays, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: MinPlus, mstar, mlu
using LinearAlgebra: rmul!

Random.seed!(1)
k = 40; n = k * k
id(i, j) = (j - 1) * k + i
I = Int[]; J = Int[]; V = Float64[]
for j in 1:k, i in 1:k
    for (di, dj) in ((1, 0), (0, 1))
        (i + di <= k && j + dj <= k) || continue
        t = rand(1.0:5.0)
        a, b = id(i, j), id(i + di, j + dj)
        push!(I, a, b); push!(J, b, a); push!(V, t, t)
    end
end
A = sparse(I, J, V, n, n);

#md # ```@raw html
#md # <CodeExplorer id="grid" />
#md # ```

# ## All Pairs
#
# `mstar(MinPlus(), A)` returns the dense matrix of travel times: `D[i, j]` is the length of a shortest route from `i`
# to `j`. It runs on all the threads Julia was started with.

D = mstar(MinPlus(), A)
D[id(1, 1), id(k, k)]    # corner to corner, in minutes

# Every row of `D` is a complete map of travel times from one intersection, so questions about all places at once are
# one line each. The closeness of each intersection is its mean travel time to every other one:

closeness = vec(sum(D; dims = 2)) ./ (n - 1)
CartesianIndices((k, k))[argmin(closeness)]    # the most central intersection, near the middle

#-

c = CartesianIndices((k, k))[argmin(closeness)]  #hide
figure(panel("D[id(1, 1), :]", gridmap(reshape(Figs.unit(D[id(1, 1), :]), k, k); marks = [(1, 1)]); sub = "minutes from the corner"),  #hide
    panel("closeness", gridmap(reshape(Figs.unit(closeness), k, k); marks = [(c[1], c[2])]); sub = "mean minutes to everywhere");  #hide
    legend = legend("near", "far"), caption = "Both maps come from the one all-pairs matrix D: a row of it, and the mean of every row. The circled intersection is the most central.")  #hide

# ## A Few Sources, Many Times
#
# When only some rows are needed, or the same network is queried again and again, factorize once and answer each query
# with two triangular solves. `rmul!(B, F)` overwrites the rows of `B` with `B A*`; a row holding the semiring's one
# (`0`) at a source and its zero (`Inf`) elsewhere becomes that source's distances.

F = mlu(MinPlus(), A)
B = fill(Inf, 1, n); B[1, id(1, 1)] = 0.0
rmul!(B, F)
vec(B) == D[id(1, 1), :]

# ## Real Networks, on One GPU or Many
#
# With the CUDA.jl extension the same problem runs on NVIDIA GPUs. The CPU factorizes the many small parts of the
# network while the GPU takes the large ones, then the GPU computes every row of the result.
#
# ```julia
# using CUDA
# using PATHPACK.GPU: apsp_gpu
#
# D = apsp_gpu(A)                                          # the star on the GPU, as a CuMatrix
# D = apsp_gpu(A; devices = collect(CUDA.devices()))       # its rows split over every GPU in the machine
# ```
#
# Measured on NVIDIA B200 GPUs, every pair of intersections:
#
# | Network | Intersections | Distances | Time |
# |---|---|---|---|
# | United States, major roads | 126,146 | 15.9 billion | 0.14 s on one GPU |
# | France | 297,144 | 88.3 billion | 0.34 s on 8 GPUs |
# | Brazil | 323,474 | 104.6 billion | 0.39 s on 8 GPUs |
# | Japan | 353,713 | 125.1 billion | 0.46 s on 8 GPUs |
# | Colorado | 435,666 | 189.8 billion | 0.39 s on 8 GPUs |
#
# On the US network, on one B200 each, ROME takes 0.75 s, cuGraph 18.1 s and Fast-APSP 105.5 s.
