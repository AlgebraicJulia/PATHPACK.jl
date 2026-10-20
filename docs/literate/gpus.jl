# # GPUs, One or Many
#
# With CUDA.jl loaded, PATHPACK.jl computes the star on NVIDIA GPUs. The CPU threads factorize the many small parts
# of the graph while the GPU factorizes the large ones; then the GPU computes every row of the result. With several
# GPUs, each one computes a block of rows. This page shows the calls. The code here is not run when the site is built,
# since the build machine has no GPU.

#md # ```@raw html
#md # <div class="pp-stats pp-t-stats">
#md #   <div class="pp-stat"><b>0.139 s</b><span>all 15.9 billion distances of the US road network (126,146 intersections), one NVIDIA B200</span></div>
#md #   <div class="pp-stat"><b>0.389 s</b><span>all 189.8 billion distances of Colorado's roads (435,666 intersections), eight B200s</span></div>
#md #   <div class="pp-stat"><b>3.1×</b><span>faster than ROME, geometric mean over ROME's own 16 test graphs; faster on all 16</span></div>
#md # </div>
#md # ```

# ## Installation
#
# Add CUDA.jl next to PATHPACK.jl:
#
# ```
# pkg> add https://github.com/AlgebraicJulia/PATHPACK.jl
# pkg> add CUDA
# ```
#
# Loading both loads the GPU extension, and `PATHPACK.GPU` then has methods:

using PATHPACK, CUDA
using PATHPACK.CPU: MinPlus, MaxMin, PlusProd, mstar
using PATHPACK.GPU: apsp_gpu, apsp_gpu!, apsp_plan

# Start Julia with several threads (`julia --threads auto`): the CPU part of the factorization runs on them.

# ## A First Call
#
# `apsp_gpu(A)` takes the same sparse matrix as `mstar`, with `A[i, j]` the weight of the arc `i → j`, and returns
# the star in the min-plus semiring: all shortest-path distances.

using SparseArrays, Random

Random.seed!(1)
n = 5000
I = rand(1:n, 2n); J = rand(1:n, 2n); keep = I .!= J
W = Float32.(rand(1:100, count(keep)))     # integers: exact in Float32
A = sparse([I[keep]; J[keep]], [J[keep]; I[keep]], [W; W], n, n, min)

D = apsp_gpu(A)                     # a CuMatrix{Float32}, on the GPU
Array(D) == mstar(MinPlus(), A)     # true: the CPU star, bit for bit

# ## Where the Result Goes
#
# The result has ``n^2`` entries, so where it lands matters as much as how fast it is computed.
#
# | Call | Result |
# |:---|:---|
# | `apsp_gpu(A)` | a `CuMatrix` on the GPU, the default (`output = :device`) |
# | `apsp_gpu(A; output = :host)` | a `Matrix` in CPU memory |
# | `apsp_gpu!(D, A)` | into a `CuMatrix` you allocated, on the current device |
# | `apsp_gpu!(H, A)` | into a `Matrix` you allocated |
#
# Keep the result on the GPU when the next step runs there too: an eigendecomposition for Isomap, a reduction for
# closeness, or a transfer to PyTorch. Copying ``n^2`` numbers to the host takes time of its own, and so does touching
# the pages of a fresh host matrix. When you call the solver many times, allocate the host matrix once and reuse it
# with `apsp_gpu!(H, A)`.

# ## Only Some Rows
#
# `apsp_gpu(A, sources)` returns only the rows of the given sources, a `length(sources) × n` matrix. Use it when only
# some sources matter, or when the whole star does not fit: call it for blocks of sources and process each block in
# turn.

for block in Iterators.partition(1:n, 10_000)
    R = apsp_gpu(A, collect(block))    # R[t, j]: from block[t] to j
    # ... reduce R, or write it to disk, before the next block
end

# ## Several GPUs
#
# Pass several devices and the rows are split over them. Each GPU holds only its own block, so the star can be larger
# than one GPU's memory.

gpus = collect(CUDA.devices())

H = apsp_gpu(A; devices = gpus, output = :host)    # one Matrix on the CPU
blocks = apsp_gpu(A; devices = gpus)                # one block per GPU

# With `output = :device` the result is a vector of pairs `(sources_g, D_g)`, and block `g` stays on GPU `g`:
# `D_g[t, j]` is the distance from `sources_g[t]` to `j`. The work is balanced between the GPUs: over our multi-GPU
# runs, the slowest GPU took 2.2% longer than the mean in the median case, and 13.3% longer at most.

# ## New Weights, Same Graph
#
# Many applications solve one graph again and again with different weights: travel times at different hours,
# resistances under different scenarios, diffusion with different parameters. A plan does once all the work that
# depends only on the graph's structure (the ordering, the symbolic factorization, the schedules) and keeps it on the
# GPU.

P = apsp_plan(A)               # once per graph
for hour in 1:24
    Aₕ = travel_times(hour)    # the pattern of A, new weights
    D = apsp_gpu(P, Aₕ)        # only the numeric work is repeated
    # ...
end

# A matrix with a different pattern is an `ArgumentError`. With a host matrix, `apsp_gpu!(H, P, Aₕ)` writes into it.

# ## Semirings, Types, Graphs
#
# Pass another semiring with the `semiring` keyword, as in `apsp_gpu(A; semiring = MaxMin())`.
#
# | Semiring | Problem | Element types on the GPU |
# |:---|:---|:---|
# | `MinPlus` (default) | shortest paths | `Float32`, `Float64` |
# | `MaxPlus` | longest paths | `Float32`, `Float64` |
# | `MaxMin` | widest paths | `Float32`, `Float64`, `Int32` |
# | `MinMax` | bottleneck paths | `Float32`, `Float64` |
# | `PlusProd` | sums over walks | `Float64` |
#
# Integer weights are not accepted for min-plus, since the integer infinity overflows. `Float32` represents integers
# exactly only up to ``2^{24}``; when the distances could exceed that, the solver warns once, and `Float64` weights
# give exact integer distances.
#
# The GPU solver handles undirected graphs, and directed graphs whose strongly connected components do not reach one
# another (for example, a strongly connected graph). For any other directed graph it throws an `ArgumentError`; use
# the CPU solver, `mstar`, which handles every graph.

# ## Memory
#
# The star of a graph with ``n`` vertices has ``n^2`` entries: ``4n^2`` bytes in `Float32` and ``8n^2`` in `Float64`.
#
# | Vertices | Entries | `Float32` | `Float64` |
# |---:|---:|---:|---:|
# | 10,000 | 100 million | 0.4 GB | 0.8 GB |
# | 50,000 | 2.5 billion | 10 GB | 20 GB |
# | 126,146 | 15.9 billion | 64 GB | 127 GB |
# | 435,666 | 189.8 billion | 759 GB | 1.5 TB |
#
# A result that does not fit in GPU memory is an error that says so before any work starts. Then split the rows over
# several GPUs, or compute blocks of rows with `apsp_gpu(A, sources)`.

# ## Measured
#
# Wall-clock time from the graph in memory to the full answer in memory, on NVIDIA B200 GPUs, all pairs. Every code
# returns the same distances.
#
# **One GPU**, against other GPU codes on the same GPU:
#
# | Graph | Vertices | PATHPACK.jl | ROME | cuGraph | Fast-APSP |
# |:---|---:|---:|---:|---:|---:|
# | US road network | 126,146 | **0.139 s** | 0.750 s | 18.05 s | 105.5 s |
# | Stanford bunny mesh | 34,834 | **55.6 ms** | 0.109 s | 1.70 s | |
# | 164k-vertex cortex | 164k | **0.90 s** | 9.46 s | | |
# | land-cover grid | 67,411 | **0.102 s** | 0.353 s | 10.13 s | |
# | swiss roll kNN graph | 10,000 | **22.1 ms** | 25.0 ms | 285.7 ms | |
#
# Over ROME's own 16 test graphs PATHPACK.jl is faster on all 16, by 3.1× in geometric mean. cuGraph is 6 to 1,600
# times slower across our graphs.
#
# **Eight GPUs**, every pair of vertices:
#
# | Graph | Vertices | Distances | Time |
# |:---|---:|---:|---:|
# | France roads | 297,144 | 88.3 billion | 0.343 s |
# | Brazil roads | 323,474 | 104.6 billion | 0.392 s |
# | Japan roads | 353,713 | 125.1 billion | 0.464 s |
# | Colorado roads | 435,666 | 189.8 billion | 0.389 s |
# | Stanford dragon mesh | 436,846 | 190.8 billion | 0.699 s |
#
# More results, including other semirings and CPU comparisons, are on the [Benchmarks](../benchmarks.md) page.
