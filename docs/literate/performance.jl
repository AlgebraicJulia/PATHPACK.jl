# # Performance Guide
#
# What makes PATHPACK.jl fast or slow on a given problem, and the settings that change it: threads, the vertex
# ordering, the element type, how much of the star to compute, and whether to use a GPU. The timings on this page are
# measured while the page is built, on a small grid, so they show trends rather than the speed of your machine.

# ## The Test Graph
#
# A 70 × 70 grid, 4,900 vertices, with integer weights from 1 to 9 in both directions: the shape of a road or a mesh.

using PATHPACK, SparseArrays, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: MinPlus, mstar, mlu
using CliqueTrees: AMF, MMD

Random.seed!(3)
k = 70; n = k * k
id(i, j) = (j - 1) * k + i
src = Int[]; dst = Int[]; w = Float64[]
for j in 1:k, i in 1:k, (a, b) in ((1, 0), (0, 1))
    (i + a <= k && j + b <= k) || continue
    u, v = id(i, j), id(i + a, j + b); t = rand(1:9)
    push!(src, u, v); push!(dst, v, u); push!(w, t, t)
end
A = sparse(src, dst, w, n, n)
D = mstar(MinPlus(), A);    # a first call, so that compilation is not timed below

# ## Measure the Second Call
#
# Julia compiles a method the first time it runs with new argument types. For a small graph the first call of `mstar`
# is mostly compilation. Time the second call, or use BenchmarkTools.jl.

# ## Threads
#
# The CPU solver runs on Julia's threads. Start Julia with `julia --threads auto` (or `-t 16`), and check with
# `Threads.nthreads()`. Every function takes the keyword `nt` to use fewer.

Threads.nthreads()

#-

t1 = @elapsed mstar(MinPlus(), A; nt = 1)
tn = @elapsed mstar(MinPlus(), A)
(one_thread_s = round(t1; digits = 3), all_threads_s = round(tn; digits = 3))

# A small graph leaves little work to share between threads. Measured on one 32-core AMD EPYC 7702 node, all pairs of
# the Stanford bunny mesh (34,834 vertices) take 12.98 s on 1 thread and 1.61 s on 16.

# ## The Ordering
#
# The factorization eliminates the vertices in some order, and the order decides how much fill, and so how much work,
# the factorization creates. PATHPACK.jl picks a fill-reducing order with CliqueTrees.jl. The keyword `alg` of `mlu`
# and `mstar` takes another algorithm from CliqueTrees.jl, or a permutation of the vertices.

orders = ["AMF()" => AMF(), "MMD()" => MMD(), "natural order" => collect(1:n), "random order" => randperm(n)]
times = map(orders) do (name, alg)
    mlu(MinPlus(), A; alg)    # once to compile
    name => round(@elapsed(mlu(MinPlus(), A; alg)); digits = 3)
end

#-

figure(panel("seconds to factorize, by vertex order", bars(first.(times), last.(times); fmt = x -> string(x, " s"), hi = first.(times) .== "AMF()")),  #hide
    caption = "Measured while this page was built, on the 70 × 70 grid above. AMF() is the default.")  #hide

# The answer does not depend on the order:

all(mstar(MinPlus(), A; alg) == D for (name, alg) in orders)

# A random order destroys the locality of the grid, and the factorization fills in and slows down. Among good orders
# the ranking depends on the graph, so for a graph you solve often, try a few. Nested dissection (`METIS()`) needs
# Metis.jl loaded:
#
# ```julia
# import Metis
# using CliqueTrees: METIS
#
# F = mlu(MinPlus(), A; alg = METIS())
# ```
#
# To add CliqueTrees.jl or Metis.jl to your environment: `pkg> add CliqueTrees Metis`.

# ## Memory and the Element Type
#
# The star is dense: ``n^2`` entries, whatever the graph. Its element type is the element type of `A`, so `Float32`
# weights halve its memory.

D32 = mstar(MinPlus(), Float32.(A))
eltype(D32), Base.format_bytes(sizeof(D32)), Base.format_bytes(sizeof(D))

#-

D32 == D    # integer weights: exact in Float32 too

# `Float32` holds every integer up to ``2^{24} = 16{,}777{,}216`` exactly, so integer distances below that are exact.
# Beyond it, or with fractional weights, `Float32` sums are rounded; use `Float64` when that matters.
#
# | Vertices | `Float32` star | `Float64` star |
# |---:|---:|---:|
# | 10,000 | 0.4 GB | 0.8 GB |
# | 50,000 | 10 GB | 20 GB |
# | 100,000 | 40 GB | 80 GB |

# ## Rows Instead of the Whole Star
#
# If the star does not fit in memory, or only some sources matter, do not compute all of it. Factorize with `mlu`
# and solve for blocks of rows with `rmul!`; each block is ``b \times n``. Reduce each block (a sum for closeness, a
# minimum, a histogram) before computing the next. See [Factorize Once, Query Many Times](queries.md).
#
# On a GPU, `apsp_gpu(A, sources)` computes a block of rows directly.

# ## CPU or GPU
#
# The GPU solver (the CUDA.jl extension, see [GPUs, One or Many](gpus.md)) pays off when the whole star, or a large
# part of it, is needed for a graph with tens of thousands of vertices or more. Measured on one NVIDIA B200 against
# PATHPACK.jl on 16 threads of a 32-core AMD EPYC 7702:
#
# | Problem | 16 CPU threads | One B200 |
# |:---|---:|---:|
# | Stanford bunny mesh, 34,834 vertices, all pairs | 1.61 s | 55.6 ms |
# | PubMed, 19,717 nodes, exact PPR diffusion | 1.93 s | 93.5 ms |
# | Mammoth mesh, 47,004 vertices, exact PPR diffusion | 4.58 s | 124.5 ms |
#
# Use the CPU solver when:
#
# - the graph is directed and its strongly connected components reach one another (a precedence graph, a one-way
#   street network); the GPU solver throws an `ArgumentError` for these;
# - the semiring or element type is not one the GPU takes (for example integer weights for min-plus, or plus-times in
#   `Float32`);
# - only a few rows or columns are needed: a factorization and a few triangular solves are cheap on a CPU;
# - there is no NVIDIA GPU.
#
# Use several GPUs when the star does not fit on one: each GPU then holds only its block of rows.
