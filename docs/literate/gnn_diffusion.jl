# # Exact Diffusion for Graph Neural Networks
#
# Graph diffusion convolution (GDC) rewires a graph before a graph neural network sees it. Each node gets new
# neighbours: the nodes that a random walk from it visits most. The most common diffusion is personalized PageRank
# (PPR). PyTorch Geometric implements it as the `GDC` transform, and its exact mode inverts a dense ``n \times n``
# matrix. PATHPACK.jl computes the same matrix as a sum over walks, with a sparse factorization.

#md # ```@raw html
#md # <div class="pp-t-hero">
#md #   <div class="pp-t-clip"><AppVisual name="gnn" ring /></div>
#md #   <div class="pp-t-facts">
#md #     <div class="pp-t-fact"><b>119×</b><span>faster than PyTorch's dense inverse on a 47,004-node mesh, 16 CPU threads, float64</span></div>
#md #     <div class="pp-t-fact"><b>22.6×</b><span>faster on PubMed (19,717 nodes): 1.93 s instead of 43.7 s</span></div>
#md #     <div class="pp-t-fact"><b>69×</b><span>faster on one NVIDIA B200 for the same mesh: 124.5 ms instead of 8.64 s</span></div>
#md #   </div>
#md # </div>
#md # <p class="pp-caption">Personalized PageRank from one paper of the Cora citation graph, drawn radially: the more often walks from the circled paper visit a paper, the closer to the center it sits. The colored third are the papers diffusion makes its neighbours.</p>
#md # ```

# ## Diffusion Is a Sum Over Walks
#
# Let ``A`` be the adjacency matrix of an undirected graph. GDC first adds self-loops and normalizes symmetrically:
#
# ```math
# T = D^{-1/2} (A + I) D^{-1/2},
# ```
#
# where ``D`` is the diagonal matrix of the degrees of ``A + I``. The PPR diffusion with teleport probability ``\alpha``
# is
#
# ```math
# S = \alpha \sum_{k \ge 0} \big((1 - \alpha) T\big)^k
#   = \alpha \big(I - (1 - \alpha) T\big)^{-1}.
# ```
#
# The sum runs over walks of every length, and a walk of length ``k`` is damped by ``(1 - \alpha)^k``. In the
# plus-times semiring (``\oplus = +``, ``\otimes = \times``) this sum is the star of ``B = (1 - \alpha) T``:
#
# ```math
# B^* = I + B + B^2 + \cdots = (I - B)^{-1}.
# ```
#
# The series converges because the eigenvalues of ``T`` lie in ``(-1, 1]``, so the spectral radius of ``B`` is at most
# ``1 - \alpha < 1``. So `α .* mstar(PlusProd(), B)` is the exact GDC matrix. PATHPACK.jl gets it from a sparse LU
# factorization of ``I - B`` that keeps the graph's sparsity, where a dense inverse treats every one of the ``n^2``
# entries as nonzero.

# ## A Small Graph With Communities
#
# Citation graphs have communities: papers mostly cite papers in their own field. A random graph with four
# communities of 150 nodes stands in for one here. Two nodes in the same community are linked with probability 3%;
# two nodes in different communities with probability 0.2%.

using PATHPACK, SparseArrays, LinearAlgebra, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: PlusProd, mstar

Random.seed!(3)
n = 600
community = rand(1:4, n)
src = Int[]; dst = Int[]
for i in 1:n, j in (i + 1):n
    p = community[i] == community[j] ? 0.03 : 0.002
    if rand() < p
        push!(src, i, j); push!(dst, j, i)
    end
end
A = sparse(src, dst, 1.0, n, n)
nnz(A) ÷ 2    # the number of edges

# ## The Diffusion Matrix
#
# Build ``T`` and ``B`` exactly as GDC does, with ``\alpha = 0.15``, and take the star.

α = 0.15
Â = A + sparse(1.0I, n, n)                      # add self-loops
d = vec(sum(Â; dims = 2))                       # degrees
T = Diagonal(1 ./ sqrt.(d)) * Â * Diagonal(1 ./ sqrt.(d))
B = (1 - α) .* T

S = α .* mstar(PlusProd(), B);

# A dense inverse gives the same matrix, up to rounding. This is what PyTorch Geometric's exact mode computes, with
# `torch.inverse`.

S_dense = α .* inv(Matrix(I - B))
maximum(abs.(S - S_dense)) / maximum(abs.(S_dense))

# Since ``T`` is symmetric, so is ``S``. Its entries are sums of products of nonnegative numbers, so none is
# negative; an entry is zero exactly when no walk joins the two nodes.

S ≈ S', minimum(S) >= 0

#-

p = sortperm(community); b = 10; nb = n ÷ b  #hide
Sb = [sum(S[p[(i - 1) * b + 1:i * b], p[(j - 1) * b + 1:j * b]]) / b^2 for i in 1:nb, j in 1:nb]  #hide
figure(panel("S, papers sorted by community", gridmap(1 .- Figs.unit(Sb; log = true)); sub = "mean of 10 × 10 blocks, log scale"),  #hide
    legend = legend("strong", "weak"), caption = "The exact diffusion matrix from mstar(PlusProd(), B). Walks stay mostly inside their community, so the four blocks on the diagonal stand out.")  #hide

# ## Sparsifying, as GDC Does
#
# GDC keeps only the strongest entries of ``S``, so the new graph stays sparse. With top-``k`` sparsification it keeps
# the ``k`` largest entries of each column, and then normalizes each column to sum to one (`normalization_out =
# "col"`). The result is a weighted graph with ``k`` incoming edges per node.

k = 16
rows = Int[]; cols = Int[]; vals = Float64[]
for j in 1:n
    top = partialsortperm(view(S, :, j), 1:k; rev = true)
    w = S[top, j]
    append!(rows, top); append!(cols, fill(j, k))
    append!(vals, w ./ sum(w))
end
S_k = sparse(rows, cols, vals, n, n)
nnz(S_k), all(sum(S_k; dims = 1) .≈ 1)

# The new neighbours are the nodes the walks visit most, so most of them share the node's community:

same = count(community[rows] .== community[cols]) / length(rows)

# ## Into a PyTorch Geometric Pipeline
#
# A PyG graph stores its edges as `edge_index` (two rows: source and target, from 0) and their weights as
# `edge_attr`. The arrays `rows`, `cols` and `vals` above are exactly these, up to the shift to 0-based indices. One
# way to hand them over is a raw binary file:
#
# ```julia
# write("gdc_edges.bin", Int64.([rows cols] .- 1))   # sources, then targets
# write("gdc_weights.bin", Float32.(vals))
# ```
#
# and in Python:
#
# ```python
# import numpy as np, torch
#
# edges = np.fromfile("gdc_edges.bin", dtype=np.int64).reshape(2, -1)
# weights = np.fromfile("gdc_weights.bin", dtype=np.float32)
# data.edge_index = torch.from_numpy(edges)
# data.edge_attr = torch.from_numpy(weights)
# ```
#
# This replaces `T.GDC(..., exact=True)` in the preprocessing. The model and the training loop do not change.
#
# On a GPU, the CUDA extension computes the same star with plus-times in `Float64`. ``B`` is symmetric, which the
# GPU solver needs:
#
# ```julia
# using CUDA
# using PATHPACK.GPU: apsp_gpu
#
# S = α .* apsp_gpu(B; semiring = PlusProd())    # a CuMatrix{Float64}
# ```

# ## Measured
#
# Exact PPR diffusion with ``\alpha = 0.15`` in `float64`, PATHPACK.jl against `torch.inverse`, the dense inverse
# behind PyTorch Geometric's exact GDC. CPU: one 32-core AMD EPYC 7702 node, PATHPACK.jl on 16 threads. GPU: one
# NVIDIA B200. The two answers agree to about ``10^{-13}``, relative.
#
# **On the CPU**, 16 threads:
#
# | Graph | Nodes | PATHPACK.jl | torch.inverse | Speedup |
# |:---|---:|---:|---:|---:|
# | Cora | 2,708 | **48.5 ms** | 182.0 ms | 3.8× |
# | CiteSeer | 3,327 | **59.8 ms** | 316.4 ms | 5.3× |
# | PubMed | 19,717 | **1.93 s** | 43.70 s | 22.6× |
# | Swiss roll kNN | 10,000 | **294 ms** | 6.31 s | 21.4× |
# | Palm mesh | 25,115 | **1.37 s** | 88.81 s | 64.9× |
# | Bunny mesh | 34,834 | **2.80 s** | 230.56 s | 82.2× |
# | Mammoth mesh | 47,004 | **4.58 s** | 545.43 s | 119.2× |
#
# **On one NVIDIA B200:**
#
# | Graph | Nodes | PATHPACK.jl | torch.inverse | Speedup |
# |:---|---:|---:|---:|---:|
# | Cora | 2,708 | **6.5 ms** | 14.9 ms | 2.3× |
# | CiteSeer | 3,327 | **4.8 ms** | 19.4 ms | 4.1× |
# | PubMed | 19,717 | **93.5 ms** | 780.0 ms | 8.3× |
# | Swiss roll kNN | 10,000 | **25.0 ms** | 145.5 ms | 5.8× |
# | Palm mesh | 25,115 | **54.3 ms** | 1.50 s | 27.6× |
# | Bunny mesh | 34,834 | **103.9 ms** | 3.70 s | 35.6× |
# | Mammoth mesh | 47,004 | **124.5 ms** | 8.64 s | 69.4× |
#
# The gap grows with the graph. A dense inverse costs ``n^3`` operations whatever the graph. The sparse factorization
# costs much less on sparse graphs, and meshes, with their small separators, are where it gains most. On PubMed,
# PyTorch Geometric's whole exact GDC transform takes 108.7 s on the same CPU node.
#
# !!! warning "Exact mode only"
#     These numbers compare exact diffusion with exact diffusion. PyTorch Geometric's exact mode is the dense one, and
#     its memory grows as ``n^2``, so on large graphs most users run GDC in its approximate mode (`exact=False`),
#     which estimates each PPR vector with a local push algorithm. The table does not compare against that mode.
#     PATHPACK.jl gives the exact matrix, and its output is still ``n^2`` numbers.
