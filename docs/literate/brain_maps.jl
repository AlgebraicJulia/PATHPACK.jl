# # Distances on Surfaces
#
# Many questions about a surface need the distance along it between every pair of its points. Brain imaging is a
# common case: to test whether two maps of the cortex are related, spatial null models build random maps with the
# same spatial smoothness as the real one, and they measure smoothness through distances along the folded cortex.
# Tools such as BrainSMASH and neuromaps start from the full geodesic distance matrix of the cortical surface mesh.
#
# On a triangle mesh, PATHPACK.jl computes graph geodesics: shortest paths along the mesh edges, with each edge as
# long as it is in space.

#md # ```@raw html
#md # <div class="pp-t-hero">
#md #   <div class="pp-t-clip"><AppVisual name="brain" /></div>
#md #   <div class="pp-t-facts">
#md #     <div class="pp-t-fact"><b>2.6 s</b><span>for every pair of a 32k-vertex cortex on 16 CPU threads; BrainSMASH's default workflow takes 1.6 hours</span></div>
#md #     <div class="pp-t-fact"><b>6.5×</b><span>faster than NetworKit on the same 16 threads (17.2 s)</span></div>
#md #     <div class="pp-t-fact"><b>0.90 s</b><span>for a 164k-vertex cortex, 27 billion distances, on one NVIDIA B200</span></div>
#md #   </div>
#md # </div>
#md # <p class="pp-caption">Distance from one point of the cortex, measured along the folded surface.</p>
#md # ```

# ## A Mesh With Known Geodesics
#
# On a sphere the exact geodesic distance is known: the great-circle distance, the angle between two points for a
# unit sphere. So a sphere mesh shows how far graph geodesics are from exact ones. Build an icosphere: start from an
# icosahedron and split every triangle into four, projecting the new vertices onto the sphere.

using PATHPACK, SparseArrays, LinearAlgebra, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: MinPlus, mstar

function icosphere(levels)
    φ = (1 + sqrt(5)) / 2
    corners = ((-1, φ, 0), (1, φ, 0), (-1, -φ, 0), (1, -φ, 0),
        (0, -1, φ), (0, 1, φ), (0, -1, -φ), (0, 1, -φ),
        (φ, 0, -1), (φ, 0, 1), (-φ, 0, -1), (-φ, 0, 1))
    V = [normalize(collect(Float64, c)) for c in corners]
    F = [(1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12),
        (2, 6, 10), (6, 12, 5), (12, 11, 3), (11, 8, 7), (8, 2, 9),
        (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10),
        (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2)]
    for _ in 1:levels
        mids = Dict{Tuple{Int, Int}, Int}()
        mid(a, b) = get!(mids, minmax(a, b)) do
            push!(V, normalize(V[a] + V[b])); length(V)
        end
        F = [f for (a, b, c) in F for f in (
            (a, mid(a, b), mid(c, a)), (b, mid(b, c), mid(a, b)),
            (c, mid(c, a), mid(b, c)), (mid(a, b), mid(b, c), mid(c, a)))]
    end
    return V, F
end

# The graph of a mesh has one edge per triangle side, weighted by its length.

function mesh_graph(V, F)
    src = Int[]; dst = Int[]; len = Float64[]
    for (a, b, c) in F, (u, v) in ((a, b), (b, c), (c, a))
        ℓ = norm(V[u] - V[v])
        push!(src, u, v); push!(dst, v, u); push!(len, ℓ, ℓ)
    end
    n = length(V)
    return sparse(src, dst, len, n, n, min)
end

V, F = icosphere(4)
A = mesh_graph(V, F)
n = length(V)

# ## Graph Geodesics Against Exact Ones

great_circle(V) = [acos(clamp(dot(p, q), -1, 1)) for p in V, q in V]
mean_ratio(D, E) = (off = .!I(size(D, 1)); sum(D[off] ./ E[off]) / count(off))

D = mstar(MinPlus(), A)
exact = great_circle(V)
off = .!I(n)
mean_ratio(D, exact), maximum(D[off] ./ exact[off])

# Graph geodesics are longer: a path along mesh edges zigzags where the true geodesic goes straight across the
# triangles. Here they are about 7% longer on average, and up to about 23% for some pairs.
#
# A finer mesh does not remove this. The zigzag keeps its shape as the triangles shrink, so the average overestimate
# does not shrink:

for levels in 2:4
    Vₗ, Fₗ = icosphere(levels)
    Dₗ = mstar(MinPlus(), mesh_graph(Vₗ, Fₗ))
    excess = 100 * (mean_ratio(Dₗ, great_circle(Vₗ)) - 1)
    println(length(Vₗ), " vertices: ", round(excess; digits = 1), "% longer")
end

# On real meshes we measured the same effect: on the Stanford bunny (34,834 vertices) graph geodesics are 6.9% longer
# than exact surface geodesics, on average. Whether that matters depends on how the distances are used.
#
# !!! note "Graph geodesics, not surface geodesics"
#     PATHPACK.jl computes exact shortest paths in the mesh graph. These are not exact geodesics on the surface. Exact
#     surface methods are far slower for all pairs; see the measurements below.

# ## Spatial Autocorrelation
#
# With every distance at hand, statistics over all pairs of vertices are one line each. Moran's ``I`` measures how
# similar a map's values are at nearby points, with weights ``w_{ij} = 1 / d_{ij}``. White noise has ``I`` close to 0.
# A smooth map, made here by blurring the noise with a kernel of the geodesic distance, has a clearly positive ``I``.

Random.seed!(5)
noise = randn(n)
smooth = exp.(-D ./ 0.3) * noise
W = 1 ./ D; W[diagind(W)] .= 0
function moran(x)
    y = x .- sum(x) / length(x)
    return length(x) / sum(W) * (y' * W * y) / (y' * y)
end
moran(noise), moran(smooth)

#-

view = normalize([0.35, 0.8, 0.5]); e1 = normalize(cross(view, [0.0, 0.0, 1.0])); e2 = cross(view, e1)  #hide
front = [v for v in 1:n if dot(V[v], view) > 0]  #hide
src0 = front[argmax([dot(V[v], view) for v in front])]  #hide
sph(val, t) = panel(t, scatter([dot(V[v], e1) for v in front], [dot(V[v], e2) for v in front], val[front]; w = 260, h = 260, r = 2.6))  #hide
figure(sph(Figs.unit(D[src0, :]), "distance from the center"), sph(Figs.unit(noise), "noise"), sph(Figs.unit(smooth), "smoothed by distance");  #hide
    caption = "The front half of the sphere. Left: a row of D. Middle and right: the noise and the field smoothed with exp(-D / 0.3), whose Moran's I is printed above.")  #hide

# Spatial null models work with the same matrix. BrainSMASH, for one, generates random maps whose variogram matches
# the real map's, from the map and the distance matrix. PATHPACK.jl can write the matrix for it to read:
#
# ```julia
# write("cortex_distances.bin", Float32.(D))    # symmetric: any order
# ```
#
# ```python
# import numpy as np
# from brainsmash.mapgen.base import Base
#
# D = np.fromfile("cortex_distances.bin", dtype=np.float32).reshape(n, n)
# surrogates = Base(x=brain_map, D=D)(n=1000)    # 1,000 null maps
# ```
#
# A real cortical surface from neuromaps loads as vertices and triangles, and goes through `mesh_graph` as above.
# For the 164k-vertex fsaverage surface the matrix holds about 27 billion distances, about 107 GB in `Float32`. If
# that does not fit, compute it in blocks of rows (`apsp_gpu(A, sources)` on a GPU, or `rmul!` with a factorization
# on the CPU), or split it over several GPUs.

# ## Measured
#
# **Cortex surfaces** (fsaverage meshes, through neuromaps), every pair of vertices. CPU: one 32-core AMD EPYC 7702
# node.
#
# | Mesh | Code | Hardware | Time |
# |:---|:---|:---|---:|
# | 32k-vertex cortex | **PATHPACK.jl** | 16 CPU threads | **2.64 s** |
# | 32k-vertex cortex | NetworKit | 16 CPU threads | 17.2 s |
# | 32k-vertex cortex | neuromaps | 16 workers | 24.1 s |
# | 32k-vertex cortex | BrainSMASH, as in its default workflow | CPU | 1.6 hours |
# | 164k-vertex cortex | **PATHPACK.jl** | one NVIDIA B200 | **0.90 s** |
# | 164k-vertex cortex | ROME | one NVIDIA B200 | 9.46 s |
#
# **Mesh geodesics** on the Stanford bunny, 34,834 vertices, every pair. The first rows compute graph geodesics; the
# last two compute geodesics on the surface itself, which are more accurate and much slower.
#
# | Code | Hardware | Time |
# |:---|:---|---:|
# | **PATHPACK.jl** | one NVIDIA B200 | **55.6 ms** |
# | ROME | one NVIDIA B200 | 0.109 s |
# | cuGraph | one NVIDIA B200 | 1.70 s |
# | **PATHPACK.jl** | 16 CPU threads | **1.61 s** |
# | **PATHPACK.jl** | 1 CPU thread | 12.98 s |
# | SciPy Dijkstra | 1 core | 206.7 s |
# | heat method (potpourri3d), about 0.8% error | CPU | 575 s, extrapolated |
# | exact MMP geodesics (pygeodesic) | CPU | 14.7 hours, extrapolated |
#
# The last two times are extrapolated to all pairs.
