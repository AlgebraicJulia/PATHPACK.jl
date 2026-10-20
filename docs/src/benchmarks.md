# Benchmarks

PATHPACK.jl against the codes people use for the same problems: GPU codes on the same GPU, graph libraries on the same
CPU cores, and the path step inside tools for machine learning, neuroscience and geometry. Every time is wall-clock,
from the graph in memory to the full answer in memory, and every comparison checks that the answers agree.

```@raw html
<div class="pp-t-bench">
  <BenchChart />
</div>
```

**Hardware.** CPU runs use one node with a 32-core AMD EPYC 7702; PATHPACK.jl runs on 16 threads unless a row says
otherwise, and so do the multithreaded libraries. GPU runs use NVIDIA B200 GPUs, one unless a row says otherwise.
Times are in seconds (s) or milliseconds (ms); in each table the fastest time is in bold.

## GPU Against GPU Codes

All pairs on one B200. ROME and Fast-APSP are recent research codes for all-pairs shortest paths on GPUs; cuGraph is
NVIDIA's graph analytics library.

| Graph | Vertices | PATHPACK.jl | ROME | cuGraph | Fast-APSP |
|:---|---:|---:|---:|---:|---:|
| US road network | 126,146 | **0.139 s** | 0.750 s | 18.05 s | 105.5 s |
| Land-cover grid (ecology) | 67,411 | **0.102 s** | 0.353 s | 10.13 s | |
| Stanford bunny mesh | 34,834 | **55.6 ms** | 0.109 s | 1.70 s | |
| Swiss roll kNN graph | 10,000 | **22.1 ms** | 25.0 ms | 285.7 ms | |
| 164k-vertex cortex | 164k | **0.90 s** | 9.46 s | | |

On the US road network that is 15.9 billion distances in 0.139 s: 5.4× faster than ROME, 130× faster than cuGraph
and 760× faster than Fast-APSP. Over **ROME's own 16 test graphs**, PATHPACK.jl is faster on all 16, by 3.1× in
geometric mean. cuGraph is 6 to 1,600 times slower across the graphs we tested.

### Proteins and Shapes

Molecules as graphs of their atoms, and surface meshes of objects, all pairs on one B200.

| Graph | PATHPACK.jl | ROME | cuGraph |
|:---|---:|---:|---:|
| Spike protein, 21,539 atoms | **47.9 ms** | 93.9 ms | 793 ms |
| ATP synthase | **26.3 ms** | 35.0 ms | 634 ms |
| GroEL | **149 ms** | 265 ms | 6.21 s |
| Virus capsid, 85,620 atoms | **296 ms** | 1.73 s | 13.06 s |
| Space Shuttle | **114 ms** | 285 ms | 8.89 s |
| Apollo 11 capsule | **217 ms** | 562 ms | 49.8 s |
| Saxophone | **123 ms** | 332 ms | 9.78 s |
| Mammoth | **68.5 ms** | 115 ms | 4.05 s |
| Palm | **31.4 ms** | 37.1 ms | 1.47 s |

### Other Semirings

The same land-cover grid (67,411 cells, every cell to every cell) on one B200, in three semirings:

| Semiring | Question | Time |
|:---|:---|---:|
| `MinPlus` | least-cost distance | 0.102 s |
| `MinMax` | worst barrier on the best route | 0.105 s |
| `PlusProd` | expected visits of a random walk | 0.177 s |

## Several GPUs

All pairs on eight B200s. Each GPU computes a block of rows.

| Graph | Vertices | Distances | Time |
|:---|---:|---:|---:|
| France roads | 297,144 | 88.3 billion | 0.343 s |
| Brazil roads | 323,474 | 104.6 billion | 0.392 s |
| Japan roads | 353,713 | 125.1 billion | 0.464 s |
| Colorado roads | 435,666 | 189.8 billion | 0.389 s |
| Stanford dragon mesh | 436,846 | 190.8 billion | 0.699 s |

The work is balanced between the GPUs: the slowest GPU took 2.2% longer than the mean in the median run, and 13.3%
longer at most.

## CPU Against Libraries, Same Cores

PATHPACK.jl and the multithreaded libraries on the same 16 cores of one node. Across the test graphs, the other
libraries are this much slower than PATHPACK.jl:

| Library | Slower by |
|:---|---:|
| graph-tool (closeness) | 1.7 to 9× |
| NetworKit | 5 to 15× |
| SciPy, 16 processes | 3 to 14× |
| Graphs.jl, parallel all pairs | 10 to 42× |

**Closeness of every junction in Paris** (11,840 junctions):

| Code | Threads | Time |
|:---|:---|---:|
| PATHPACK.jl | 16 | **0.282 s** |
| graph-tool | 16 | 0.952 s |
| NetworKit | 16 | 2.49 s |
| igraph | 1 | 19.86 s |
| NetworkX | 1 | 289.75 s |

**Every cell to every cell of a landscape** (67,411 land-cover cells):

| Code | Threads | Time |
|:---|:---|---:|
| PATHPACK.jl | 16 | **8.82 s** |
| NetworKit | 16 | 134.4 s |
| SciPy | 1 | 1007.9 s |

## Inside Other Tools

The all-pairs step of tools in other fields, replaced by one PATHPACK.jl call.

**Exact diffusion for graph neural networks.** Personalized PageRank diffusion (GDC, ``\alpha = 0.15``) in `float64`,
against `torch.inverse`, the dense inverse in PyTorch Geometric's exact GDC. Details on the
[tutorial page](generated/gnn_diffusion.md).

| Graph | Nodes | CPU: PATHPACK.jl | CPU: torch | B200: PATHPACK.jl | B200: torch |
|:---|---:|---:|---:|---:|---:|
| Cora | 2,708 | **48.5 ms** | 182.0 ms | **6.5 ms** | 14.9 ms |
| CiteSeer | 3,327 | **59.8 ms** | 316.4 ms | **4.8 ms** | 19.4 ms |
| PubMed | 19,717 | **1.93 s** | 43.70 s | **93.5 ms** | 780.0 ms |
| Swiss roll kNN | 10,000 | **294 ms** | 6.31 s | **25.0 ms** | 145.5 ms |
| Palm mesh | 25,115 | **1.37 s** | 88.81 s | **54.3 ms** | 1.50 s |
| Bunny mesh | 34,834 | **2.80 s** | 230.56 s | **103.9 ms** | 3.70 s |
| Mammoth mesh | 47,004 | **4.58 s** | 545.43 s | **124.5 ms** | 8.64 s |

The answers agree to about ``10^{-13}``, relative. PyTorch Geometric's exact mode is the dense one; on large graphs
most users run its approximate mode instead, which these numbers do not compare against.

**Isomap.** The shortest-path step on a 10,000-point swiss roll ([tutorial](generated/isomap.md)):

| Code | Hardware | Time |
|:---|:---|---:|
| PATHPACK.jl | 16 CPU threads | **0.206 s** |
| NetworKit | 16 CPU threads | 1.27 s |
| SciPy | 16 processes | 1.68 s |
| scikit-learn Isomap, path step | 1 core | 21.78 s |
| SciPy Dijkstra | 1 core | 21.77 s |
| igraph | 1 core | 31.58 s |

**Brain maps.** Every pair of vertices of a 32k-vertex cortex surface ([tutorial](generated/brain_maps.md)):

| Code | Hardware | Time |
|:---|:---|---:|
| PATHPACK.jl | 16 CPU threads | **2.64 s** |
| NetworKit | 16 CPU threads | 17.2 s |
| neuromaps | 16 workers | 24.1 s |
| BrainSMASH, as in its default workflow | CPU | 1.6 hours |

**Mesh geodesics.** Every pair of vertices of the Stanford bunny (34,834 vertices). PATHPACK.jl, ROME, cuGraph and
SciPy compute graph geodesics, which are on average 6.9% longer than exact surface geodesics. The heat method and MMP
compute geodesics on the surface itself.

| Code | Hardware | Time |
|:---|:---|---:|
| PATHPACK.jl | one B200 | **55.6 ms** |
| PATHPACK.jl | 16 CPU threads | 1.61 s |
| PATHPACK.jl | 1 CPU thread | 12.98 s |
| SciPy Dijkstra | 1 core | 206.7 s |
| Heat method (potpourri3d), about 0.8% error | CPU | 575 s, extrapolated |
| Exact MMP geodesics (pygeodesic) | CPU | 14.7 hours, extrapolated |

## Where PATHPACK.jl Does Not Win

- **Few sources.** On one query of 145 sources, NetworKit is faster than PATHPACK.jl. Single-source methods do work
  per source, and with few sources there is little to share; PATHPACK.jl pays for a factorization of the whole graph.
- **Unweighted grids.** On an unweighted warehouse grid PATHPACK.jl is only 1.15× faster than graph-tool and
  NetworKit.
- **One mesh on the GPU.** ROME is faster on one of our graphs, a biplane mesh, by 1.12×.

## How We Measure

Each time is the wall-clock time from the graph in memory to the full answer in memory: for all pairs, every distance
of the ``n \times n`` result. Loading files and writing results are not counted, for any code. The answers of every
code are checked against each other. Every other code runs with the best settings we found for it. CPU comparisons
use the same cores for every multithreaded code; single-threaded codes are marked as such.

The benchmark scripts and raw measurements are public in the
[PATHPACK.jl repository](https://github.com/AlgebraicJulia/PATHPACK.jl).
