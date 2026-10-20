# Tutorials

Every tutorial is a Julia script whose code runs when this site is built, so the outputs on each page are
PATHPACK.jl's own. Each one has a notebook download under its title, and all of them run as is in a
[Codespace](getting-started.md#Install).

New to PATHPACK.jl? Start with [Get Started](getting-started.md), then [Road Networks](generated/road_networks.md)
and [One Graph, Many Questions](generated/semirings.md).

## The Basics

| Tutorial | What it shows | Semiring |
|:---|:---|:---|
| [Road Networks](generated/road_networks.md) | Every travel time on a street grid, the most central intersection, and rows from a factorization | `MinPlus` |
| [One Graph, Many Questions](generated/semirings.md) | One landscape, three questions: cheapest route, worst barrier, where a random walk goes | `MinPlus`, `MinMax`, `PlusProd` |
| [Shortest and Widest Paths](generated/shortest_paths.md) | A small directed graph, all pairs and single-source queries | `MinPlus`, `MaxMin` |
| [Factorize Once, Query Many Times](generated/queries.md) | `mlu` and triangular solves: some rows or columns without the whole matrix | `MinPlus` |

## Other Questions, Same Call

| Tutorial | What it shows | Semiring |
|:---|:---|:---|
| [Longest Paths and Schedules](generated/critical_paths.md) | Earliest start times and the critical path of a project | `MaxPlus` |
| [Most Reliable Routes](generated/reliability.md) | The route most likely to work when links fail | `MaxProd` |
| [Bandwidth and Widest Paths](generated/widest_paths.md) | The largest capacity between every pair of nodes | `MaxMin`, `MinMax` |
| [Access Audits](generated/access_audits.md) | Who can read which files on a Linux file system, by any route, and the leak a bind mount opens | `OrAnd` |

## Applications

| Tutorial | What it shows | Semiring |
|:---|:---|:---|
| [Exact Diffusion for GNNs](generated/gnn_diffusion.md) | Personalized PageRank diffusion, the matrix PyTorch Geometric's GDC computes | `PlusProd` |
| [Isomap and Manifold Learning](generated/isomap.md) | Unrolling a swiss roll from the distances of its nearest-neighbour graph | `MinPlus` |
| [Distances on Surfaces](generated/brain_maps.md) | Distances along a folded surface, for spatial null models of brain maps | `MinPlus` |

## Scaling Up

| Page | What it shows |
|:---|:---|
| [GPUs, One or Many](generated/gpus.md) | `apsp_gpu` on one GPU and on several, plans, and keeping results on the device |
| [Performance Guide](generated/performance.md) | Threads, orderings, element types and when to use a GPU |
| [Benchmarks](benchmarks.md) | PATHPACK.jl against GPU codes, graph libraries and the tools of each field |
