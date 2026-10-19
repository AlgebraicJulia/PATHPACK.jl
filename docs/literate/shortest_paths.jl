# # Shortest and Widest Paths
#
# This example computes all-pairs shortest paths and widest paths on a small directed graph, and then answers
# single-source and single-target queries from one factorization.

using PATHPACK, SparseArrays, LinearAlgebra
using PATHPACK.CPU: MinPlus, MaxMin, mstar, mlu

# The entry `A[i, j]` is the weight of the arc `i → j`.

A = sparse([1, 1, 2, 3, 4, 4], [2, 3, 4, 2, 1, 5], [3.0, 8.0, 1.0, 4.0, 2.0, 7.0], 5, 5)

# ## All Pairs
#
# In the min-plus semiring the star `A*` holds the distances: `A*[i, j]` is the length of a shortest path from `i` to
# `j`, and `Inf` means that `j` cannot be reached from `i`.

D = mstar(MinPlus(), A)

# In the max-min semiring the same call gives the capacity of the widest path between every pair of vertices.

W = mstar(MaxMin(), A)

# ## Queries
#
# A factorization answers queries without forming the whole star. `lmul!(F, b)` overwrites `b` with `A* b`; with the
# unit vector at vertex 5 (`0` there, the semiring's one, and `Inf`, its zero, elsewhere), that is column 5 of `A*`:
# the distances to vertex 5.

F = mlu(MinPlus(), A)
b = fill(Inf, 5); b[5] = 0.0
lmul!(F, b)

# `rmul!(B, F)` overwrites the rows of `B` with `B A*`; with the unit row vector at vertex 1, that is row 1 of `A*`:
# the distances from vertex 1.

B = fill(Inf, 1, 5); B[1] = 0.0
rmul!(B, F)

# Both agree with the star.

b == D[:, 5] && vec(B) == D[1, :]
