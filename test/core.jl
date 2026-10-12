using PATHPACK, Random, SparseArrays
using PATHPACK.CPU: MinPlus, MaxMin, OrAnd, mstar

# The reference closure A* = I + A + A^2 + ... by Floyd-Warshall, in the semiring (plus, times, zero, one)
function floydwarshall(plus, times, zero, one, A)
    n = size(A, 1); D = fill(zero, n, n)
    for i in 1:n D[i, i] = one end
    for (i, j, v) in zip(findnz(A)...) D[i, j] = plus(D[i, j], v) end
    for k in 1:n, j in 1:n, i in 1:n D[i, j] = plus(D[i, j], times(D[i, k], D[k, j])) end
    return D
end

@testset "closure against Floyd-Warshall: n = $n, density $p" for (n, p, seed) in ((40, 0.08, 1), (150, 0.02, 2), (300, 0.01, 3))
    A = sprand(MersenneTwister(seed), n, n, p); nonzeros(A) .= 1 .+ 99 .* nonzeros(A)   # a directed graph, weights in [1, 100]
    I, J, _ = findnz(A); B = sparse(I, J, ones(Int32, length(I)), n, n)                  # its pattern, 0/1 (Int32, as on the GPU;
                                                                                          # Bool matrices are not supported)
    @test all(isapprox.(mstar(MinPlus(), A), floydwarshall(min, +, Inf, 0.0, A); rtol = 1e-12))   # shortest paths
    @test mstar(MaxMin(), A) == floydwarshall(max, min, -Inf, Inf, A)                             # widest paths (exact)
    @test mstar(OrAnd(), B) == floydwarshall(|, &, Int32(0), PATHPACK.CPU.sone(OrAnd(), Int32, Val(:N)), B)   # reachability: bitwise or, and
end
