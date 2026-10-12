# ===== precompilation =====
#
# While the package precompiles, the solver runs on a small graph, so that the first call in a session does not
# compile it: the star, the factorization, and its solves, in the min-plus semiring (Float64 and Float32) and the
# max-min semiring (Float64), on one thread and on several. Other semirings and element types compile on first use.

using PrecompileTools: @setup_workload, @compile_workload

# a 16 x 16 grid (undirected) and a 64-cycle with chords (directed), with arcs from the grid to the cycle only: two
# strongly connected components, fronts of several sizes, and integer weights (deterministic)
function precompile_graph(::Type{T}) where {T}
    I = Int[]; J = Int[]; V = T[]
    arc!(i, j, w) = (push!(I, i); push!(J, j); push!(V, T(w)))
    id(x, y) = x + 16 * (y - 1)

    for y in 1:16, x in 1:16
        x < 16 && (w = 1 + (7x + 11y) % 13; arc!(id(x, y), id(x + 1, y), w); arc!(id(x + 1, y), id(x, y), w))
        y < 16 && (w = 1 + (5x + 3y) % 11; arc!(id(x, y), id(x, y + 1), w); arc!(id(x, y + 1), id(x, y), w))
    end

    for k in 1:64
        arc!(256 + k, 256 + mod1(k + 1, 64), 1 + k % 7)
        k % 4 == 0 && arc!(256 + k, 256 + mod1(k + 17, 64), 2 + k % 5)
    end

    for k in 1:8
        arc!(id(2k, 2k), 256 + 8k, 3)
    end

    return sparse(I, J, V, 320, 320, min)
end

@setup_workload begin
    @compile_workload begin
        for (s, T) in ((MinPlus(), Float64), (MinPlus(), Float32), (MaxMin(), Float64)), nt in (1, 2)
            A = precompile_graph(T); n = size(A, 1)
            mstar(s, A; nt)
            F = mlu(s, A; nt)
            b = fill(szero(s, T, Val(:N)), n); b[1] = sone(s, T, Val(:N))
            lmul!(F, b; nt)
            B = fill(szero(s, T, Val(:N)), n, 3); B[1, 1] = B[2, 2] = B[3, 3] = sone(s, T, Val(:N))
            lmul!(F, B; nt)
            R = permutedims(B)
            rmul!(R, F; nt)
        end
    end
end
