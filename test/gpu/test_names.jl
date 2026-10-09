# The CPU's dense names on the GPU (dense/dense.jl) against the same functions of PATHPACK.CPU,
# for every semiring the GPU computes, in Float32 and Float64. The weights are exact (integers, powers of
# two), so the idempotent semirings must agree bit for bit; plus-times agrees up to rounding.
include(joinpath(@__DIR__, "setup.jl"))
using .CPU: MinPlus, MaxPlus, MinMax, MaxMin, MinProd, MaxProd, MinPlusLaw, MaxPlusLaw, MinProdLaw, MaxProdLaw, PlusProd,
    szero, sone, sgemx!, sgetrf!, strsx!, strtri!, sgetrs!, sgetri!
using CUDA, LinearAlgebra, SparseArrays, Random, Test

# the semirings, and the kind of exact weights each needs (a finite closure)
const CASES = [
    (MinPlus(), :pos), (MaxPlus(), :neg), (MinMax(), :pos), (MaxMin(), :pos), (MinPlusLaw(), :pos), (MaxPlusLaw(), :neg),
    (MinProd(), :up), (MaxProd(), :down), (MinProdLaw(), :up), (MaxProdLaw(), :down), (PlusProd(), :small),
]

function weight(rng, kind, ::Type{T}, d) where {T}
    kind === :pos && return T(rand(rng, 1:100))
    kind === :neg && return -T(rand(rng, 1:100))
    kind === :up && return T(2)^rand(rng, 0:1)
    kind === :down && return T(2)^-rand(rng, 0:1)
    return T(rand(rng, 1:4)) / T(16d)                      # :small, plus-times: a convergent closure
end

# a directed grid (arcs both ways, different weights) and a random strongly connected directed graph
function pattern(name, rng)
    if name === :grid
        nx, ny = 12, 10; id(i, j) = i + (j - 1) * nx; I = Int[]; J = Int[]
        for j in 1:ny, i in 1:nx, (a, b) in ((i + 1, j), (i, j + 1))
            (a <= nx && b <= ny) || continue
            append!(I, (id(i, j), id(a, b))); append!(J, (id(a, b), id(i, j)))
        end
        return I, J, nx * ny
    else
        n = 150; I = collect(1:n); J = [mod1(i + 1, n) for i in 1:n]          # a directed cycle
        for _ in 1:3n
            a, b = rand(rng, 1:n, 2); a == b || (push!(I, a); push!(J, b))
        end
        return I, J, n
    end
end

function graph(name, kind, ::Type{T}, rng) where {T}
    I, J, n = pattern(name, rng)
    d = maximum(values(Dict(i => count(==(i), I) for i in I)))
    return sparse(I, J, [weight(rng, kind, T, d) for _ in I], n, n, (x, y) -> x)
end

dense(s, A::SparseMatrixCSC{T}) where {T} = (D = fill(szero(s, T, Val(:N)), size(A)); for (i, j, v) in zip(findnz(A)...); D[i, j] = v; end; D)

same(s, x, y) = s isa PlusProd ? isapprox(x, y; rtol = eltype(x) == Float32 ? 1e-4 : 1e-10) : isequal(x, y)

gpu_ok(s, T) = try Ext.check_semiring(s, T); true catch e; e isa ArgumentError || rethrow(); false end

skipped = String[]

@testset "CPU names on the GPU" begin
    for (s, kind) in CASES, T in (Float32, Float64)
        name = string(nameof(typeof(s)), " ", T)
        gpu_ok(s, T) || (push!(skipped, name); continue)
        rng = Xoshiro(7)

        @testset "$name dense" begin
            A = dense(s, graph(:grid, kind, T, rng))[1:80, 1:80]
            n = size(A, 1)
            B = [weight(rng, kind, T, 4) for _ in 1:n, _ in 1:n]
            C = [weight(rng, kind, T, 4) for _ in 1:n, _ in 1:n]

            for tA in (Val(:N), Val(:T)), tB in (Val(:N), Val(:T))
                @test same(s, Array(sgemx!(s, tA, tB, CuArray(C), CuArray(A), CuArray(B))), sgemx!(s, tA, tB, copy(C), A, B; nt = 1))
            end

            LU = sgetrf!(s, copy(A); nt = 1)
            @test same(s, Array(sgetrf!(s, CuArray(A))), LU)

            for (side, uplo, diag) in ((:R, :U, :N), (:R, :U, :U), (:R, :L, :U), (:L, :L, :U))
                args = (s, Val(side), Val(:N), Val(uplo), Val(diag))
                @test same(s, Array(strsx!(args..., CuArray(LU), CuArray(B))), strsx!(args..., LU, copy(B); nt = 1))
            end

            for (uplo, diag) in ((:U, :N), (:U, :U), (:L, :U))
                @test same(s, Array(strtri!(s, Val(uplo), Val(diag), CuArray(LU))), strtri!(s, Val(uplo), Val(diag), copy(LU); nt = 1))
            end

            @test same(s, Array(sgetrs!(s, Val(:R), Val(:N), CuArray(LU), CuArray(B))), sgetrs!(s, Val(:R), Val(:N), LU, copy(B); nt = 1))
            @test same(s, Array(sgetri!(s, CuArray(C), CuArray(LU))), sgetri!(s, copy(C), LU; nt = 1))
        end
    end
end

println("not computed on the GPU (check_semiring): ", isempty(skipped) ? "none" : join(skipped, ", "))
