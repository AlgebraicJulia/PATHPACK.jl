abstract type AbstractSLU{T} <: Factorization{T} end

const TransSLU{T} = TransposeFactorization{T, <:AbstractSLU{T}}

const MaybeTransSLU{T} = Union{
    AbstractSLU{T},
     TransSLU{T},
}

function Base.Matrix(F::AbstractSLU{T}; nt::Integer = nthreads()) where {T}
    n = size(F, 1)
    C = Matrix{T}(undef, n, n)
    return sgetri!(F, C; nt)
end

function Base.parent(F::AbstractSLU)
    return F
end

function Base.adjoint(F::AbstractSLU)
    return TransposeFactorization(F)
end

function Base.transpose(F::AbstractSLU)
    return TransposeFactorization(F)
end

# ===== mlu =====

"""
    mlu(s::AbstractSemiring, A; alg = AMF(), nt = Threads.nthreads())

Factorize the star of the matrix `A` over the semiring `s`, `A* = U* L*`, for queries. A sparse `A` is ordered to
reduce fill (`alg`: any elimination algorithm of CliqueTrees.jl, or a permutation) and factorized with the
multifrontal method; a dense `A` is factorized directly. Then `lmul!(F, B)` gives `A* B` (columns of the star),
`rmul!(B, F)` gives `B A*` (rows), and `Matrix(F)` the whole star. `nt` is the number of threads.

```julia
F = mlu(MinPlus(), A)
B = fill(Inf, 1, size(A, 1)); B[1] = 0.0
rmul!(B, F)    # the distances from vertex 1
```
"""
function mlu(s::AbstractSemiring, A::SparseMatrixCSC; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM, nt::Integer = nthreads())
    F = ChordalSLU(s, A; alg)
    copyto!(F, A)
    return lu!(F; nt)
end

#
# Factorize the closure of a matrix A:
#
#   A* = U* L*
#
function mlu(s::AbstractSemiring, A::AbstractMatrix; nt::Integer = nthreads())
    F = DenseSLU(s, FMatrix(A))
    return lu!(F; nt)
end

# ===== mstar =====

#
# Compute the closure of a matrix A:
#
#   A*
#
"""
    mstar(s::AbstractSemiring, A; alg = AMF(), nt = Threads.nthreads())

Compute the star `A* = I ⊕ A ⊕ A² ⊕ ⋯` of the matrix `A` over the semiring `s`, as a dense matrix. `A[i, j]` is the
weight of the arc `i → j`; `A*[i, j]` is the ⊕-sum over all paths from `i` to `j` of the ⊗-products of their arcs'
weights. With [`MinPlus`](@ref) that is the shortest-path distance between every pair of vertices.

```julia
D = mstar(MinPlus(), A)    # shortest paths
W = mstar(MaxMin(), A)     # widest paths
```

The result has `n²` entries; to compute only some rows or columns, factorize with [`mlu`](@ref).
"""
function mstar(s::AbstractSemiring, A::AbstractMatrix; nt::Integer = nthreads())
    return Matrix(mlu(s, A; nt); nt)
end

function mstar(s::AbstractSemiring, A::SparseMatrixCSC; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM, nt::Integer = nthreads())
    return Matrix(mlu(s, A; alg, nt); nt)
end

# ===== pstar =====

#
# Compute the partial closure of a matrix A
#
#   A*
#
"""
    pstar(s::AbstractSemiring, A::SparseMatrixCSC; alg, nt)

The partial star of a sparse matrix: the entries of `A*` on the sparsity pattern of the factors, as a sparse matrix in
the original vertex order.
"""
function pstar(s::AbstractSemiring, A::SparseMatrixCSC; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM, nt::Integer = nthreads())
    F = mlu(s, A; alg, nt); sgetrp!(F; nt)
    L = sparse(F.L); tril!(L, -1)
    U = sparse(F.U)
    IL, JL, VL = findnz(L)
    IU, JU, VU = findnz(U)
    C = sparse(vcat(IL, IU), vcat(JL, JU), vcat(VL, VU), size(L)...)
    return permute(C, F.rinvp, F.cinvp)
end

# ===== lu! =====

#
# Factorize the closure of a matrix A:
#
#   A* = U* L*
#
function LinearAlgebra.lu!(F::AbstractSLU; nt::Integer = nthreads())
    return sgetrf!(F; nt)
end

# ===== lmul! / rmul! =====

#
# Find the least solution to the
# fixpoint equation
#
#   AX + B = X.
#
function LinearAlgebra.lmul!(F::MaybeTransSLU, B::AbstractVecOrMat; nt::Integer = nthreads())
    P, trans = unwrap(F)
    return sgetrs!(P, Val(:L), trans, B; nt)
end

#
# Find the least solution to the
# fixpoint equation
#
#   XA + B = X.
#
function LinearAlgebra.rmul!(B::AbstractMatrix, F::MaybeTransSLU; nt::Integer = nthreads())
    P, trans = unwrap(F)
    return sgetrs!(P, Val(:R), trans, B; nt)
end

# ===== ldiv! / rdiv! =====

#
# Find the greatest solution to the
# fixpoint equation
#
#   A \ X ∧ B = X.
#
function LinearAlgebra.ldiv!(F::AbstractSLU, B::AbstractVecOrMat; nt::Integer = nthreads())
    return sgetrs!(F, Val(:L), Val(:C), B; nt)
end

function LinearAlgebra.ldiv!(F::TransSLU, B::AbstractVecOrMat; nt::Integer = nthreads())
    return sgetrs!(parent(F), Val(:L), Val(:R), B; nt)
end

#
# Find the greatest solution to the
# fixpoint equation
#
#   X / A ∧ B = X.
#
function LinearAlgebra.rdiv!(B::AbstractMatrix, F::AbstractSLU; nt::Integer = nthreads())
    return sgetrs!(F, Val(:R), Val(:C), B; nt)
end

function LinearAlgebra.rdiv!(B::AbstractMatrix, F::TransSLU; nt::Integer = nthreads())
    return sgetrs!(parent(F), Val(:R), Val(:R), B; nt)
end

# ===== * =====

function Base.:*(F::MaybeTransSLU, B::AbstractVecOrMat)
    return lmul!(F, copy(B))
end

function Base.:*(B::AbstractMatrix, F::MaybeTransSLU)
    return rmul!(copy(B), F)
end

# ===== \ / / =====

function Base.:\(F::MaybeTransSLU, B::AbstractVecOrMat)
    return ldiv!(F, copy(B))
end

function Base.:/(B::AbstractMatrix, F::MaybeTransSLU)
    return rdiv!(copy(B), F)
end
