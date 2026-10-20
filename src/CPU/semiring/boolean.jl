# The Boolean lattice
#
#   (2ⁿ, ∩, ∪)
#
# - elements are subsets
# - addition is intersection
# - multiplication is union
#
"""
    AndOr()

The Boolean lattice `(2ⁿ, ∩, ∪)` on sets of bits stored in integers: the dual of [`OrAnd`](@ref).
"""
struct AndOr <: AbstractSemiring end

# The dual Boolean lattice
#
#   (2ⁿ, ∪, ∩)
#
# - elements are subsets
# - addition is union
# - multiplication is intersection
#
"""
    OrAnd()

The Boolean lattice `(2ⁿ, ∪, ∩)` on sets of bits stored in integers. With labels as weights,
`mstar(OrAnd(), A)[i, j]` is the set of labels that some path carries from `i` to `j`; with one bit, it is
reachability. Use an integer type such as `Int32`.
"""
const OrAnd = DualQuantale{AndOr}

function sid(::AndOr, a, ::Val{:C})
    return ~a
end

function slte(s::AndOr, a, b)
    return splus(s, a, b, Val(:N)) == b
end

function szero(::AndOr, ::Type{T}, ::Val{:N}) where {T <: Number}
    return typemax(T)
end

function sone(::AndOr, ::Type{T}, ::Val{:N}) where {T}
    return zero(T)
end

function splus(::AndOr, a, b, ::Val{:N})
    return a & b
end

function sprod(::AndOr, a, b, ::Val{:N}, ::Val{:N})
    return a | b
end

function sprod(s::Union{AndOr, OrAnd}, a, b, ::Val{:C}, ::Val{:N})
    return splus(s, ~a, b, Val(:N))
end

function smuladd(s::Union{AndOr, OrAnd}, a, b, c, ::Val{:N}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:N), Val(:N)), c, Val(:N))
end

function smuladd(s::Union{AndOr, OrAnd}, a, b, c, ::Val{:C}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:C), Val(:N)), c, Val(:C))
end

function sstar(s::Union{AndOr, OrAnd}, a)
    return sone(s, a, Val(:N))
end

function issymmetric(::Type{AndOr})
    return true
end

function islattice(::Type{AndOr})
    return true
end
