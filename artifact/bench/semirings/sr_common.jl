# Shared by gpu_semirings.jl (GPU) and sr_check.jl (CPU): the weights of each semiring, the CPU reference, the check.
using Serialization
const SEMI = Dict("MinPlus" => P.MinPlus(), "MaxPlus" => P.MaxPlus(), "MinPlusLaw" => P.MinPlusLaw(), "MaxMin" => P.MaxMin(), "MinMax" => P.MinMax(),
                  "MinProd" => P.MinProd(), "MaxProd" => P.MaxProd(), "PlusProd" => P.PlusProd(), "OrAnd" => P.OrAnd(), "AndOr" => P.AndOr(),
                  "MaxPlusLaw" => P.MaxPlusLaw(), "MinProdLaw" => P.MinProdLaw(), "MaxProdLaw" => P.MaxProdLaw())
const TYPES = Dict("Float32" => Float32, "Float64" => Float64, "Int32" => Int32)
parsecfg(c) = (a = split(c, ':'); (String(a[1]), SEMI[a[1]], TYPES[a[2]]))
loadgraph(name) = startswith(name, "grid") ? fromgraph(Graphs.grid([parse(Int, name[5:end]), parse(Int, name[5:end])])) : graph(name)
function offdiag(A0)
    I_, J_, V_ = findnz(A0); k = I_ .!= J_
    return I_[k], J_[k], V_[k]
end
function weights(nm, T, A0)
    n = size(A0, 1)
    nm in ("MinPlus", "MinPlusLaw") && return SparseMatrixCSC{T, Int}(A0)
    nm in ("MaxPlus", "MaxPlusLaw") && return SparseMatrixCSC{T, Int}(-A0)          # (MaxPlusLaw: [-inf, 0])
    I_, J_, _ = offdiag(A0); u = rand(MersenneTwister(19), length(I_))
    nm == "OrAnd" && return sparse(I_, J_, ones(T, length(I_)), n, n)
    nm in ("MaxMin", "MinMax") && return sparse(I_, J_, T <: Integer ? T.(1 .+ floor.(Int, 100 .* u)) : T.(1 .+ 99 .* u), n, n)
    nm == "AndOr" && return sparse(I_, J_, T.(1 .+ floor.(Int, 255 .* u)), n, n)          # nonzero 8-bit masks (zero: typemax, one: 0)
    nm in ("MinProd", "MinProdLaw") && return sparse(I_, J_, T.(1 .+ 0.01 .* u), n, n)
    nm in ("MaxProd", "MaxProdLaw") && return sparse(I_, J_, T.(1 .- 0.01 .* u), n, n)
    if nm == "PlusProd"
        c = 1 .+ 99 .* u; rs = zeros(n); for (i, w) in zip(I_, c); rs[i] += w; end
        return sparse(I_, J_, T.(c ./ (1.1 .* rs[I_])), n, n)
    end
    error("no weights for $nm")
end
# reference columns A* e_v (paths into v), CPU closure in the same semiring, Float64 copies of the same weights
function reference(s, A, cols)
    R = float(eltype(A)) === eltype(A) ? SparseMatrixCSC{Float64, Int}(A) : A
    F = P.ChordalSLU(s, R; alg = AMF()); copyto!(F, R); lu!(F; nt = NT)
    B = unitvecs(s, eltype(R), size(R, 1), cols)
    return lmul!(F, B; nt = NT)
end
function bfs_reference(s, A, cols)                    # Boolean: reachability into v (exact), as a second reference
    n = size(A, 1); T = eltype(A); one_ = P.sone(s, T, Val(:N)); z = P.szero(s, T, Val(:N)); R = fill(z, n, length(cols))
    rv = rowvals(A)
    for (j, v) in enumerate(cols)
        seen = falses(n); q = [v]; seen[v] = true
        while !isempty(q)
            x = pop!(q)
            for p in nzrange(A, x); u = rv[p]; seen[u] || (seen[u] = true; push!(q, u)); end
        end
        for i in 1:n; seen[i] && (R[i, j] = T(1)); end
        R[v, j] = one_
    end
    return R
end
const EXACT = Set(["MaxMin", "MinMax", "OrAnd", "AndOr"])
function check(nm, s, T, G, R)
    nm in EXACT && return (ok = isequal(Float64.(G), Float64.(R)) || isequal(G, R), maxrel = 0.0, mode = "exact")
    if nm == "PlusProd"
        err = maximum(j -> maximum(abs.(Float64.(G[:, j]) .- R[:, j])) / max(maximum(abs.(R[:, j])), floatmin()), axes(R, 2))
        return (ok = err <= (T === Float32 ? 1e-4 : 1e-10), maxrel = err, mode = "normwise per column")
    end
    z = Float64(P.szero(s, T, Val(:N))); err = 0.0; pat = true
    for i in eachindex(R)
        r = R[i]; g = Float64(G[i])
        (r == z) == (g == z) || (pat = false; continue)
        r == z && continue
        err = max(err, r == 0 ? (g == 0 ? 0.0 : Inf) : abs(g - r) / abs(r))
    end
    tol = T === Float64 ? 1e-10 : nm in ("MinProd", "MaxProd", "MinProdLaw", "MaxProdLaw") ? 1e-3 : 1e-4
    return (ok = pat && err <= tol, maxrel = err, mode = "elementwise relative, zero pattern")
end
