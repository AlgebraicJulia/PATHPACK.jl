# Shared setup for the benchmarks: packages, environment, timing, provenance, results, and the solver calls we time.
using PATHPACK, CliqueTrees, Metis, Graphs, SparseArrays, LinearAlgebra, Random, Statistics, Printf, JSON, Dates
using CliqueTrees: AMF, METIS, BestFill
const P = PATHPACK.CPU
const MF = CliqueTrees.Multifrontal
using .P: MinPlus, MaxMin, mlu, mstar, dijkstra!, TDWorkspace, sgetre!, sgetrs!

# ---------------- environment ----------------
const DATA = get(ENV, "BENCH_DATA") do; error("BENCH_DATA is not set: point it at the graph data folder (hpc/env.sh does this)"); end
const RESULTS = get(ENV, "BENCH_RESULTS", "results"); mkpath(RESULTS)
const NT = parse(Int, get(ENV, "BENCH_NT", "1"))                 # threads given to the solver and the baselines
NT <= Threads.nthreads() || error("BENCH_NT=$NT exceeds julia -t $(Threads.nthreads())")
const OUT_GB = parse(Float64, get(ENV, "BENCH_OUT_GB", "2.5"))   # memory budget for an n x n closure
const RHS_GB = parse(Float64, get(ENV, "BENCH_RHS_GB", "0.5"))   # memory budget for a block of right-hand sides
const NSRC = parse(Int, get(ENV, "BENCH_NSRC", "64"))            # sampled sources (queries) / targets (all-pairs baseline)
LinearAlgebra.BLAS.set_num_threads(1)

# ---------------- provenance and results ----------------
# git in the library's own checkout (nothing in a download without git, or in a folder inside another repository)
function libgit(args::Cmd)
    try
        d = realpath(pkgdir(PATHPACK)); quiet(c) = readchomp(pipeline(c; stderr = devnull))
        realpath(quiet(`git -C $d rev-parse --show-toplevel`)) == d || return nothing
        return quiet(`git -C $d $args`)
    catch
        return nothing
    end
end
function provenance()
    commit = something(libgit(`rev-parse --short HEAD`), "unknown")
    dirty = (st = libgit(`status --porcelain --untracked-files=no`)) === nothing ? nothing : !isempty(st)
    return Dict("pathpack_commit" => commit, "pathpack_dirty" => dirty, "cliquetrees_version" => string(pkgversion(CliqueTrees)),
                "julia" => string(VERSION), "threads" => NT, "host" => gethostname(),
                "placement" => get(ENV, "BENCH_PLACEMENT", ""), "date" => string(now()))
end
resultfile(script, name) = joinpath(RESULTS, "$(script)_$(name)_nt$(NT).json")
jclean(x) = x
jclean(x::AbstractFloat) = isfinite(x) ? x : nothing                 # JSON has no NaN/Inf
jclean(d::AbstractDict) = Dict(k => jclean(v) for (k, v) in d)
jclean(v::AbstractVector) = map(jclean, v)
function save(script, name, d)
    d = merge(d, Dict("script" => script, "graph" => name, "provenance" => provenance()))
    open(resultfile(script, name), "w") do io; JSON.print(io, jclean(d), 1); end
end

# ---------------- timing ----------------
tmin(f; reps = 3) = minimum(@elapsed(f()) for _ in 1:reps)
tmed(ts) = median(ts)

# ---------------- solves in elimination coordinates (PATHPACK's internal blocked routines: exactly what the public
# sgetrs!/sgetre! call between their two permutations; the renumbering is excluded from timings) ----------------
btf(F) = (F.S.Bptr, F.S.Fptr, F.S.nBptr, MF.pointers(F.S.N), MF.targets(F.S.N), F.Nval)
pool(F, B, nt) = B isa AbstractVector ? nothing : P.spool_mt(F.s, eltype(B), nt)
# general solve with a block of right-hand sides; side :L = column layout (A* B), :R = row layout (B A*)
gen_solve!(F, B, side, nt) = P.sgetrs_mt!(F.s, side, Val(:N), F.L, F.U, btf(F)..., B, pool(F, B, nt), nt)
# subtree-parallel solve for few right-hand sides (workspace built once per factor, block size and thread count)
struct SubWork{W, Q}; W::W; pool::Q; nt::Int; end
subwork(F, k, nt) = SubWork(TDWorkspace(F, k; nt), P.spool_mt(F.s, eltype(F.Nval), nt), nt)
sub_solve!(F, B, side, w::SubWork) = P.sgetrs_mt!(F.s, side, Val(:N), F.L, F.U, btf(F)..., B, w.W, B isa AbstractVector ? nothing : w.pool, w.nt)
# single query: column j (:N, single target) or row j (:T, single source) of A_p^*
elemwork(F, nt) = (TDWorkspace(F, 1; nt), Vector{eltype(F.Nval)}(undef, size(F, 1)), nt)
function elem!(F, j, trans, ew)
    W, x, nt = ew
    fill!(x, P.szero(F.s, eltype(x), trans)); x[j] = P.sone(F.s, eltype(x), trans)
    P.sgetre!(F.s, Val(:L), trans, F.L, F.U, btf(F)..., x, W, nothing, j, nt, F.S.fcc[F.S.S.idx[j]])
    return x
end

# ---------------- output allocation: the first write to a fresh n x n matrix faults its pages in from the OS, which is
# slow (one fault per 4 KB page) and scales poorly with threads; timed as its own phase, columns spread over nt threads ----
function touch!(C, nt)
    m = size(C, 2); k = cld(m, nt)
    Threads.@threads :static for t in 1:nt
        fill!(view(C, :, (t - 1) * k + 1:min(t * k, m)), zero(eltype(C)))   # contiguous columns: a memset
    end
    return C
end

# ---------------- the closure, step by step: X <- U* (triangular inverse), then X <- X L* (solve with n right-hand sides)
# (single component: exactly the fused sgetri!; several components: the fused routine, split reported as missing)
fused_closure!(F, C, nt) = P.sgetri!(F.s, F.L, F.U, F.S.Bptr, F.S.Fptr, F.S.fcc, F.S.nBptr, MF.pointers(F.S.N), MF.targets(F.S.N), F.Nval, C; nt = nt)   # PATHPACK >= 210f34d
function closure_phases!(F, C, nt)
    if P.ncc(F) == 1
        q = P.spool_mt(F.s, eltype(C), nt)
        t_inv = @elapsed P.strtri_mt!(F.s, Val(:N), F.U, C, q, nt)
        t_sol = @elapsed P.strsx_mt!(F.s, Val(:R), Val(:N), Val(:U), F.L, C, q, nt)
        return t_inv + t_sol, t_inv, t_sol
    end
    return (@elapsed fused_closure!(F, C, nt)), nothing, nothing
end
function phases_match_fused(F, nt; nmax = 12000)                # the split must give exactly the fused result
    n = size(F, 1); (P.ncc(F) == 1 && n <= nmax) || return nothing
    C1 = Matrix{eltype(F.Nval)}(undef, n, n); C2 = similar(C1)
    closure_phases!(F, C1, nt); fused_closure!(F, C2, nt)
    ok = isequal(C1, C2); C1 = C2 = nothing; GC.gc(); return ok
end

# ---------------- Dijkstra (PATHPACK's generic label-setting code: any selective semiring) ----------------
unitvecs(s, T, n, cols) = (B = fill(P.szero(s, T, Val(:N)), n, length(cols)); for (j, u) in enumerate(cols); B[u, j] = P.sone(s, T, Val(:N)); end; B)
struct DijWork{I}; f::Vector{I}; b::Vector{I}; end
dijwork(A::SparseMatrixCSC{T, I}) where {T, I} = DijWork(Vector{I}(undef, size(A, 1)), Vector{I}(undef, size(A, 1)))
# one query: x <- A* e_k, i.e. paths into k along A's arcs (single target); pass A^T for single source
function dij!(s, A, k, w::DijWork, x)
    fill!(x, P.szero(s, eltype(x), Val(:N))); x[k] = P.sone(s, eltype(x), Val(:N))
    return P.dijkstra_st!(s, x, w.f, w.b, A)
end

# ---------------- symbolic statistics ----------------
function fstats(F)
    S = F.S.S; res = S.res; sep = S.sep; nnzLU = 0; wmax = 0; ops = 0.0
    for j in vertices(res)
        nn = MF.eltypedegree(res, j); na = MF.eltypedegree(sep, j)
        nnzLU += nn * nn + 2 * nn * na; wmax = max(wmax, nn + na); ops += nn^3 / 3 + nn^2 * na + nn * na^2
    end
    return (nnzLU = nnzLU + MF.ne(F.S.N), maxfront = wmax, nsuper = length(vertices(res)), muladds = ops)
end
# elimination index of original vertex k for a single query, exactly as the public sgetre! maps it
elimindex(F, k, trans::Symbol) = (P.isforward(:U, trans, :L) ? F.cinvp : F.rinvp)[k]
