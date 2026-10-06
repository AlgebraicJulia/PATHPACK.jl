# ===== the top, level by level =====
#
# factor_top_batched! factors the top fronts of a plan with static update slots (P.levels) with the
# batched kernels above. Per level:
#
#   assembly              one launch: every child of every front of the level
#   for each 64-pivot step k (the fronts with at least k blocks of pivots, J = their k-th block):
#     diagonal blocks     one launch: L₁₁[J, J] ← LU, TL = L[J, J]*, TU = U[J, J]*, U₁₁[J, J] ← its upper part
#     panels              one launch: [L₁₁[J, R] | U₁₂[J, :]] ← TL [⋯] (also into U₁₁[J, R]), [L₁₁[R, J]; L₂₁[:, J]] ← [⋯] TU
#     trailing update     one launch: L₁₁[R, R], U₁₂[R, :], L₂₁[:, R] ⊕= (panel) (panel)        (R: the pivots after J)
#   Schur complements     one launch: M ← M ⊕ L₂₁ U₁₂
#
# (the assembly also starts each front: merge_diag_gpu! and the zero M; the diagonal blocks after the
# first are factored in the trailing update before them). This is the blocked right-looking LU of each
# front with inverted diagonal blocks, as the per-front path (sgetrf_gpu!, strsx_gpu!, strsx_left_gpu!),
# with the same products; for idempotent ⊕ the factor is bit-identical.
#
# A launch is a grid of (tiles, fronts of the level): each thread block finds its tile from the table of
# fronts, the step and its index (level_tile_kernel!), so the host only lists the launches. The table
# depends only on the plan: it is built on the first factorization and kept (TOP_BATCH), which also lets
# a recorded graph replay it.

const BATCHED_TOP = Ref(true)          # (A/B switch for benchmarks: false runs the per-front path)

struct TopBatch
    fronts::CuVector{NTuple{9, Int64}}       # (L₁₁, U₁₁, L₂₁, U₁₂, M, n₁, n₂, first child, children), levels in order
    kids::CuVector{NTuple{3, Int64}}         # (M_c, na_c, rel pointer)
    work::CuVector                           # TL, TU of each front of a level
    launches::Vector{NTuple{5, Int}}         # (kind, first front - 1, fronts, grid width, step)
end

const TB_ASM, TB_DIAG, TB_PANEL, TB_TRAIL, TB_SCHUR = 1, 2, 3, 4, 5

const TOP_BATCH = WeakKeyDict{Any, TopBatch}()

use_batched_top(P::FactorPlan{Sem, T}) where {Sem, T} =
    BATCHED_TOP[] && !isempty(P.levels) && config().direct_assembly && config().fused_front && sizeof(T) <= 4

# the tiles of a front of n₁ pivots and n₂ separator vertices in a launch at step k (as level_tile)
function level_tiles(kind, n1, n2, k)
    j1 = min(64k, n1); nr = cld(n1 - j1, 64); nc = cld(n2, 64)
    kind == TB_PANEL && return 64 * (k - 1) < n1 ? 2 * (nr + nc) : 0
    kind == TB_TRAIL && return nr * (nr + 2nc)
    return nc * nc
end

function top_batch(P::FactorPlan{Sem, T}) where {Sem, T}
    tb = lock(() -> get(TOP_BATCH, P, nothing), TOP_BATCH)
    isnothing(tb) || return tb
    @assert !CUDA.is_capturing() "the batched schedule must be built before graph capture"
    nb = 64; sz = sizeof(T)
    work = CuVector{T}(undef, 2 * nb * nb * maximum(length, P.levels))
    CUDA.enable_synchronization!(work, false)
    LD, UD, LL, UL = top_arrays(P)
    # (pointer() is slow, ~0.2 µs, and the plan's array fields are not concrete: the addresses as Int64)
    bLD::Int64, bUD::Int64, bLL::Int64, bUL::Int64, bMg::Int64, bMb::Int64 = devaddr.((LD, UD, LL, UL, P.Mg, P.Mb))
    fronts = NTuple{9, Int64}[]; kids = NTuple{3, Int64}[]
    launches = NTuple{5, Int}[]

    for level in P.levels, t in level
        n1, n2, Dp, Lp, out, ks = P.tasks[t]
        k1 = length(kids) + 1

        for (gpu, off, na, rp) in ks
            push!(kids, ((gpu ? bMg : bMb) + (off - 1) * sz, na, rp))
        end

        push!(fronts, (bLD + (Dp - 1) * sz, bUD + (Dp - 1) * sz, bLL + (Lp - 1) * sz, bUL + (Lp - 1) * sz, bMg + (out - 1) * sz, n1, n2, k1, length(ks)))
    end

    f0 = 0                                       # the fronts of a level are consecutive

    for level in P.levels
        nf = length(level)
        dims = [(P.tasks[t][1], P.tasks[t][2]) for t in level]

        push!(launches, (TB_ASM, f0, nf, maximum(sum, dims), 0))

        for k in 1:maximum(d -> cld(d[1], nb), dims)
            k == 1 && push!(launches, (TB_DIAG, f0, nf, 1, k))                # (the next ones: in the trailing update)

            for kind in (TB_PANEL, TB_TRAIL)
                w = maximum(d -> level_tiles(kind, d..., k), dims)
                w > 0 && push!(launches, (kind, f0, nf, w, k))
            end
        end

        w = maximum(d -> level_tiles(TB_SCHUR, d..., 0), dims)
        w > 0 && push!(launches, (TB_SCHUR, f0, nf, w, 0))
        f0 += nf
    end

    up(v) = (d = CuVector(isempty(v) ? [ntuple(_ -> Int64(0), fieldcount(eltype(v)))] : v); CUDA.enable_synchronization!(d, false); d)
    tb = TopBatch(up(fronts), up(kids), work, launches)
    lock(() -> (TOP_BATCH[P] = tb), TOP_BATCH)
    return tb
end

#
# The tile task (as tile_task!) of tile t (from 0) of front f at step k, and whether there is one:
#
#   panels (k)    L₁₁[J, c] ← TL L₁₁[J, c] (also into U₁₁), U₁₂[J, c] ← TL U₁₂[J, c],
#                 L₁₁[r, J] ← L₁₁[r, J] TU, L₂₁[r, J] ← L₂₁[r, J] TU            (in place)
#   trailing (k)  L₁₁[r, c], U₁₂[r, c], L₂₁[r, c] ⊕= X[r, J] Y[J, c]             (r, c after J)
#   Schur         M[r, c] ⊕= L₂₁[r, :] U₁₂[:, c]
#
# over the 64-blocks r, c (in this order; level_tiles counts them). TL, TU are at wl, wu.
#
@inline function level_tile(::Val{KIND}, f::NTuple{9, Int64}, k::Int64, t::Int64, wl::Int64, wu::Int64, ::Type{T}) where {KIND, T}
    l11, u11, l21, u12, mm, n1, n2, _, _ = f
    sz = sizeof(T)
    at(x, ld, i, j) = x + ((i - 1) + (j - 1) * ld) * sz
    none = (false, ntuple(_ -> Int64(0), Val(13)))
    nc = cld(n2, 64)

    if KIND == TB_SCHUR
        t < nc * nc || return none
        i, j = cm_index(t, nc)
        r0 = 64i - 63; c0 = 64j - 63
        return (true, (at(mm, n2, r0, c0), n2, Int64(0), min(64, n2 - r0 + 1), min(64, n2 - c0 + 1), Int64(0),
            at(l21, n2, r0, 1), n2, at(u12, n1, 1, c0), n1, n1, Int64(0), Int64(0)))
    end

    j0 = 64k - 63
    j0 <= n1 || return none
    b = min(64, n1 - j0 + 1); j1 = j0 + b - 1; nr = cld(n1 - j1, 64)

    if KIND == TB_PANEL
        if t < nr                                    # L₁₁[J, c], also into U₁₁
            c0 = j1 + 1 + 64t; x = at(l11, n1, j0, c0)
            return (true, (x, n1, at(u11, n1, j0, c0), b, min(64, n1 - c0 + 1), Int64(1), wl, Int64(64), x, n1, b, Int64(0), Int64(0)))
        end

        t -= nr

        if t < nc                                    # U₁₂[J, c]
            c0 = 1 + 64t; x = at(u12, n1, j0, c0)
            return (true, (x, n1, Int64(0), b, min(64, n2 - c0 + 1), Int64(1), wl, Int64(64), x, n1, b, Int64(0), Int64(0)))
        end

        t -= nc

        if t < nr                                    # L₁₁[r, J]
            r0 = j1 + 1 + 64t; x = at(l11, n1, r0, j0)
            return (true, (x, n1, Int64(0), min(64, n1 - r0 + 1), b, Int64(1), x, n1, wu, Int64(64), b, Int64(0), Int64(0)))
        end

        t -= nr

        if t < nc                                    # L₂₁[r, J]
            r0 = 1 + 64t; x = at(l21, n2, r0, j0)
            return (true, (x, n2, Int64(0), min(64, n2 - r0 + 1), b, Int64(1), x, n2, wu, Int64(64), b, Int64(0), Int64(0)))
        end

        return none
    end

    if t < nr * nr                                   # L₁₁[R, R]
        i, j = cm_index(t, nr)
        r0 = j1 + 64i - 63; c0 = j1 + 64j - 63
        return (true, (at(l11, n1, r0, c0), n1, Int64(0), min(64, n1 - r0 + 1), min(64, n1 - c0 + 1), Int64(0),
            at(l11, n1, r0, j0), n1, at(l11, n1, j0, c0), n1, b, Int64(0), Int64(0)))
    end

    t -= nr * nr

    if t < nr * nc                                   # U₁₂[R, :]
        i, j = cm_index(t, nr)
        r0 = j1 + 64i - 63; c0 = 64j - 63
        return (true, (at(u12, n1, r0, c0), n1, Int64(0), min(64, n1 - r0 + 1), min(64, n2 - c0 + 1), Int64(0),
            at(l11, n1, r0, j0), n1, at(u12, n1, j0, c0), n1, b, Int64(0), Int64(0)))
    end

    t -= nr * nc

    if t < nc * nr                                   # L₂₁[:, R]
        i, j = cm_index(t, nc)
        r0 = 64i - 63; c0 = j1 + 64j - 63
        return (true, (at(l21, n2, r0, c0), n2, Int64(0), min(64, n2 - r0 + 1), min(64, n1 - c0 + 1), Int64(0),
            at(l21, n2, r0, j0), n2, at(l11, n1, j0, c0), n1, b, Int64(0), Int64(0)))
    end

    return none
end

# panels, trailing updates or Schur complements of the fronts foff + 1, … of a level: block (t, f) does
# tile t of front f; TL, TU of front f at work + 2 (f - 1) 64² (as written by level_diag). The first tile
# of a trailing update is the next diagonal block, L₁₁[J + 1, J + 1]: its block then factors it at once
# (level_diag for step k + 1), while the other blocks finish the update.
function level_tile_kernel!(s::AbstractSemiring, op, kind::Val{KIND}, scale::Val, idem::Val, ::Type{T}, fronts, foff::Int32, k::Int64,
        work::Int64) where {KIND, T}
    As, Bt, DG = batch_shared(T)
    f = Int(blockIdx().y); t = Int(blockIdx().x) - 1
    fr = @inbounds fronts[foff + f]
    wl = work + 2 * (f - 1) * 4096 * sizeof(T)
    ok, task = level_tile(kind, fr, k, t, wl, wl + 4096 * sizeof(T), T)
    ok || return
    tile_task!(s, op, T, As, Bt, task...)

    if KIND == TB_TRAIL && t == 0
        sync_threads()                                      # (the block's global writes are visible to it)
        level_diag(s, op, scale, idem, T, As, Bt, DG, fr, k + 1, wl)
    end

    return
end

# the k-th diagonal block of each front of a level that has one (as diag_block_kernel!, LU)
function level_diag_kernel!(s::AbstractSemiring, op, scale::Val, idem::Val, ::Type{T}, fronts, foff::Int32, k::Int64, work::Int64) where {T}
    f = Int(blockIdx().x)
    level_diag(s, op, scale, idem, T, batch_shared(T)..., @inbounds(fronts[foff + f]), k, work + 2 * (f - 1) * 4096 * sizeof(T))
    return
end

@inline function level_diag(s, op, scale::Val, idem::Val, ::Type{T}, CH, RH, DG, fr::NTuple{9, Int64}, k::Int64, wl::Int64) where {T}
    l11, u11, _, _, _, n1, _, _, _ = fr
    j0 = 64k - 63
    j0 <= n1 || return
    sz = sizeof(T)
    e = ((j0 - 1) + (j0 - 1) * n1) * sz
    diag_block!(s, op, scale, Val(true), idem, T, CH, RH, DG, l11 + e, l11 + e, n1, min(64, n1 - j0 + 1), l11 + e, u11 + e, wl, Int64(64),
        wl + 4096 * sz, Int64(64))
    return
end

# (op, a Val or nothing that only the device decides, through a function barrier: a union-typed op
# in the launch loop made the host compiler crash, LLVM 18 LazyCallGraph)
factor_top_batched!(P::FactorPlan{Sem, T}) where {Sem, T} = factor_top_batched!(P, pair_op(P.F.s, T))

function factor_top_batched!(P::FactorPlan{Sem, T}, op) where {Sem, T}
    s = P.F.s
    scale = Val(!isintegral(s))
    B = top_batch(P)
    work::Int64 = devaddr(B.work)
    idem = Val(idem_plus(s, T))

    for (what, off, n, w, k) in B.launches
        o = Int32(off)

        if what == TB_ASM
            @phase FTIMER[] :assemble @cuda threads = ASM_NT blocks = (w, n) assemble_kernel!(s, T, B.fronts, B.kids, P.reltgt, o)
        elseif what == TB_DIAG
            @phase FTIMER[] :lu_diag @cuda threads = TB_NT blocks = n level_diag_kernel!(s, op, scale, idem, T, B.fronts, o, k, work)
        elseif what == TB_PANEL
            @phase FTIMER[] :panel_trsm @cuda threads = TB_NT blocks = (w, n) level_tile_kernel!(s, op, Val(TB_PANEL), scale, idem, T, B.fronts, o, k, work)
        elseif what == TB_TRAIL
            @phase FTIMER[] :lu_gemm @cuda threads = TB_NT blocks = (w, n) level_tile_kernel!(s, op, Val(TB_TRAIL), scale, idem, T, B.fronts, o, k, work)
        else
            @phase FTIMER[] :schur_gemm @cuda threads = TB_NT blocks = (w, n) level_tile_kernel!(s, op, Val(TB_SCHUR), scale, idem, T, B.fronts, o, k, work)
        end
    end

    return P
end
