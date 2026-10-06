# ===== fused diagonal block of a small front =====
#
# For a front with n₁ ≤ 64 pivots the old path issues ~11 launches for the diagonal block and the two
# panel solves (LU kernel; for each panel identity, diagonal solve, fill, GEMM, copy). This kernel does
# the LU of the block and both closures TL = L₁₁* (unit lower) and TU = U₁₁* in one launch, with the
# block in shared memory, so that the panel solves are plain GEMMs: L₂₁ ← L₂₁ TU (in place) and
# U₁₂ ← TL U₁₂. These are the same operations as the inversion path of strsx_gpu! / strsx_left_gpu!
# (which the old path takes for panels of more than 64 rows or columns).
#

function front_diag_kernel!(s::AbstractSemiring, ::Val{SCALE}, A::AbstractMatrix{T}, TL::AbstractMatrix{T}, TU::AbstractMatrix{T}) where {SCALE, T}
    S = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    X = CuStaticSharedArray(T, (DIAG_NB + 1, DIAG_NB))
    b = size(A, 1)
    tid = threadIdx().x - 1; nt = blockDim().x
    z = szero(s, T, Val(:N)); o = sone(s, T, Val(:N))

    @inbounds begin
        e = tid
        while e < b * b
            ci, cj = cm_index(e, b)
            S[ci, cj] = A[ci, cj]
            e += nt
        end
        sync_threads()
        #
        # LU, as sgetrf_diag_kernel!
        #
        for p in 1:b
            if SCALE
                sp = sstar(s, S[p, p])
                k = p + 1 + tid
                while k <= b
                    S[k, p] = sprod(s, S[k, p], sp, Val(:N), Val(:N))
                    k += nt
                end
                sync_threads()
            end

            m = b - p
            e = tid
            while e < m * m
                ci, cj = cm_index(e, m)
                k = p + ci
                j = p + cj
                S[k, j] = smuladd(s, S[k, p], S[p, j], S[k, j], Val(:N), Val(:N))
                e += nt
            end
            sync_threads()
        end

        e = tid
        while e < b * b
            ci, cj = cm_index(e, b)
            A[ci, cj] = S[ci, cj]
            e += nt
        end
        q = tid % DIAG_KS
        r = tid ÷ DIAG_KS + 1
        #
        # TU = U₁₁*: X ← I, then column by column X[r, j] ← (X[r, j] ⊕ ⊕_{k<j} X[r, k] S[k, j]) S[j, j]*
        #
        e = tid
        while e < b * b
            ci, cj = cm_index(e, b)
            X[ci, cj] = ci == cj ? o : z
            e += nt
        end
        sync_threads()

        for j in 1:b
            part = z
            if r <= b
                k = 1 + q
                while k < j
                    part = smuladd(s, X[r, k], S[k, j], part, Val(:N), Val(:N))
                    k += DIAG_KS
                end
            end
            part = ks_reduce(s, part)
            if r <= b && q == 0
                acc = splus(s, X[r, j], part, Val(:N))
                SCALE && (acc = sprod(s, acc, sstar(s, S[j, j]), Val(:N), Val(:N)))
                X[r, j] = acc
            end
            sync_threads()
        end

        e = tid
        while e < b * b
            ci, cj = cm_index(e, b)
            TU[ci, cj] = X[ci, cj]
            e += nt
        end
        sync_threads()
        #
        # TL = L₁₁*: X ← I, then row by row X[i, c] ← X[i, c] ⊕ ⊕_{k<i} S[i, k] X[k, c]
        #
        e = tid
        while e < b * b
            ci, cj = cm_index(e, b)
            X[ci, cj] = ci == cj ? o : z
            e += nt
        end
        sync_threads()
        c = r

        for i in 1:b
            part = z
            if c <= b
                k = 1 + q
                while k < i
                    part = smuladd(s, S[i, k], X[k, c], part, Val(:N), Val(:N))
                    k += DIAG_KS
                end
            end
            part = ks_reduce(s, part)
            (c <= b && q == 0) && (X[i, c] = splus(s, X[i, c], part, Val(:N)))
            sync_threads()
        end

        e = tid
        while e < b * b
            ci, cj = cm_index(e, b)
            TL[ci, cj] = X[ci, cj]
            e += nt
        end
    end

    return
end

# L₂₁ ← L₂₁ U₁₁* and U₁₂ ← L₁₁* U₁₂ as GEMMs with the closures TU, TL (W: scratch of n₁² + 2 n₁ n₂)
function front_panels!(s, L₂₁, U₁₂, TL, TU, W, n₁, n₂)
    if inplace_ok(n₁)
        @phase FTIMER[] :panel_trsm sgemx_gpu!(s, L₂₁, L₂₁, TU; overwrite = true)        # in place, one tile wide
    else
        Y = reshape(view(W, (n₁ * n₁ + n₁ * n₂ + 1):(n₁ * n₁ + 2 * n₁ * n₂)), n₂, n₁)
        @phase FTIMER[] :panel_trsm copy_gpu!(Y, L₂₁)
        @phase FTIMER[] :panel_trsm sgemx_gpu!(s, L₂₁, Y, TU; overwrite = true)
    end

    X = reshape(view(W, (n₁ * n₁ + 1):(n₁ * n₁ + n₁ * n₂)), n₁, n₂)
    @phase FTIMER[] :panel_trsm copy_gpu!(X, U₁₂)
    @phase FTIMER[] :panel_trsm sgemx_gpu!(s, U₁₂, TL, X; overwrite = true)
    return
end


# L₁₁[i, j] ← U₁₁[i, j] for i ≤ j: the diagonal block in one array (as combine_gpu! without a front matrix)
function merge_diag_gpu!(L::AbstractMatrix, U::AbstractMatrix)
    function kernel(L, U)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(L, 1)
            @inbounds while j <= size(L, 2)
                i <= j && (L[i, j] = U[i, j])
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(L, 1), size(L, 2), L, U)
    return L
end

# a child's update added straight into the blocks of its parent: positions reltgt[rp:rp+na-1] in the
# parent's front (1:n₁ residual, then separator) select L₁₁, L₂₁, U₁₂ or the parent's update M
function extendadd_direct_gpu!(s::AbstractSemiring, L11, L21, U12, M, buf::CuVector, off::Int, na::Int, reltgt, rp::Int)
    function kernel(s, L11, L21, U12, M, buf, off, na, reltgt, rp)
        v = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        w = blockIdx().y
        n₁ = size(L11, 1)

        if v <= na
            @inbounds begin
                i = Int(reltgt[rp + v - 1]); j = Int(reltgt[rp + w - 1])
                x = buf[off + (w - 1) * na + v - 1]

                if i <= n₁ && j <= n₁
                    L11[i, j] = splus(s, L11[i, j], x, Val(:N))
                elseif j <= n₁
                    L21[i - n₁, j] = splus(s, L21[i - n₁, j], x, Val(:N))
                elseif i <= n₁
                    U12[i, j - n₁] = splus(s, U12[i, j - n₁], x, Val(:N))
                else
                    M[i - n₁, j - n₁] = splus(s, M[i - n₁, j - n₁], x, Val(:N))
                end
            end
        end

        return
    end

    tb = min(256, 32 * cld(na, 32))
    @cuda threads = tb blocks = (cld(na, tb), na) kernel(s, L11, L21, U12, M, buf, off, na, reltgt, rp)
    return
end

# ===== front kernels =====

# F[inj[v], inj[w]] ← F[inj[v], inj[w]] ⊕ M[v, w], M = reshape(buf[off:off + na² - 1], na, na)
function extendadd_gpu!(s::AbstractSemiring, F::AbstractMatrix, buf::CuVector, off::Int, na::Int, relptr, reltgt, rp::Int)
    function kernel(s, F, buf, off, na, reltgt, rp)
        v = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        w = blockIdx().y

        if v <= na
            @inbounds begin
                iv = reltgt[rp + v - 1]
                iw = reltgt[rp + w - 1]
                F[iv, iw] = splus(s, F[iv, iw], buf[off + (w - 1) * na + v - 1], Val(:N))
            end
        end

        return
    end

    tb = min(256, 32 * cld(na, 32))
    @cuda threads = tb blocks = (cld(na, tb), na) kernel(s, F, buf, off, na, reltgt, rp)
    return F
end

# L₁₁[i, j] ← F[i, j] ⊕ (i > j ? L₁₁[i, j] : U₁₁[i, j])
function combine_gpu!(s::AbstractSemiring, L::AbstractMatrix, U::AbstractMatrix, F::AbstractMatrix)
    function kernel(s, L, U, F)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(L, 1)
            @inbounds while j <= size(L, 2)
                L[i, j] = splus(s, F[i, j], i > j ? L[i, j] : U[i, j], Val(:N))
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(L, 1), size(L, 2), s, L, U, F)
    return L
end

# U[i, j] ← L[i, j] for i ≤ j
function copyupper_gpu!(U::AbstractMatrix, L::AbstractMatrix)
    function kernel(U, L)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(L, 1)
            @inbounds while j <= size(L, 2)
                i <= j && (U[i, j] = L[i, j])
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(L, 1), size(L, 2), U, L)
    return U
end

# X ← X ⊕ F[i0 .+ (1:m), j0 .+ (1:n)]   (or X ← F[…] with overwrite)
function addblock_gpu!(s::AbstractSemiring, X::AbstractMatrix, F::AbstractMatrix, i0::Int, j0::Int; overwrite::Bool = false)
    function kernel(s, X, F, i0, j0, ::Val{OVERWRITE}) where {OVERWRITE}
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j = blockIdx().y

        if i <= size(X, 1)
            @inbounds while j <= size(X, 2)
                X[i, j] = OVERWRITE ? F[i0 + i, j0 + j] : splus(s, X[i, j], F[i0 + i, j0 + j], Val(:N))
                j += gridDim().y
            end
        end

        return
    end

    launch2d(kernel, size(X, 1), size(X, 2), s, X, F, i0, j0, Val(overwrite))
    return X
end

# ===== batched kernels: diagonal blocks, tiles, assembly =====
#
# The top of the tree is factored level by level, and each launch below covers the same step of every
# front of a level (and precompute_ops! forms the operators of all large fronts in a few launches):
# a launch is a list of tasks, one per thread block. This replaces the ~30 launches per 64 pivots of
# the per-front path, most of them single-block kernels on a GPU of 100+ SMs, by 2 launches per 64
# pivots per level. A task names its matrices by device address (Int64; the matrices of one launch
# live in different arrays) and leading dimension.

# element k (from 1) of the T array at device address a
@inline gptr(::Type{T}, a::Int64) where {T} = reinterpret(Core.LLVMPtr{T, 1}, a)
@inline gload(::Type{T}, a::Int64, k::Int64) where {T} = unsafe_load(gptr(T, a), k, Val(Base.datatype_alignment(T)))
@inline gstore!(a::Int64, x::T, k::Int64) where {T} = unsafe_store!(gptr(T, a), x, k, Val(Base.datatype_alignment(T)))

# the device address of x[i] (host side)
devaddr(x::CuArray, i::Integer = 1) = reinterpret(Int64, pointer(x, i))

#
# Register tiles: a thread of a 256-thread block holds 4 × 4 entries of a 64 × 64 block, rows 4tx + a + 1
# and columns 4ty + c + 1 (tx = tid mod 16, ty = tid ÷ 16, a, c ∈ 0:3), as entry 4c + a + 1 of a 16-tuple.
# The thread's part of a column (or row) vector in shared memory is then one 128-bit load (ld4), conflict
# free across a warp, and warp w holds the columns 8w + 1:8w + 8.
#
const TB_NT = 256

@inline tile_row(tx, a) = 4tx + a + 1
@inline tile_col(ty, c) = 4ty + c + 1

# X[i, j] ← X[i, j] ⊕ l[i] u[j] on the thread's entries
@inline function rank1(s::AbstractSemiring, X::NTuple{16, T}, l::NTuple{4, T}, u::NTuple{4, T}) where {T}
    return ntuple(k -> smuladd(s, l[((k - 1) & 3) + 1], u[((k - 1) >> 2) + 1], X[k], Val(:N), Val(:N)), Val(16))
end

# two rank-1 updates, X ⊕ l0 u0 ⊕ l1 u1; with op = Val(:min) / Val(:max) (min-plus / max-plus Float32 on
# compute capability 10.x, pair_ok) as 2 adds and one 3-input min / max per entry (FMNMX3; rank2_min3, as
# GEMM kernel v6): the same values. (sm_120 has no FMNMX3: there the 3-input form is 30-50% slower.)
@inline rank2(s::AbstractSemiring, ::Nothing, X::NTuple{16}, l0, u0, l1, u1) = rank1(s, rank1(s, X, l0, u0), l1, u1)
@inline rank2(s::AbstractSemiring, op::Val, X::NTuple{16}, l0, u0, l1, u1) = rank2_min3(op, X, l0, u0, l1, u1)

# the op argument of rank2 for semiring s and element type T (host side)
pair_op(s, ::Type{T}) where {T} = pair_ok(s, T) ? min3_op(s) : nothing

# the thread's entries of the b × b block at sl / su (strictly lower part from sl, the rest from su); zero outside
@inline function load_block(::Type{T}, sl::Int64, su::Int64, ld::Int64, b::Int64, tx, ty, z::T) where {T}
    return ntuple(Val(16)) do k
        i = tile_row(tx, (k - 1) & 3); j = tile_col(ty, (k - 1) >> 2)
        (i <= b && j <= b) ? gload(T, i > j ? sl : su, i + (j - 1) * ld) : z
    end
end

@inline smul4(s, x::NTuple{4, T}, y::T) where {T} = ntuple(a -> sprod(s, x[a], y, Val(:N), Val(:N)), Val(4))

# a[i] = column p as published (unscaled): L[i, p] below p (scaled with LU), XU[i, p] above p and 1 at p (scaled)
@inline function scale_col(s, ::Val{SCALE}, ::Val{LU}, av::NTuple{4, T}, sp::T, p, tx) where {SCALE, LU, T}
    SCALE || return av
    LU && return smul4(s, av, sp)
    return ntuple(a -> tile_row(tx, a - 1) > p ? av[a] : sprod(s, av[a], sp, Val(:N), Val(:N)), Val(4))
end

# W restarted at zero in row Q + 1 right of p (the owner of row p) or in column Q + 1 below p (of column p)
@inline restart_row(::Val{Q}, W::NTuple{16, T}, p, ty, z::T) where {Q, T} =
    ntuple(k -> ((k - 1) & 3) == Q && tile_col(ty, (k - 1) >> 2) > p ? z : W[k], Val(16))
@inline restart_col(::Val{Q}, W::NTuple{16, T}, p, tx, z::T) where {Q, T} =
    ntuple(k -> ((k - 1) >> 2) == Q && tile_row(tx, (k - 1) & 3) > p ? z : W[k], Val(16))

#
# One pivot p = 4p4 + Q + 1 of diag_block!: W ⊕= a b, with the owners of row and column p restarting
# their S entries as X first (unless ⊕ is idempotent and there is no scaling: then L[i, p] ⊕ L[i, p] 1 and
# U[p, j] ⊕ 1 U[p, j] are already the X entries); then the owners of row and column p + 1 publish them.
# (Q static: the thread's row or column p is its entry Q + 1.)
#
@inline function diag_pivot(s::AbstractSemiring, ::Val{SCALE}, ::Val{LU}, ::Val{IDEM}, ::Val{Q}, W::NTuple{16, T}, Sf::NTuple{16, T},
        CH, RH, DG, p4, b, tx, ty, z::T, o::T) where {SCALE, LU, IDEM, Q, T}
    p = 4p4 + Q + 1
    sp = SCALE ? sstar(s, @inbounds(DG[p])) : o
    av = scale_col(s, Val(SCALE), Val(LU), ld4(CH, 4tx + 1 + 64(p - 1)), sp, p, tx)     # a[i]: column p (L below p, XU above)
    bv = ld4(RH, 4ty + 1 + DIAG_LD * (p - 1))                                           # b[j]: row p (U right of p, XL left), 1 at p

    if !(LU && IDEM && !SCALE)
        tx == p4 && (W = restart_row(Val(Q), W, p, ty, z))
        ty == p4 && (W = restart_col(Val(Q), W, p, tx, z))
    end

    W = rank1(s, W, av, bv)
    pn = p + 1

    if pn <= b                                             # publish row and column p + 1 (S from Sf with LU = false)
        qn = (Q + 1) & 3; pn4 = (pn - 1) >> 2

        if tx == pn4
            Base.Cartesian.@nexprs 4 c -> begin
                k = 4c - 3 + qn; j = tile_col(ty, c - 1)
                @inbounds RH[j, pn] = j == pn ? o : (j > pn && !LU) ? Sf[k] : W[k]
                j == pn && (@inbounds DG[pn] = LU ? W[k] : Sf[k])
            end
        end

        if ty == pn4
            Base.Cartesian.@nexprs 4 a -> begin
                k = 4qn + a; i = tile_row(tx, a - 1)
                @inbounds CH[i, pn] = i == pn ? o : (i > pn && !LU) ? Sf[k] : W[k]
            end
        end
    end

    sync_threads()
    return W
end

#
# Pivots p, p + 1 = 4p4 + Q + 1, 4p4 + Q + 2 (Q = 0 or 2) with one barrier, for LU with idempotent ⊕ and no
# scaling (no restarts): row and column p + 1 were published before pivot p, and every thread applies
# pivot p to them for its own rows and columns (a1 = a + a0 S[p, p + 1], b1 = b + L[p + 1, p] b0, the
# owners' operations); then W ⊕= a0 b0 ⊕ a1 b1 (rank2). The kept row and column p + 1 are those before
# pivot p, corrected in the same way when the results are read (pair_fix).
#
@inline function diag_pair(s::AbstractSemiring, op, ::Val{Q}, W::NTuple{16, T}, CH, RH, DG, p4, b, tx, ty, z::T, o::T) where {Q, T}
    p = 4p4 + Q + 1
    av = ld4(CH, 4tx + 1 + 64(p - 1)); bv = ld4(RH, 4ty + 1 + DIAG_LD * (p - 1))

    if p + 1 <= b
        g = @inbounds RH[p + 1, p]; h = @inbounds CH[p + 1, p]            # S[p, p + 1], L[p + 1, p]
        ar = ld4(CH, 4tx + 1 + 64p); br = ld4(RH, 4ty + 1 + DIAG_LD * p)            # as published before pivot p
        a1 = ntuple(a -> tile_row(tx, a - 1) == p + 1 ? o : smuladd(s, av[a], g, ar[a], Val(:N), Val(:N)), Val(4))
        b1 = ntuple(c -> tile_col(ty, c - 1) == p + 1 ? o : smuladd(s, h, bv[c], br[c], Val(:N), Val(:N)), Val(4))
        W = rank2(s, op, W, av, bv, a1, b1)
    else
        W = rank1(s, W, av, bv)
    end

    if p + 2 <= b                                          # publish rows and columns p + 2, p + 3
        q4 = (p + 1) >> 2; qn = (Q + 2) & 3                # (one thread group holds both)

        if tx == q4
            Base.Cartesian.@nexprs 4 c -> begin
                j = tile_col(ty, c - 1)
                @inbounds RH[j, p + 2] = j == p + 2 ? o : W[4c - 3 + qn]
                @inbounds RH[j, p + 3] = j == p + 3 ? o : W[4c - 2 + qn]
                j == p + 2 && (@inbounds DG[p + 2] = W[4c - 3 + qn])
                j == p + 3 && (@inbounds DG[p + 3] = W[4c - 2 + qn])
            end
        end

        if ty == q4
            Base.Cartesian.@nexprs 4 a -> begin
                i = tile_row(tx, a - 1)
                @inbounds CH[i, p + 2] = i == p + 2 ? o : W[4qn + a]
                @inbounds CH[i, p + 3] = i == p + 3 ? o : W[4qn + 4 + a]
            end
        end
    end

    sync_threads()
    return W
end

# the kept value of row (rows = true) or column q at k, final: for q even, pivot q - 1 applied (diag_pair)
@inline function pair_fix(s, ::Val{PAIRS}, CH, RH, q, k, rows::Bool) where {PAIRS}
    @inbounds if rows
        v = RH[k, q]
        PAIRS && iseven(q) && k != q - 1 && (v = smuladd(s, CH[q, q - 1], RH[k, q - 1], v, Val(:N), Val(:N)))
    else
        v = CH[k, q]
        PAIRS && iseven(q) && k != q - 1 && (v = smuladd(s, CH[k, q - 1], RH[q, q - 1], v, Val(:N), Val(:N)))
    end
    return v
end

const DIAG_LD = 68                                         # rows of RH: the output reads RH[j, i] along i (4-way, not 32-way, conflicts)

#
# One b × b diagonal block (b ≤ 64) per thread block, in registers:
#
#   LU = true    S ← LU of S in place (as sgetrf_diag_kernel!), TL = L*, TU = U*
#   LU = false   only the closures TL = L* (L: the strictly lower part of S, unit) and TU = U* of a factor
#
# all three right-looking, over the pivots p = 1, …, b:
#
#   S[i, p] ← S[i, p] S[p, p]*  (i > p, LU)          XU[r, p] ← XU[r, p] S[p, p]*      (r < p, SCALE)
#   S[i, j] ← S[i, j] ⊕ S[i, p] S[p, j]              (i, j > p, LU)
#   XL[i, j] ← XL[i, j] ⊕ S[i, p] XL[p, j]           (i > p ≥ j; XL = I at the start)
#   XU[i, j] ← XU[i, j] ⊕ XU[i, p] S[p, j]           (i ≤ p < j; XU = I at the start)
#
# The three updates of a pivot touch disjoint entries, so they are one rank-1 update W ⊕= a b of one
# accumulator per entry, with a[i] = S[i, p] (i > p), XU[i, p] (i ≤ p) and b[j] = S[p, j] (j > p),
# XL[p, j] (j ≤ p): W[i, j] is S[i, j] while p < min(i, j), then XL[i, j] or XU[i, j] (restarted at
# zero) while p < max(i, j), and then no longer used. An entry's S is final when p reaches min(i, j)
# and its X when p reaches max(i, j): the entries of row and column p, which their owners publish at
# the end of pivot p - 1 (one barrier per pivot, or per two with diag_pair). Every published row and
# column is kept (RH, CH), and
# the results are read from them at the end: U and XL from the rows, L and XU from the columns. Every
# entry costs one multiply-add per pivot instead of up to three, with no masks. The products are those
# of sgetrf_diag_kernel! and strsx_*_diag_shared_kernel!; XL and XU sum them in pivot order instead of
# a tree, the same for idempotent ⊕ (min, max). (With LU = false, S is the factor itself: Sf.)
# Per 64 × 64 block: 10-14 µs, against 42 µs for sgetrf_diag_kernel! and 34 µs for each of the two
# closure kernels on a B200.
#
# A task: (sl, su, lds, b, a, u, tl, ldtl, tu, ldtu). S is read from sl (strictly lower part) and su
# (the rest), leading dimension lds; with LU the factored block goes to a, and its upper part to u if
# u ≠ 0 (as copyupper_gpu!). TL and TU go to tl and tu if nonzero. IDEM: ⊕ is idempotent (min, max).
#
function diag_block_kernel!(s::AbstractSemiring, op, scale::Val, lu::Val, idem::Val, ::Type{T}, tasks, toff::Int32) where {T}
    diag_block!(s, op, scale, lu, idem, T, batch_shared(T)..., (@inbounds tasks[toff + blockIdx().x])...)
    return
end

# the shared memory of the batched kernels: As, Bt of tile_task! are CH, RH, DG of diag_block! (one block may do both)
@inline batch_shared(::Type{T}) where {T} =
    (CuStaticSharedArray(T, (DIAG_NB, DIAG_NB)), CuStaticSharedArray(T, (DIAG_LD, DIAG_NB)), CuStaticSharedArray(T, DIAG_NB))

@inline function diag_block!(s::AbstractSemiring, op, ::Val{SCALE}, ::Val{LU}, ::Val{IDEM}, ::Type{T}, CH, RH, DG, sl::Int64, su::Int64, lds::Int64, b::Int64,
        aout::Int64, uout::Int64, tl::Int64, ldtl::Int64, tu::Int64, ldtu::Int64) where {SCALE, LU, IDEM, T}
    # CH[:, p]: column p as published; RH[:, p]: row p; DG[p] = S[p, p]
    tid = Int(threadIdx().x) - 1
    tx = tid & 15; ty = tid >> 4
    z = szero(s, T, Val(:N)); o = sone(s, T, Val(:N))

    Sf = load_block(T, sl, su, lds, b, tx, ty, z)
    W = LU ? Sf : ntuple(_ -> z, Val(16))
    #
    # publish pivot 1, as diag_pivot does for p + 1
    #
    if tx == 0
        Base.Cartesian.@nexprs 4 c -> begin
            j = tile_col(ty, c - 1)
            @inbounds RH[j, 1] = j == 1 ? o : Sf[4c - 3]
            j == 1 && (@inbounds DG[1] = Sf[1])
        end
    end

    if ty == 0
        Base.Cartesian.@nexprs 4 a -> (i = tile_row(tx, a - 1); @inbounds CH[i, 1] = i == 1 ? o : Sf[a])
    end

    sync_threads()
    sv = Val(SCALE); lv = Val(LU); iv = Val(IDEM)
    PAIRS = LU && IDEM && !SCALE

    if PAIRS                                               # (also publish row and column 2 before pivot 1)
        if tx == 0
            Base.Cartesian.@nexprs 4 c -> begin
                j = tile_col(ty, c - 1)
                @inbounds RH[j, 2] = j == 2 ? o : Sf[4c - 2]
                j == 2 && (@inbounds DG[2] = Sf[4c - 2])
            end
        end

        ty == 0 && Base.Cartesian.@nexprs 4 a -> (i = tile_row(tx, a - 1); @inbounds CH[i, 2] = i == 2 ? o : Sf[4 + a])
        sync_threads()

        for p4 in 0:((b - 1) >> 2)
            W = diag_pair(s, op, Val(0), W, CH, RH, DG, p4, b, tx, ty, z, o)
            4p4 + 3 <= b && (W = diag_pair(s, op, Val(2), W, CH, RH, DG, p4, b, tx, ty, z, o))
        end
    else
        for p4 in 0:((b - 1) >> 2)
            W = diag_pivot(s, sv, lv, iv, Val(0), W, Sf, CH, RH, DG, p4, b, tx, ty, z, o)
            4p4 + 2 <= b && (W = diag_pivot(s, sv, lv, iv, Val(1), W, Sf, CH, RH, DG, p4, b, tx, ty, z, o))
            4p4 + 3 <= b && (W = diag_pivot(s, sv, lv, iv, Val(2), W, Sf, CH, RH, DG, p4, b, tx, ty, z, o))
            4p4 + 4 <= b && (W = diag_pivot(s, sv, lv, iv, Val(3), W, Sf, CH, RH, DG, p4, b, tx, ty, z, o))
        end
    end
    #
    # the results, from the published rows and columns (coalesced: consecutive threads, consecutive rows)
    #
    e = tid
    pv = Val(PAIRS)

    @inbounds while e < DIAG_NB * DIAG_NB
        i = (e & 63) + 1; j = (e >> 6) + 1

        if i <= b && j <= b
            r = pair_fix(s, pv, CH, RH, i, j, true)       # row i at j: U[i, j] (j > i), XL[i, j] (j < i)
            c = pair_fix(s, pv, CH, RH, j, i, false)      # column j at i: L[i, j] (i > j), XU[i, j] (i < j)

            if LU                                          # U right of the diagonal, L (scaled) left of it
                d = DG[i]
                PAIRS && iseven(i) && (d = smuladd(s, CH[i, i - 1], RH[i, i - 1], d, Val(:N), Val(:N)))
                v = i < j ? r : i == j ? d : (SCALE ? sprod(s, c, sstar(s, DG[j]), Val(:N), Val(:N)) : c)
                gstore!(aout, v, i + (j - 1) * lds)
                (uout != 0 && i <= j) && gstore!(uout, v, i + (j - 1) * lds)
            end

            tl != 0 && gstore!(tl, i > j ? r : i == j ? o : z, i + (j - 1) * ldtl)

            if tu != 0                                     # XU (scaled), XU[i, i] = S[i, i]*
                v = i < j ? (SCALE ? sprod(s, c, sstar(s, DG[j]), Val(:N), Val(:N)) : c) :
                    i == j ? (SCALE ? sprod(s, o, sstar(s, DG[i]), Val(:N), Val(:N)) : o) : z
                gstore!(tu, v, i + (j - 1) * ldtu)
            end
        end

        e += TB_NT
    end

    return
end

# ⊕ is idempotent (min or max: the atomic kinds), for diag_block! (host side)
idem_plus(s, ::Type{T}) where {T} = atomic_kind(s, Val(:N), T) isa Union{Val{:min}, Val{:max}}

#
# In the tile kernel each step of a product reads 4 contiguous values of A (As[i, kk], i contiguous) and of
# B (Bt[j, kk] = B[kk, j], rows padded to 68 for the transposing store) as two 128-bit shared loads, for
# 16 multiply-adds.
#
const BT_LD = 68

# the 16 entries of the m × n matrix at a that thread tid moves into shared memory: rows tid mod 64,
# columns tid ÷ 64 + 4r (zero outside)
@inline function fetch_tile(::Type{T}, a::Int64, ld::Int64, m::Int64, n::Int64, tid, z::T) where {T}
    i = tid & 63; j0 = tid >> 6
    return ntuple(r -> (j = j0 + 4(r - 1); (i < m) & (j < n) ? gload(T, a, i + 1 + j * ld) : z), Val(16))
end

@inline function stash_a!(As, x::NTuple{16}, tid)
    i = (tid & 63) + 1; j0 = tid >> 6
    Base.Cartesian.@nexprs 16 r -> (@inbounds As[i, j0 + 4r - 3] = x[r])
    return
end

@inline function stash_b!(Bt, x::NTuple{16}, tid)           # Bt[j, kk] = B[kk, j]
    kk = (tid & 63) + 1; j0 = tid >> 6
    Base.Cartesian.@nexprs 16 r -> (@inbounds Bt[j0 + 4r - 3, kk] = x[r])
    return
end

# X[e:e + 3] for e - 1 a multiple of 4 (in an array aligned to 16 bytes): one 128-bit load for 4-byte numbers
@inline function ld4(X::CuDeviceArray{T}, e::Int64) where {T}
    if sizeof(T) == 4 && T <: Union{Float32, Int32, UInt32}
        v = unsafe_load(reinterpret(Core.LLVMPtr{NTuple{4, VecElement{T}}, CUDA.AS.Shared}, pointer(X, e)), 1, Val(16))
        return (v[1].value, v[2].value, v[3].value, v[4].value)
    else
        return @inbounds (X[e], X[e + 1], X[e + 2], X[e + 3])
    end
end

# X ← X ⊕ As[:, 1:k] Bt[:, 1:k]ᵀ on the thread's entries, two steps at a time
@inline function tile_mac(s::AbstractSemiring, op, X::NTuple{16, T}, As, Bt, k::Int64, tx, ty) where {T}
    ea = 4tx + 1; eb = 4ty + 1

    for _ in 1:(k >> 1)
        X = rank2(s, op, X, ld4(As, ea), ld4(Bt, eb), ld4(As, ea + 64), ld4(Bt, eb + BT_LD))
        ea += 128; eb += 2BT_LD
    end

    isodd(k) && (X = rank1(s, X, ld4(As, ea), ld4(Bt, eb)))
    return X
end

#
# One tile C[1:m, 1:n] (m, n ≤ 64) per thread block:
#
#   X ← A B                         (A: m × k, B: k × n, k in chunks of 64)
#   X ← X D (D: n × n) or D X (D: m × m), if asked
#   C ← X, or C ← C ⊕ X             (and C2 ← the same, if C2 ≠ 0)
#
# A task: (c, ldc, c2, m, n, flags, a, lda, b, ldb, k, d, ldd), flags 1 overwrite, 2 X D, 4 D X. The next
# chunk of A and B is fetched into registers while the current one is computed. With k ≤ 64 both
# operands are in shared memory before C is written, so the task may overwrite its own A or B (a panel
# solve in place).
#
function tile_kernel!(s::AbstractSemiring, op, ::Type{T}, tasks, toff::Int32) where {T}
    As, Bt, _ = batch_shared(T)
    tile_task!(s, op, T, As, Bt, (@inbounds tasks[toff + blockIdx().x])...)
    return
end

@inline function tile_task!(s::AbstractSemiring, op, ::Type{T}, As, Bt, c::Int64, ldc::Int64, c2::Int64, m::Int64, n::Int64, flags::Int64,
        a::Int64, lda::Int64, b::Int64, ldb::Int64, k::Int64, d::Int64, ldd::Int64) where {T}
    tid = Int(threadIdx().x) - 1
    tx = tid & 15; ty = tid >> 4
    z = szero(s, T, Val(:N))
    X = ntuple(_ -> z, Val(16))
    sz = sizeof(T)

    if k > 0
        ra = fetch_tile(T, a, lda, m, min(k, 64), tid, z)
        rb = fetch_tile(T, b, ldb, min(k, 64), n, tid, z)
        q0 = 0

        while q0 < k
            kq = min(64, k - q0)
            stash_a!(As, ra, tid); stash_b!(Bt, rb, tid)
            sync_threads()

            if q0 + 64 < k                                   # the next chunk: columns of A, rows of B
                kn = min(64, k - q0 - 64)
                ra = fetch_tile(T, a + (q0 + 64) * lda * sz, lda, m, kn, tid, z)
                rb = fetch_tile(T, b + (q0 + 64) * sz, ldb, kn, n, tid, z)
            end

            X = tile_mac(s, op, X, As, Bt, kq, tx, ty)
            sync_threads()
            q0 += 64
        end
    end

    if flags & 6 != 0
        if flags & 2 != 0                                    # X D: As ← X, Bt ← D
            Base.Cartesian.@nexprs 16 kk -> (@inbounds As[4tx + ((kk - 1) & 3) + 1, 4ty + ((kk - 1) >> 2) + 1] = X[kk])
            stash_b!(Bt, fetch_tile(T, d, ldd, n, n, tid, z), tid)
            kd = n
        else                                                 # D X: As ← D, Bt ← Xᵀ
            Base.Cartesian.@nexprs 16 kk -> (@inbounds Bt[4ty + ((kk - 1) >> 2) + 1, 4tx + ((kk - 1) & 3) + 1] = X[kk])
            stash_a!(As, fetch_tile(T, d, ldd, m, m, tid, z), tid)
            kd = m
        end

        sync_threads()
        X = tile_mac(s, op, ntuple(_ -> z, Val(16)), As, Bt, kd, tx, ty)
    end

    ow = flags & 1 != 0

    Base.Cartesian.@nexprs 16 kk -> begin
        i = 4tx + ((kk - 1) & 3) + 1; j = 4ty + ((kk - 1) >> 2) + 1

        if i <= m && j <= n
            e = i + (j - 1) * ldc
            v = ow ? X[kk] : splus(s, gload(T, c, e), X[kk], Val(:N))
            gstore!(c, v, e)
            c2 != 0 && gstore!(c2, v, e)
        end
    end

    return
end

const ASM_NT = 256

#
# Assembly of the fronts of a level, straight into the factor (as extendadd_direct_gpu!), one thread
# block per column j of a front [L₁₁ U₁₂; L₂₁ M]:
#
#   L₁₁[1:j, j] ← U₁₁[1:j, j]  (j ≤ n₁: the diagonal block in one array)        M[:, j - n₁] ← 0  (j > n₁)
#   F[rel_c[v], j] ← F[rel_c[v], j] ⊕ M_c[v, w]   for each child c with rel_c[w] = j, v = 1:na_c
#
# The children are taken in their order, with a barrier between two (they may add to the same
# entries), so the sums are those of one extendadd launch per child, without races, for every semiring.
# (Adding all children at once with atomic min / max was no faster on an RTX 5060 Laptop, and 2.6×
# slower on email-Enron, where thousands of children share hub entries.) rel_c is increasing, so w is
# found by bisection. A front: (l11, u11, l21, u12, m, n₁, n₂, k1, nk); a child:
# (address of M_c, na_c, rel pointer).
#
function assemble_kernel!(s::AbstractSemiring, ::Type{T}, fronts, kids, reltgt, foff::Int32) where {T}
    l11, u11, l21, u12, mm, n1, n2, k1, nk = @inbounds fronts[foff + blockIdx().y]
    j = Int(blockIdx().x)
    j > n1 + n2 && return
    HK = CuStaticSharedArray(Int32, ASM_NT)               # the children of a chunk that hold column j, in order
    HW = CuStaticSharedArray(Int32, ASM_NT)               # and its position in them
    WC = CuStaticSharedArray(Int32, ASM_NT >> 5)          # hits per warp
    tid = Int(threadIdx().x) - 1
    lane = tid & 31; wid = tid >> 5
    z = szero(s, T, Val(:N))

    @inbounds begin
        i = tid + 1

        if j <= n1
            while i <= j
                gstore!(l11, gload(T, u11, i + (j - 1) * n1), i + (j - 1) * n1)
                i += ASM_NT
            end
        else
            while i <= n2
                gstore!(mm, z, i + (j - n1 - 1) * n2)
                i += ASM_NT
            end
        end

        sync_threads()
        c0 = k1

        while c0 < k1 + nk
            kc = c0 + tid
            w = 0

            if kc < k1 + nk
                _, na, rp = kids[kc]
                lo = rp; hi = rp + na - 1

                if reltgt[lo] <= j <= reltgt[hi]
                    while lo < hi
                        mid = (lo + hi) >> 1
                        reltgt[mid] < j ? (lo = mid + 1) : (hi = mid)
                    end

                    reltgt[lo] == j && (w = lo - rp + 1)
                end
            end

            mask = vote_ballot_sync(0xffffffff, w > 0)
            lane == 0 && (WC[wid + 1] = count_ones(mask))
            sync_threads()
            base = 0; total = 0

            for v in 1:(ASM_NT >> 5)
                x = Int(WC[v])
                v <= wid && (base += x)
                total += x
            end

            if w > 0
                h = base + count_ones(mask & ((UInt32(1) << lane) - UInt32(1))) + 1
                HK[h] = kc % Int32; HW[h] = w % Int32
            end

            sync_threads()

            for h in 1:total
                mc, na, rp = kids[HK[h]]
                cw = Int(HW[h]) - 1
                v = tid + 1

                while v <= na
                    r = Int(reltgt[rp + v - 1])
                    x = gload(T, mc, v + cw * na)

                    if r <= n1
                        a, e = j <= n1 ? (l11, r + (j - 1) * n1) : (u12, r + (j - n1 - 1) * n1)
                    else
                        a, e = j <= n1 ? (l21, r - n1 + (j - 1) * n2) : (mm, r - n1 + (j - n1 - 1) * n2)
                    end

                    gstore!(a, splus(s, gload(T, a, e), x, Val(:N)), e)
                    v += ASM_NT
                end

                sync_threads()
            end

            c0 += ASM_NT
            sync_threads()
        end
    end

    return
end

