# ===== kernel v7 on a subset of the rows =====
#
#   C[rowsC, :] ← C[rowsC, :] ⊕ A[rowsA, :] ⊗ B      (C[rowsC, :] ← A[rowsA, :] ⊗ B with OW)
#
# for the m rows given by row maps (toprows.jl): logical row l (from 0) is storage row rowsC[l + 1] - 1 of
# C (a device vector), offset + l (an Int), or l (nothing); likewise for A. C and A may also pick their
# columns by an index vector, as for kernel v7 (gemm_layout). The structure is sgemx_kernel7!'s (warp
# tiles, two shared stages, register-staged prefetch, the same panel loops, v8's paired steps on
# sm_100): the same operations in the same order per entry, so bit-identical results. The rows of A are
# loaded as scalars (a thread's 4 rows of a float4 are not contiguous in storage), each thread's storage
# rows looked up once; the epilogue is a checked scalar read-⊕-write. No split-K.
#
# (A file of its own: GEMM_TUNE_KEY hashes sgemx.jl and sgemx_simt.jl, and these kernels are chosen
# by a fixed rule, not by the tuner.)

@inline srow0(::Nothing, l) = l
@inline srow0(rows::Integer, l) = rows + l
@inline srow0(rows::AbstractVector, l) = (@inbounds rows[l + 1]) - 1

# the thread's NA float4 of the A panel: rows sr (storage, from 0; -1 past m), column kA + dp; one 16-byte
# load when va (the 4 rows consecutive, the first a multiple of 4, A aligned)
@generated function rows_fetch_a(pA, lda, sr::NTuple{4, Int32}, va::Bool, kA, k, z, live, colsA, ::Val{NA}, ::Val{NT}, ::Val{BM}) where {NA, NT, BM}
    loads = map(0:(NA - 1)) do l
        dm = (l * NT) % (BM ÷ 4) * 4; dp = (l * NT) ÷ (BM ÷ 4)
        @assert dm == 0
        load = quote
            c = kA + Int32($dp)
            ok = c < k
            col = $(colsA === Nothing ? :(Int(c)) : :(ok ? Int(@inbounds colsA[c + Int32(1)]) - 1 : 0))
            p = pA + 4 * col * Int(lda)
            va & ok ? f4(v7_ldg4(p + 4 * Int(sr[1]))) :
            (ok & (sr[1] >= Int32(0)) ? v7_ld(p + 4 * Int(sr[1])) : z, ok & (sr[2] >= Int32(0)) ? v7_ld(p + 4 * Int(sr[2])) : z,
             ok & (sr[3] >= Int32(0)) ? v7_ld(p + 4 * Int(sr[3])) : z, ok & (sr[4] >= Int32(0)) ? v7_ld(p + 4 * Int(sr[4])) : z)
        end
        l == NA - 1 ? :(live ? $load : (z, z, z, z)) : load
    end
    return :($(Expr(:meta, :inline)); ($(loads...),))
end

# C ⊕= acc (C ← acc with OW) for the thread's 8 rows (gi + 4LM·(g - 1) + v) × TN columns, through the row
# and column maps; a group of 4 rows as one 16-byte access where they are consecutive and aligned. The
# loads of each half of the tile come before its stores (as v7_store!); addresses are recomputed from a
# column base and a 32-bit row at each access (32 live 64-bit addresses would spill).
@generated function rows_store!(s, pC, ldc, acc::NTuple{N}, gi, gj, m, n, z, rowsC, colsC, vecC::Bool, ::Val{OW}, ::Val{LM}) where {N, OW, LM}
    LN = 32 ÷ LM; TN = N ÷ 8
    sym(x...) = Symbol(x...)
    r(g, v) = sym(:r, g, :_, v)
    pre = Expr[]
    for g in 1:2, v in 0:3
        dr = 4 * LM * (g - 1) + v
        push!(pre, :($(r(g, v)) = gi + Int32($dr) < m ? srow0(rowsC, gi + Int32($dr)) % Int32 : Int32(-1)))
    end
    for g in 1:2
        push!(pre, :($(sym(:vg, g)) = vecC & ($(r(g, 0)) % Int32(4) == Int32(0)) & ($(r(g, 1)) == $(r(g, 0)) + Int32(1)) &
            ($(r(g, 2)) == $(r(g, 0)) + Int32(2)) & ($(r(g, 3)) == $(r(g, 0)) + Int32(3))))
    end
    body = Expr[]
    for half in 0:(TN ÷ 4 - 1)
        cols = (4 * half + 1):(4 * half + 4)
        lds = Expr[]; sts = Expr[]
        for c in cols
            dc = c <= 4 ? c - 1 : 4 * LN + c - 5
            col = colsC === Nothing ? :(Int(gj) + $dc) : :(Int(@inbounds colsC[gj + Int32($(dc + 1))]) - 1)
            cv = sym(:cv, c); cb = sym(:cb, c)
            push!(lds, :($cv = gj + Int32($dc) < n))
            push!(lds, :($cb = $cv ? pC + 4 * $col * Int(ldc) : pC))
            for g in 1:2
                vq = sym(:vq, c, :_, g)
                xs = [sym(:x, c, :_, g, :_, v) for v in 0:3]; es = [8 * (c - 1) + 4 * (g - 1) + v + 1 for v in 0:3]
                ok(v) = :($cv & ($(r(g, v)) >= Int32(0)))
                ad(v) = :($cb + 4 * Int($(r(g, v))))
                push!(lds, :($vq = $(sym(:vg, g)) & $cv))
                OW || push!(lds, :(($(xs...),) = $vq ? f4(v7_ldg4($(ad(0)))) : ($([:($(ok(v)) ? v7_ld($(ad(v))) : z) for v in 0:3]...),)))
                new = [OW ? :(acc[$(es[v])]) : :(splus(s, acc[$(es[v])], $(xs[v]), Val(:N))) for v in 1:4]
                push!(sts, :(if $vq
                    v7_stg4!($(ad(0)), f4($(new...)))
                else
                    $([:($(ok(v - 1)) && v7_st!($(ad(v - 1)), $(new[v]))) for v in 1:4]...)
                end))
            end
        end
        append!(body, lds); append!(body, sts)
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(pre...)
            $(body...)
        end
        return
    end
end

function sgemx_rows_kernel!(s::AbstractSemiring, pC0::UInt64, ldc::Int32, pA0::UInt64, lda::Int32, pB0::UInt64, ldb::Int32, m::Int32, n::Int32, k::Int32,
        vecA::Bool, vecB::Bool, vecC::Bool, colsA, colsC, rowsA, rowsC, ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TN}, ::Val{OW}, ::Val{PAIR}, ::Val{LM}) where {BM, BN, BK, TN, OW, PAIR, LM}
    V = Float32
    WM = 8 * LM; WN = (32 ÷ LM) * TN
    NT = (BM ÷ WM) * (BN ÷ WN) * 32
    FA = BM * BK ÷ 4; FB = BK * BN ÷ 4
    NA = cld(FA, NT); NB = cld(FB, NT)
    LDB = BN + 4
    ABYTES = Int32(4 * BM * BK)
    BBYTES = Int32(4 * LDB * BK)

    pA = reinterpret(LLVMPtr{V, AS.Global}, pA0)
    pB = reinterpret(LLVMPtr{V, AS.Global}, pB0)
    pC = reinterpret(LLVMPtr{V, AS.Global}, pC0)

    As = CuStaticSharedArray(V, BM * BK * 2)
    Bs = CuStaticSharedArray(V, LDB * BK * 2)
    sAp = pointer(As); sBp = pointer(Bs)

    t = threadIdx().x - Int32(1)
    i0 = (blockIdx().y - Int32(1)) * Int32(BM)
    j0 = (blockIdx().x - Int32(1)) * Int32(BN)
    innerB = j0 + Int32(BN) <= n
    z = szero(s, V, Val(:N))
    u = sone(s, V, Val(:N))
    liveA = FA % NT == 0 || t + Int32((NA - 1) * NT) < Int32(FA)
    liveB = FB % NT == 0 || t + Int32((NB - 1) * NT) < Int32(FB)
    q1 = cld(k, Int32(BK))

    m4 = t % Int32(BM ÷ 4); pa = t ÷ Int32(BM ÷ 4)
    k4 = t % Int32(BK ÷ 4); cb = t ÷ Int32(BK ÷ 4)
    rowA = i0 + Int32(4) * m4
    sr = ntuple(v -> rowA + Int32(v - 1) < m ? (srow0(rowsA, rowA + Int32(v - 1)) % Int32) : Int32(-1), Val(4))
    va = vecA & (sr[1] % Int32(4) == Int32(0)) & (sr[2] == sr[1] + Int32(1)) & (sr[3] == sr[1] + Int32(2)) & (sr[4] == sr[1] + Int32(3))
    colB = j0 + cb
    gB = pB + 4 * (Int(Int32(4) * k4) + Int(colB) * Int(ldb))
    stepB = 4 * BK

    wA = Int32(4) * (pa * Int32(BM) + Int32(4) * m4)
    wB = Int32(4) * (Int32(4) * k4 * Int32(LDB) + cb)
    warp = t ÷ Int32(32); lane = t % Int32(32)
    wm = warp % Int32(BM ÷ WM); wn = warp ÷ Int32(BM ÷ WM)
    lm = lane % Int32(LM); ln = lane ÷ Int32(LM)
    rA = Int32(4) * (wm * Int32(WM) + lm * Int32(4))
    rB = Int32(4) * (wn * Int32(WN) + ln * Int32(4))

    acc = ntuple(_ -> z, Val(8 * TN))

    @inbounds begin
        full = Int32(BK) <= k
        ra = rows_fetch_a(pA, lda, sr, va, pa, k, z, liveA, colsA, Val(NA), Val(NT), Val(BM))
        rb = v7_fetch_b(gB, ldb, Int32(4) * k4, colB, k, n, u, vecB & innerB & full, liveB, Val(NB), Val(NT), Val(BK))
        v7_stash_a!(sAp + wA, ra, liveA, Val(NA), Val(NT), Val(BM))
        v7_stash_b!(sBp + wB, rb, liveB, Val(NB), Val(NT), Val(BK), Val(LDB))
        sync_threads()
        cur = Int32(0)
        q = Int32(1)

        while q <= q1
            more = q < q1

            if more
                gB += stepB
                k0 = q * Int32(BK)
                full = k0 + Int32(BK) <= k
                ra = rows_fetch_a(pA, lda, sr, va, k0 + pa, k, z, liveA, colsA, Val(NA), Val(NT), Val(BM))
                rb = v7_fetch_b(gB, ldb, k0 + Int32(4) * k4, colB, k, n, u, vecB & innerB & full, liveB, Val(NB), Val(NT), Val(BK))
            end

            acc = if PAIR
                v8_panel(min3_op(s), acc, sAp + (cur * ABYTES + rA), sBp + (cur * BBYTES + rB), Val(BK), Val(BM), Val(LDB), Val(TN), Val(LM))
            else
                v7_panel(s, acc, sAp + (cur * ABYTES + rA), sBp + (cur * BBYTES + rB), Val(BK), Val(BM), Val(LDB), Val(TN), Val(LM))
            end

            if more
                cur ⊻= Int32(1)
                v7_stash_a!(sAp + (cur * ABYTES + wA), ra, liveA, Val(NA), Val(NT), Val(BM))
                v7_stash_b!(sBp + (cur * BBYTES + wB), rb, liveB, Val(NB), Val(NT), Val(BK), Val(LDB))
            end

            sync_threads()
            q += Int32(1)
        end
    end

    gi = i0 + wm * Int32(WM) + lm * Int32(4)
    gj = j0 + wn * Int32(WN) + ln * Int32(4)
    rows_store!(s, pC, ldc, acc, gi, gj, m, n, z, rowsC, colsC, vecC, Val(OW), Val(LM))
    return
end

# a row map as a kernel argument: nothing, an Int offset, or a device vector (or view of one)
rows_arg(::Nothing) = nothing
rows_arg(r::Integer) = Int(r)
rows_arg(r::AbstractVector) = r

#
# C[rowsC, :] (⊕)= A[rowsA, :] ⊗ B for m rows (row maps as in srow0: nothing, an offset from 0, a device
# vector of storage rows). In place (C and A the same columns): one tile as wide as C (at most
# ROWS_INPLACE columns), so every block reads all of its rows of A before it writes them. Tiles: 64-row
# warps, the output's width rounded to 32, for outputs up to 128 columns; else 128 × 64 (the tuner's usual
# choice for the closure's U steps). Float32 with kernel-v7 layouts only (top_rows checks).
#
function rows_gemm!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V}, m::Integer, rowsC, rowsA;
        overwrite::Bool = false, inplace::Bool = false) where {V}
    n = size(C, 2); k = size(A, 2)
    @assert size(B) == (k, n)
    (m > 0 && n > 0 && k > 0) || return C
    lc = gemm_layout(C); la = gemm_layout(A); lb = strided_layout(B)
    (V === Float32 && !isnothing(lc) && !isnothing(la) && !isnothing(lb)) || error("rows_gemm!: needs Float32 operands with kernel-v7 layouts")
    (pc, ldc, ic), (pa, lda, ia), (pb, ldb) = lc, la, lb

    if inplace || n <= ROWS_INPLACE
        @assert n <= ROWS_INPLACE
        # (one warp of 8 × 8 tiles, 64 × 32, spills on sm_100: up to 64 columns 8 × 4 tiles, widths in steps of 16)
        BM, BN, TN, LM = n <= 64 ? (64, cld(n, 16) * 16, 4, 8) : (64, cld(n, 32) * 32, 8, 8)
    else
        BM, BN, TN, LM = 128, 64, 8, 4
    end

    rows_launch!(s, Val(BM), Val(BN), Val(8), Val(TN), Val(LM), Val(overwrite), UInt64(UInt(pc)), ldc % Int32, UInt64(UInt(pa)), lda % Int32,
        UInt64(UInt(pb)), ldb % Int32, m % Int32, n % Int32, k % Int32, vec_ok(pa, lda), vec_ok(pb, ldb), vec_ok(pc, ldc), ia, ic, rows_arg(rowsA), rows_arg(rowsC))
    return C
end

function rows_launch!(s, ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TN}, ::Val{LM}, ::Val{OW}, pc, ldc, pa, lda, pb, ldb, m, n, k, veca, vecb, vecc, ia, ic, ra, rc) where {BM, BN, BK, TN, LM, OW}
    @assert v7_ok(Float32, BM, BN, BK, TN, LM)
    pair = pair_ok(s, Float32)
    NT = v7_threads(BM, BN, TN, LM)
    maxregs = pair ? V8_MAXREGS : NT >= 256 ? V7_MAXREGS : V8_MAXREGS
    maxregs = min(maxregs, 16384 ÷ (32 * cld(NT ÷ 32, 4)) ÷ 8 * 8)
    blocks = (cld(Int(n), BN), cld(Int(m), BM))
    @cuda threads = NT blocks = blocks maxregs = maxregs sgemx_rows_kernel!(s, pc, ldc, pa, lda, pb, ldb, m, n, k, veca, vecb, vecc, ia, ic, ra, rc,
        Val(BM), Val(BN), Val(BK), Val(TN), Val(OW), Val(pair), Val(LM))
    return
end
