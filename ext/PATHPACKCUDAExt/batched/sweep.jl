# ===== kernels =====

#
# One block per (front f, chunk of right-hand sides); thread t owns row t.
#
#   C₁ ← C₁ U₁₁*
#   C₂ ← C₂ ⊕ C₁ U₁₂       (atomic)
#
function upward_kernel!(s::AbstractSemiring, trans::Val, scale::Val, C::AbstractMatrix{T}, order, off::Int,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    f = @inbounds order[off + blockIdx().x]
    t = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    while t <= size(C, 1)                       # each thread may own several rows (coarsening)
        upward_front!(s, trans, scale, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, Val(true))
        t += blockDim().x * gridDim().y
    end

    return
end

# blocks of tb rows covering nrhs rows (the batched kernels loop over rows, so fewer blocks also work)
rhs_blocks(nrhs::Integer, tb::Integer) = max(1, cld(nrhs, tb))

#
# Row t of C through front f of the U sweep. ATOMIC selects an atomic
# scatter (other threads may update the same entries) or a plain one
# (this thread owns row t).
#
@inline function upward_front!(s::AbstractSemiring, trans::Val, ::Val{SCALE}, C::AbstractMatrix{T}, t, f,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, ::Val{ATOMIC}) where {SCALE, T, ATOMIC}
    @inbounds begin
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
        Sp = Sptr[f]; na = Sptr[f + 1] - Sp
        Dp = Dptr[f]; Lp = Lptr[f]

        for j in 1:nn
            acc = C[t, Rp + j - 1]

            for k in 1:(j - 1)
                acc = smuladd(s, C[t, Rp + k - 1], Dval[Dp + (j - 1) * nn + k - 1], acc, Val(:N), trans)
            end

            if SCALE
                acc = sprod(s, acc, sstar(s, Dval[Dp + (j - 1) * nn + j - 1]), Val(:N), trans)
            end

            C[t, Rp + j - 1] = acc
        end

        for r in 1:na
            m = szero(s, T, trans)

            for j in 1:nn
                m = smuladd(s, C[t, Rp + j - 1], Lval[Lp + (r - 1) * nn + j - 1], m, Val(:N), trans)
            end

            c = Stgt[Sp + r - 1]

            if ATOMIC
                atomic_splus!(s, trans, C, t, c, m)
            else
                C[t, c] = splus(s, C[t, c], m, trans)
            end
        end
    end

    return
end



#
#   C₁ ← C₁ ⊕ C₂ L₂₁
#   C₁ ← C₁ L₁₁*           (unit diagonal)
#
# Up to DOWN_NB residual columns of row t are kept in registers: each
# gathered separator value C[t, sep[r]] is loaded once for all of them, and
# when nn ≤ DOWN_NB the unit-lower solve with L₁₁ runs in registers too.
#
const DOWN_NB = 8



#
# Register-blocked L sweep: one compile-time residual width per front (nn ≤ 8, no masked lanes),
# the residual values of row t in registers, and four separator loads in flight. Same operations,
# same order per entry as downward_front!. (Found by the C++ port experiment: its "reg" kernel, written
# in Julia here, is as fast as the C++ one.)
#
# zi: the residual entries C[t, res] are known to be the semiring zero (not read; see skip_fill)
@generated function down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, ::Val{NN}, zi::Bool = false) where {NN}
    xs = [Symbol(:x, j) for j in 1:NN]
    loads = [:($(xs[j]) = zi ? szero(s, eltype(C), trans) : C[t, Rp + $(j - 1)]) for j in 1:NN]
    upd = [:($(xs[j]) = smuladd(s, c, Lval[Lp + $(j - 1) * na + r - 1], $(xs[j]), Val(:N), trans)) for j in 1:NN]
    solve = Expr[]

    for j in NN:-1:1, k in (j + 1):NN
        push!(solve, :($(xs[j]) = smuladd(s, $(xs[k]), Dval[Dp + $(j - 1) * NN + $(k - 1)], $(xs[j]), Val(:N), trans)))
    end

    stores = [:(C[t, Rp + $(j - 1)] = $(xs[j])) for j in 1:NN]

    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(loads...)
            r = 1

            while r + 3 <= na
                c1 = C[t, Stgt[Sp + r - 1]]; c2 = C[t, Stgt[Sp + r]]; c3 = C[t, Stgt[Sp + r + 1]]; c4 = C[t, Stgt[Sp + r + 2]]
                c = c1; $(upd...); r += 1
                c = c2; $(upd...); r += 1
                c = c3; $(upd...); r += 1
                c = c4; $(upd...); r += 1
            end

            while r <= na
                c = C[t, Stgt[Sp + r - 1]]
                $(upd...)
                r += 1
            end

            $(solve...)
            $(stores...)
        end

        return
    end
end

# wide = Val(false): fronts wider than 8 through downward_front! (fewer registers; the capped layered walk)
@inline function downward_front_reg!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, zi::Bool = false, ::Val{WIDE} = Val(true)) where {WIDE}
    @inbounds begin
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
        Sp = Sptr[f]; na = Sptr[f + 1] - Sp
        Dp = Dptr[f]; Lp = Lptr[f]
    end

    if nn == 1
        down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(1), zi)
    elseif nn == 2
        down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(2), zi)
    elseif nn == 3
        down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(3), zi)
    elseif nn == 4
        down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(4), zi)
    elseif nn <= 8
        nn == 5 ? down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(5), zi) :
        nn == 6 ? down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(6), zi) :
        nn == 7 ? down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(7), zi) :
                  down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(8), zi)
    elseif WIDE
        downward_front_wide!(s, trans, C, t, Rp, Sp, na, nn, Dp, Lp, Stgt, Dval, Lval, zi)
    else
        downward_front!(s, trans, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval, zi)
    end

    return
end

#
# A front wider than 8 columns, in chunks of WIDE_NB from its last columns: a chunk gathers each
# separator value, and each finished column after it, once for its columns (downward_front! gathers
# them once per column), then solves with its block of L₁₁ in registers. The same terms as downward_front!, the
# finished columns before those of the chunk: bit-identical for idempotent ⊕ (min-plus, max-plus,
# max-min), a rounding change otherwise.
#
const WIDE_NB = 4                    # (8 needs ~170 registers next to down_reg!'s variants in one kernel)

@inline function downward_front_wide!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, t, Rp, Sp, na, nn, Dp, Lp, Stgt, Dval, Lval, zi::Bool)
    j1 = nn

    while j1 >= 1
        j0 = max(1, j1 - WIDE_NB + 1)
        w = j1 - j0 + 1

        if w == WIDE_NB
            down_chunk!(s, trans, C, t, Rp, Sp, na, nn, Dp, Lp, Stgt, Dval, Lval, j0, Val(WIDE_NB), zi)
        else
            down_chunk_dispatch!(s, trans, C, t, Rp, Sp, na, nn, Dp, Lp, Stgt, Dval, Lval, j0, w, zi)
        end

        j1 = j0 - 1
    end

    return
end

@inline function down_chunk_dispatch!(s, trans, C, t, Rp, Sp, na, nn, Dp, Lp, Stgt, Dval, Lval, j0, w, zi)
    w == 1 ? down_chunk!(s, trans, C, t, Rp, Sp, na, nn, Dp, Lp, Stgt, Dval, Lval, j0, Val(1), zi) :
    w == 2 ? down_chunk!(s, trans, C, t, Rp, Sp, na, nn, Dp, Lp, Stgt, Dval, Lval, j0, Val(2), zi) :
             down_chunk!(s, trans, C, t, Rp, Sp, na, nn, Dp, Lp, Stgt, Dval, Lval, j0, Val(3), zi)
    return
end

# columns j₀:j₀+W-1 of a front with nn columns (those after them are finished, in C)
@generated function down_chunk!(s, trans, C, t, Rp, Sp, na, nn, Dp, Lp, Stgt, Dval, Lval, j0, ::Val{W}, zi::Bool) where {W}
    xs = [Symbol(:x, j) for j in 1:W]
    loads = [:($(xs[j]) = zi ? szero(s, eltype(C), trans) : C[t, Rp + j0 + $(j - 2)]) for j in 1:W]
    upd = [:($(xs[j]) = smuladd(s, c, Lval[Lp + (j0 + $(j - 2)) * na + r - 1], $(xs[j]), Val(:N), trans)) for j in 1:W]
    after = [:($(xs[j]) = smuladd(s, c, Dval[Dp + (j0 + $(j - 2)) * nn + k - 1], $(xs[j]), Val(:N), trans)) for j in 1:W]
    solve = Expr[]

    for j in W:-1:1, k in (j + 1):W
        push!(solve, :($(xs[j]) = smuladd(s, $(xs[k]), Dval[Dp + (j0 + $(j - 2)) * nn + j0 + $(k - 2)], $(xs[j]), Val(:N), trans)))
    end

    stores = [:(C[t, Rp + j0 + $(j - 2)] = $(xs[j])) for j in 1:W]

    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(loads...)
            r = 1

            while r + 3 <= na
                c1 = C[t, Stgt[Sp + r - 1]]; c2 = C[t, Stgt[Sp + r]]; c3 = C[t, Stgt[Sp + r + 1]]; c4 = C[t, Stgt[Sp + r + 2]]
                c = c1; $(upd...); r += 1
                c = c2; $(upd...); r += 1
                c = c3; $(upd...); r += 1
                c = c4; $(upd...); r += 1
            end

            while r <= na
                c = C[t, Stgt[Sp + r - 1]]
                $(upd...)
                r += 1
            end

            for k in (j0 + $W):nn           # the finished columns after the chunk
                c = C[t, Rp + k - 1]
                $(after...)
            end

            $(solve...)
            $(stores...)
        end

        return
    end
end

function downward_kernel_reg!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, order, off::Int,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    f = @inbounds order[off + blockIdx().x]
    t = threadIdx().x + (blockIdx().y - 1) * blockDim().x

    while t <= size(C, 1)
        downward_front_reg!(s, trans, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval)
        t += blockDim().x * gridDim().y
    end

    return
end

#
#   C[i, j] ← C[i, j] ⊕ m, atomically
#
# C[i, j] ← C[i, j] ⊕ m atomically. When ⊕ is min, max or + on a machine type, this is one native
# reduction (RED: fire and forget, no returned value); otherwise a compare-and-swap loop.
@inline atomic_splus!(s::AbstractSemiring, trans::Val, C::CuDeviceMatrix{T}, i, j, m::T) where {T} =
    atomic_splus!(atomic_kind(s, trans, T), s, trans, C, i, j, m)
@inline atomic_splus!(s::AbstractSemiring, trans::Val, C::ColMapped{T}, i, j, m::T) where {T} =
    atomic_splus!(atomic_kind(s, trans, T), s, trans, C.p, i, (@inbounds C.cm[j]), m)

atomic_kind(s, trans, T) = Val(:cas)
atomic_kind(::CPU.MinPlus, ::Val{:N}, ::Type{<:Union{Float32, Float64, Int32, Int64}}) = Val(:min)
atomic_kind(::CPU.DualQuantale{CPU.MinPlus}, ::Val{:N}, ::Type{<:Union{Float32, Float64, Int32, Int64}}) = Val(:max)   # MaxPlus
atomic_kind(::CPU.DualQuantale{CPU.MinMax}, ::Val{:N}, ::Type{<:Union{Float32, Float64, Int32, Int64}}) = Val(:max)   # MaxMin
atomic_kind(::CPU.PlusProd, ::Val{:N}, ::Type{<:Union{Float32, Float64}}) = Val(:add)

@inline atomic_ptr(::Type{U}, C, i, j) where {U} = reinterpret(Core.LLVMPtr{U, CUDA.AS.Global}, pointer(C, i + (j - 1) * size(C, 1)))

@inline function atomic_splus!(::Val{:add}, s, trans, C::CuDeviceMatrix{T}, i, j, m::T) where {T}
    CUDA.atomic_add!(atomic_ptr(T, C, i, j), m)
    return
end

# integers: native min / max. IEEE floats: their order is the signed-integer order of the bits for
# x ≥ 0 and the reversed unsigned order for x < 0, so min is a signed min when m ≥ 0 and an unsigned
# max when m < 0 (and the converse for max). Exact; a NaN m is dropped, as by minnum / maxnum.
for (K, op, rop) in ((:min, :atomic_min!, :atomic_max!), (:max, :atomic_max!, :atomic_min!))
    @eval @inline function atomic_splus!(::Val{$(QuoteNode(K))}, s, trans, C::CuDeviceMatrix{T}, i, j, m::T) where {T}
        if T <: Integer
            CUDA.$op(atomic_ptr(T, C, i, j), m)
        elseif !isnan(m)
            S = sizeof(T) == 4 ? Int32 : Int64
            U = sizeof(T) == 4 ? UInt32 : UInt64

            if signbit(m)
                CUDA.$rop(atomic_ptr(U, C, i, j), reinterpret(U, m))
            else
                CUDA.$op(atomic_ptr(S, C, i, j), reinterpret(S, m))
            end
        end

        return
    end
end

@inline function atomic_splus!(::Val{:cas}, s::AbstractSemiring, trans::Val, C::CuDeviceMatrix{T}, i, j, m::T) where {T}
    U = sizeof(T) == 4 ? UInt32 : UInt64
    ptr = reinterpret(Core.LLVMPtr{U, CUDA.AS.Global}, pointer(C, i + (j - 1) * size(C, 1)))
    old = @inbounds C[i, j]

    while true
        new = splus(s, old, m, trans)

        if new === old
            return
        end

        prev = CUDA.atomic_cas!(ptr, reinterpret(U, old), reinterpret(U, new))

        if prev == reinterpret(U, old)
            return
        end

        old = reinterpret(T, prev)
    end
end

