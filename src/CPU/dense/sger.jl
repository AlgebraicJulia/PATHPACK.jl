# ===== sger_kern! =====

function sger_kern!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, pc::Ptr{T}, ldc::Integer, px::Ptr{T}, y::AbstractVector, nr::Integer, nc::Integer) where {T}
    Z = sizeof(T)

    @inbounds for c in oneto(nc)
        saxpy_kern!(s, tA, tB, side, pc + (c - 1) * ldc * Z, px, y[c], nr)
    end

    return
end

# ===== sger! =====

function sger!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, x::AbstractVector{T}, y::AbstractVector, M::AbstractMatrix{T}; nt::Integer = nthreads()) where {T}
    nr = length(x)
    nc = length(y)

    if nt <= 1 || nc <= SGEMX_LEAF || nr * nc < SGEMX_LEAF * max(nr, nc)
        sger_st!(s, tA, tB, side, x, y, M)
    else
        sger_mt!(s, tA, tB, side, x, y, M, nt)
    end

    return M
end

function sger_st!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, x::AbstractVector{T}, y::AbstractVector, M::AbstractMatrix{T}) where {T}
    @assert size(M, 1) == length(x)
    @assert size(M, 2) == length(y)

    @preserve x M sger_kern!(s, tA, tB, side, pointer(M), stride(M, 2), pointer(x), y, length(x), length(y))

    return M
end

function sger_mt!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, x::AbstractVector{T}, y::AbstractVector, M::AbstractMatrix{T}, nt::Integer) where {T}
    nr = length(x)
    nc = length(y)

    if nt <= 1 || nc <= SGEMX_LEAF || nr * nc < SGEMX_LEAF * max(nr, nc)
        sger_st!(s, tA, tB, side, x, y, M)
    else
        tasks = FVector{Task}(undef, nt - 1)

        for t in 1:nt - 1
            strt = fld(nc * (t - 1), nt) + 1
            stop = fld(nc * t, nt)
            yv = view(y, strt:stop)
            Mv = view(M, :, strt:stop)
            tasks[t] = @spawn sger_st!(s, tA, tB, side, x, $yv, $Mv)
        end

        strt = fld(nc * (nt - 1), nt) + 1
        yv = view(y, strt:nc)
        Mv = view(M, :, strt:nc)
        sger_st!(s, tA, tB, side, x, yv, Mv)

        for t in tasks
            wait(t)
        end
    end

    return M
end
