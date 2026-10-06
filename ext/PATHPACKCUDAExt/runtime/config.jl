# ===== configuration =====
#
# Every tunable choice of the solver lives in one immutable GPUConfig, held in a ScopedValue: the
# defaults apply everywhere, and `with_config(f; kw...)` changes them only for the code that runs
# inside `f` (and the tasks it starts), so concurrent callers, threads and GPUs never see each
# other's settings. None of these change results beyond rounding of non-idempotent semirings
# (plus-times); for min-plus, max-plus and max-min every setting gives bit-identical output.
#
# The defaults are the measured best (or within a few percent of it) on an RTX 5060 Laptop, L4,
# RTX PRO 6000 and B200 (bench/portable.jl, bench/gemm_shapes.jl).

using Base.ScopedValues: ScopedValue, with

"""
    GPUConfig(; kw...)

Settings of the GPU solver (see `with_config`).

Defaults can be set per process with environment variables `SEMIRINGGPU_<SETTING>` (for example
`SEMIRINGGPU_MERGE=1`), read when the module loads.

Dense GEMM
- `gemm_tune = true`: pick the GEMM kernel per GPU and shape class by timing candidates once, and
  cache the choice on disk (`SEMIRINGGPU_TUNE=off` in the environment turns the default off).
- `gemm_kernel = 0`: 0 lets the solver choose; 2, 4, 6, 7 or 8 forces that kernel version (testing).

Solve and closure
- `merge = 128`, `merge_alpha = 0.5`: merge chains of fronts into fronts of up to `merge` pivots for
  the solve, while the semiring-zero padding stays below `merge_alpha` of the merged block (1 = off).
- `skip_fill = true`: do not fill the n × n result before the solve; rows are zeroed only where needed.
- `layered_min_rows = 4096`: from this many right-hand sides on, the L sweep below the top of the tree
  runs as one layered launch per layer (fewer rows use level-by-level launches).
- `layer_size = 0`: fronts per region of the layered sweep (0: about √(2 · fronts)).
- `layer_cache = true`: the layered sweep keeps each row's recent values in shared memory (slots,
  placed by a host-side plan; layered.jl); false runs the plain layered walk.
- `layer_slots = 0`: slots per row of that cache (0: from the device's shared memory, at most 64).
- `layer_cache_min = 300000`: that cache only when rows × (work per row) / (fronts) below the top of the
  tree reaches this: its plan costs the host ~0.3–0.5 µs per front, and it saves the GPU ~0.25 ms per 10⁹
  multiply-adds on meshes (B200), little on social and power-law graphs, whose fronts are small.
- `plan_overlap = true`: build that host-side plan on another thread while the GPU sweeps the top of
  the tree (false: build it when the sweep reaches it, with the GPU idle).
- `top_rows = true`: in the closure, each front of the top of the tree runs its U step on the rows whose
  source lies in its subtree only, and its dense L step on the others with the separator part only
  (toprows.jl; the other rows are the semiring zero there). false: every front on every row.

Numeric factorization
- `factor_merge = 128`: merge chains of fronts at the top of the tree for the GPU factorization (1 = off).
- `factor_merge_min = 16`: that merge only for a top of at least this many fronts (the fronts that reach
  the GPU's front size and their ancestors); on a smaller top it costs more host time than it saves.
- `factor_balance = 8`: the CPU factors the bottom of the tree in subtrees, one thread each. Subtrees with
  more than 1/(b × threads) of the work join the GPU top, for the b in 1, 2, 4, … ≤ `factor_balance`
  that saves the most CPU time net of the top levels it adds, if any (0 = off: a graph whose fronts are
  all small then has no top, and one subtree, so the whole tree is factored on one thread).
- `factor_level_work = 200000`: the CPU work (multiply-adds, a front counted as at least 256) that a
  level added to the GPU top must save: a level costs ~50–70 µs of launches, a multiply-add ~0.3 ns.
- `fused_front = true`: LU and both triangular closures of a ≤ 64-pivot diagonal block in one kernel.
- `direct_assembly = true`: add children's updates straight into the factor blocks (no front matrix).
- `factor_gpu = true`: the top of the tree is factored on the GPU (hybrid). false: the whole factorization
  on the CPU threads (the bottom subtrees in parallel, then the top fronts with multithreaded dense
  kernels), then the factor is uploaded; the GPU computes only the inverse and the solve. The same
  factor up to the rounding of non-idempotent semirings.

Symbolic factorization
- `ordering = "auto"`: the fill-reducing ordering (any order gives the same distances; only fill and speed
  change). "auto": CliqueTrees.AutoOrder — per graph, HubAMF on graphs with hubs (maximum degree > 2√n),
  BFSND (parallel nested dissection, halo AMF leaves) on lattice-like graphs, AMF otherwise; "amf": AMF;
  "hub": HubAMF; "bfsnd": BFSND everywhere; "amd", "metis": SuiteSparse AMD and METIS NodeND (need AMD.jl /
  Metis.jl loaded); "nd-dense": partial nested dissection with dense leaves of ≤ 150 vertices (ROME-style).

The `alg` keyword of `apsp_gpu`, `apsp_gpu!` and `apsp_plan` (an elimination algorithm or permutation for
CliqueTrees, e.g. `BestFill(AMF(), METIS())`) replaces `ordering` for that call.
"""
Base.@kwdef struct GPUConfig
    gemm_tune::Bool = get(ENV, "SEMIRINGGPU_TUNE", "auto") != "off"
    gemm_kernel::Int = 0
    merge::Int = 128
    merge_alpha::Float64 = 0.5
    skip_fill::Bool = true
    layered_min_rows::Int = 4096
    layer_size::Int = 0
    layer_cache::Bool = true
    layer_slots::Int = 0
    layer_cache_min::Int = 300000
    plan_overlap::Bool = true
    top_rows::Bool = true
    factor_merge::Int = 128
    factor_merge_min::Int = 16
    factor_balance::Int = 8
    factor_level_work::Int = 200000
    fused_front::Bool = true
    direct_assembly::Bool = true
    factor_gpu::Bool = true
    ordering::String = "auto"
end

# a copy of c with some fields replaced
function GPUConfig(c::GPUConfig; kw...)
    for k in keys(kw)
        hasfield(GPUConfig, k) || throw(ArgumentError("GPUConfig has no setting `$k`; settings: $(join(fieldnames(GPUConfig), ", "))"))
    end
    return GPUConfig(; (f => get(kw, f, getfield(c, f)) for f in fieldnames(GPUConfig))...)
end

function check(c::GPUConfig)
    c.gemm_kernel in (0, 2, 4, 6, 7, 8) || throw(ArgumentError("gemm_kernel must be 0 (automatic), 2, 4, 6, 7 or 8, not $(c.gemm_kernel)"))
    c.merge >= 1 && c.factor_merge >= 1 || throw(ArgumentError("merge widths must be at least 1"))
    0 <= c.merge_alpha || throw(ArgumentError("merge_alpha must be nonnegative"))
    c.layered_min_rows >= 1 && c.layer_size >= 0 || throw(ArgumentError("layered_min_rows must be ≥ 1 and layer_size ≥ 0"))
    c.layer_slots >= 0 && c.layer_cache_min >= 0 || throw(ArgumentError("layer_slots and layer_cache_min must be ≥ 0"))
    c.factor_merge_min >= 0 && c.factor_balance >= 0 && c.factor_level_work >= 0 ||
        throw(ArgumentError("factor_merge_min, factor_balance and factor_level_work must be ≥ 0"))
    c.ordering in ORDERINGS || throw(ArgumentError("ordering must be one of $(join(ORDERINGS, ", ")), not $(repr(c.ordering))"))
    return c
end

# the `alg` of the call (apsp_gpu(A; alg)): an elimination algorithm or permutation, or nothing (`ordering`)
const ELIMINATION_ALG = ScopedValue{Any}(nothing)
# (an algorithm: CPU.ssymbolic orders each strongly connected component on its own, so a permutation of the
# whole graph is not an `alg` it can take)
function with_alg(f, alg)
    isnothing(alg) && return f()
    alg isa CliqueTrees.EliminationAlgorithm ||
        throw(ArgumentError("alg must be an elimination algorithm of CliqueTrees (e.g. AMF(), BestFill(AMF(), METIS())), not a $(typeof(alg))"))
    return with(f, ELIMINATION_ALG => alg)
end

const ORDERINGS = ("auto", "amf", "hub", "bfsnd", "amd", "metis", "nd-dense")

"The fill-reducing ordering of the symbolic factorization (see `GPUConfig`'s `ordering`)."
function elimination_algorithm(c::GPUConfig = config())
    alg = ELIMINATION_ALG[]
    isnothing(alg) || return alg
    o = c.ordering
    # (AutoOrder, HubAMF and BFSND are in CliqueTrees' itay/gpu branch; without them, "auto" is AMF)
    o == "auto" && return isdefined(CliqueTrees, :AutoOrder) ? CliqueTrees.AutoOrder() : CliqueTrees.AMF()
    o == "amf" && return CliqueTrees.AMF()
    o == "hub" && return CliqueTrees.HubAMF()
    o == "bfsnd" && return CliqueTrees.BFSND()
    o == "amd" && return CliqueTrees.AMD()
    o == "metis" && return CliqueTrees.METIS()
    return CliqueTrees.BFSND(; levels = 40, minsize = 150, leaf = CliqueTrees.Natural())
end

# process-wide defaults may be set by environment variables SEMIRINGGPU_<SETTING> (e.g.
# SEMIRINGGPU_MERGE=1, SEMIRINGGPU_SKIP_FILL=false), read once when the module loads
function settings_from_env(env = ENV)
    kw = Pair{Symbol, Any}[]

    for f in fieldnames(GPUConfig)
        v = get(env, "SEMIRINGGPU_" * uppercase(String(f)), nothing)
        isnothing(v) && continue
        T = fieldtype(GPUConfig, f)
        push!(kw, f => (T === String ? String(v) : T === Bool ? parse(Bool, v) : parse(T, v)))
    end

    return (; kw...)
end

# the process defaults, read from the environment when the module is loaded (init_config!, called by
# __init__: a precompiled module must see the session's environment, not that of precompilation);
# CONFIG holds the settings changed by with_config, or nothing outside any with_config
const DEFAULT_CONFIG = Ref{GPUConfig}()
const CONFIG = ScopedValue{Union{Nothing, GPUConfig}}(nothing)

init_config!() = (DEFAULT_CONFIG[] = check(GPUConfig(GPUConfig(); settings_from_env()...)); nothing)
init_config!()

"The settings in effect (see `with_config`)."
config() = something(CONFIG[], DEFAULT_CONFIG[])

"""
    with_config(f; kw...)

Run `f()` with some settings of `GPUConfig` changed, e.g. `with_config(() -> closure_gpu(G); merge = 1)`.
The change is scoped: other tasks, threads and GPUs keep their own settings.
"""
with_config(f; kw...) = with(f, CONFIG => check(GPUConfig(config(); kw...)))

# internal: false while work is issued on several streams at once, where the GEMM autotuner cannot
# time candidates cleanly (they then use the heuristic choice)
const TUNING = ScopedValue(true)
without_tuning(f) = with(f, TUNING => false)

# internal: a Dict{Symbol, Float64} to time the factorization kernels by phase (synchronizes), or nothing
const FTIMER = ScopedValue{Any}(nothing)

# ===== step timer =====
#
# with_steps(f) runs f() and returns (f(), steps): the wall time of every step of the call that is marked
# with @step, by name, nested steps under their parent ("plan/top merge"). Each step synchronizes the
# device before and after, so GPU work is charged to the step that issued it (and work that would have
# overlapped across steps no longer does). Off (one null test per step) outside with_steps.

mutable struct StepTimer
    times::Dict{String, Float64}
    gc::Dict{String, Float64}           # of which garbage collection
    bytes::Dict{String, Int}            # host memory allocated
    counts::Dict{String, Int}
    order::Vector{String}               # in order of first start (parents before their steps)
    stack::Vector{Tuple{String, UInt64, UInt64, Int}}
end

StepTimer() = StepTimer(Dict{String, Float64}(), Dict{String, Float64}(), Dict{String, Int}(), Dict{String, Int}(), String[],
    Tuple{String, UInt64, UInt64, Int}[])

const STEPS = ScopedValue{Union{Nothing, StepTimer}}(nothing)

function step_begin!(tm::StepTimer, name::AbstractString)
    CUDA.device_synchronize()
    path = join((first.(tm.stack)..., name), "/")
    haskey(tm.times, path) || (push!(tm.order, path); tm.times[path] = 0.0)
    push!(tm.stack, (String(name), time_ns(), Base.gc_time_ns(), Base.gc_bytes()))
    return
end

function step_end!(tm::StepTimer)
    CUDA.device_synchronize()
    t1 = time_ns(); g1 = Base.gc_time_ns(); b1 = Base.gc_bytes()
    name, t0, g0, b0 = pop!(tm.stack)
    path = join((first.(tm.stack)..., name), "/")
    tm.times[path] += (t1 - t0) / 1e9
    tm.gc[path] = get(tm.gc, path, 0.0) + (g1 - g0) / 1e9
    tm.bytes[path] = get(tm.bytes, path, 0) + (b1 - b0)
    tm.counts[path] = get(tm.counts, path, 0) + 1
    return
end

# t seconds as a step `name` of the current step, when a StepTimer is active (a part timed by its caller)
function step_add!(name::AbstractString, t::Real)
    tm = STEPS[]
    isnothing(tm) && return
    path = join((first.(tm.stack)..., name), "/")
    haskey(tm.times, path) || (push!(tm.order, path); tm.times[path] = 0.0)
    tm.times[path] += t
    tm.counts[path] = get(tm.counts, path, 0) + 1
    return
end

# `@step name expr`: expr, timed as step `name` when a StepTimer is active (no closure: assignments in
# expr, e.g. a begin … end block, stay in the caller's scope)
macro step(name, ex)
    return quote
        local tm = STEPS[]
        local on = tm !== nothing && !CUDA.is_capturing()      # (no synchronization while a graph is recorded)
        on && step_begin!(tm, $(esc(name)))
        local tr = TRACE[]; local t0 = tr === nothing ? UInt64(0) : time_ns()
        local r = $(esc(ex))
        tr === nothing || trace!(tr, $(esc(name)), t0)
        on && step_end!(tm)
        r
    end
end

# `with_trace(f)`: f(), with every step's host interval (start, end, thread) recorded as it runs, without
# synchronizing (unlike the step timer): the timeline of a real call, its overlaps and critical path
const TRACE = Ref{Union{Nothing, Vector{Tuple{String, Int, UInt64, UInt64}}}}(nothing)
const TRACE_LOCK = ReentrantLock()

trace!(tr, name, t0) = (t1 = time_ns(); lock(() -> push!(tr, (String(name), Threads.threadid(), t0, t1)), TRACE_LOCK))

function with_trace(f)
    tr = Tuple{String, Int, UInt64, UInt64}[]
    TRACE[] = tr
    t0 = time_ns()
    r = try f() finally TRACE[] = nothing end
    return r, [(name, tid, (a - t0) / 1e6, (b - t0) / 1e6) for (name, tid, a, b) in tr], (time_ns() - t0) / 1e6
end

with_steps(f) = (tm = StepTimer(); r = with(f, STEPS => tm); (r, tm))

"""
    D, t = with_phases(() -> apsp_gpu(A))

Run `f()` with the step timer and return its result and the time (seconds) of each phase of the call:

- `symbolic`: ordering and symbolic factorization (CPU)
- `numeric_setup`: the factor's storage, the entries of A copied in, the factorization's plan (CPU)
- `numeric_cpu`, `numeric_gpu`: the numeric factorization's CPU part (the bottom of the tree, or all of
  it with `factor_gpu = false`) and GPU part (the top of the tree)
- `transfer`: host ↔ device copies of the factorization and of the result (`output = :host`)
- `solve_setup`: the solve's structure and factor on the device, the dense operators of large fronts,
  the result's allocation
- `inverse`: the closure's U* phase (rows of U*, each source's path and the top fronts)
- `solve`: the closure's L* phase
- `relabel`: the result's columns into the labels of A (when the closure did not write them in place)
- `other`, `total`; `steps`: every step's time, by name (as `with_steps`)

Each step synchronizes the device and the steps run one after the other (the overlaps of an untimed
call are off), so the total exceeds the time of an untimed call.
"""
function with_phases(f)
    r, tm = with_steps(f)
    return r, phases(tm)
end

function phases(tm::StepTimer)
    t = tm.times
    get0(k) = get(t, k, 0.0)
    acc = Dict{Symbol, Float64}(k => 0.0 for k in (:symbolic, :numeric_setup, :numeric_cpu, :numeric_gpu, :transfer, :solve_setup,
        :inverse, :solve, :relabel, :other))
    kids(p) = [k for k in keys(t) if startswith(k, p * "/") && !occursin('/', k[(length(p) + 2):end])]
    for (k, v) in t
        occursin('/', k) && continue
        if k == "factorize"
            rest = v
            for c in kids(k)
                w = t[c]; rest -= w; name = c[(length(k) + 2):end]
                acc[occursin("GPU", name) ? :numeric_gpu : name == "upload" ? :transfer : :numeric_cpu] += w
            end
            acc[:numeric_cpu] += max(rest, 0.0)
        elseif k == "closure"
            rest = v
            for c in kids(k)
                w = t[c]; rest -= w; name = c[(length(k) + 2):end]
                acc[startswith(name, "U_") || startswith(name, "fill") ? :inverse : :solve] += w
            end
            acc[:solve] += max(rest, 0.0)
        else
            phase = k == "symbolic" ? :symbolic :
                k in ("copy entries", "factor storage", "plan") ? :numeric_setup :
                k in ("to host",) ? :transfer :
                k in ("solve setup", "operators", "memory check", "allocate", "host matrix", "refresh factor") ? :solve_setup :
                k == "relabel columns" ? :relabel : :other
            acc[phase] += v
        end
    end
    total = sum(get0(k) for k in keys(t) if !occursin('/', k); init = 0.0)
    return (; (k => acc[k] for k in (:symbolic, :numeric_setup, :numeric_cpu, :numeric_gpu, :transfer, :solve_setup, :inverse, :solve,
        :relabel, :other))..., total, steps = copy(t))
end

# the steps as an indented tree: ms, share of the total, calls, and the time of each parent not in a child
function print_steps(io::IO, tm::StepTimer; total = sum(v for (k, v) in tm.times if !occursin('/', k); init = 0.0))
    for path in tm.order
        depth = count(==('/'), path)
        t = tm.times[path]
        kids = [k for k in tm.order if startswith(k, path * "/") && count(==('/'), k) == depth + 1]
        rest = isempty(kids) ? "" : string("   (other ", round(1e3 * (t - sum(tm.times[k] for k in kids)); digits = 1), " ms)")
        calls = tm.counts[path] > 1 ? string("  ×", tm.counts[path]) : ""
        gc = get(tm.gc, path, 0.0) >= 5e-4 ? string("  [GC ", round(1e3 * tm.gc[path]; digits = 1), " ms]") : ""
        mb = get(tm.bytes, path, 0) / 2^20
        gc *= mb >= 1 ? string("  {", round(mb; digits = 1), " MiB}") : ""
        println(io, rpad("  "^depth * last(split(path, '/')), 34), lpad(round(1e3t; digits = 1), 9), " ms ",
            lpad(round(100t / total; digits = 1), 5), "%", calls, gc, rest)
    end
end

print_steps(tm::StepTimer; kw...) = print_steps(stdout, tm; kw...)

# ===== input checks =====

"""
    check_semiring(s, T)

Throw an `ArgumentError` when the semiring `s` cannot be computed exactly with element type `T` by
the GPU kernels: the semiring zero must stay zero under ⊗ with finite values (else, for example,
an integer infinity overflows when a weight is added to it) and be the identity of ⊕.
"""
function check_semiring(s::AbstractSemiring, ::Type{T}) where {T}
    z = szero(s, T, Val(:N)); o = sone(s, T, Val(:N))
    samples = T <: Integer ? T[0, 1, 2, 100, typemax(T) >> 4] : T[0, 1, 2, 100]

    for w in samples
        x = CPU.sprod(s, z, w, Val(:N), Val(:N))
        y = CPU.sprod(s, w, z, Val(:N), Val(:N))

        if !isequal(x, z) || !isequal(y, z)
            throw(ArgumentError("$(nameof(typeof(s))) with $T: zero ⊗ $w = $x is not the semiring zero $z " *
                (T <: Integer ? "(the integer infinity overflows). Use Float32 or Float64 weights, or an integer semiring whose zero absorbs under ⊗." : "")))
        end

        isequal(splus(s, z, w, Val(:N)), w) || throw(ArgumentError("$(nameof(typeof(s))) with $T: zero ⊕ $w ≠ $w"))
    end

    isequal(CPU.sprod(s, o, T(1), Val(:N), Val(:N)), T(1)) || throw(ArgumentError("$(nameof(typeof(s))) with $T: one ⊗ 1 ≠ 1"))
    return nothing
end
