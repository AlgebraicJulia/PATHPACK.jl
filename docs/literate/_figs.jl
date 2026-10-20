# Figures for the tutorials, drawn from the values each tutorial computes, as inline SVG. Colors are classes, not
# values (k0..k31 along the site's distance ramp, kg grey, ka accent), so tutorials.css can give each class a light and
# a dark color and the figures follow the site's theme. Tutorials include this file in a hidden line; it is not part
# of the code a reader runs.
module Figs

export Fig, figure, panel, network, gridmap, matrixfig, gantt, bars, scatter, legend, checkgrid

struct Fig
    html::String
end
Base.show(io::IO, ::MIME"text/html", f::Fig) = print(io, f.html)

esc(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
r1(x) = round(x; digits = 1)
num(x) = (y = r1(x); isinteger(y) ? string(Int(y)) : string(y))

"""The ramp class of `x` in [0, 1]; grey for NaN or Inf."""
cls(x) = isfinite(x) ? "k$(clamp(round(Int, 31x), 0, 31))" : "kg"

"""Scale `v` to [0, 1] between its finite extremes (or `lo`, `hi`); `log = true` on a log scale."""
function unit(v; lo = nothing, hi = nothing, log = false)
    tr(x) = log ? Base.log(max(x, floatmin(Float64))) : float(x)
    w = tr.(v); f = filter(isfinite, w)
    a = lo === nothing ? minimum(f) : tr(lo)
    b = hi === nothing ? maximum(f) : tr(hi)
    return [isfinite(x) ? clamp((x - a) / max(b - a, eps()), 0, 1) : NaN for x in w]
end

"""A panel: a title and an SVG body in a viewBox of `w` × `h`."""
panel(title, w, h, body; sub = "") =
    """<div class="pp-fig-panel"><div class="pp-fig-title"><b>$(esc(title))</b>$(isempty(sub) ? "" : " <span>$(esc(sub))</span>")</div><svg viewBox="0 0 $w $h" role="img" aria-label="$(esc(title))">$body</svg></div>"""

panel(title, b::Tuple; sub = "") = panel(title, b[2], b[3], b[1]; sub)

"""A legend of the ramp from `lo` to `hi`."""
legend(lo, hi; label = "") =
    """<div class="pp-fig-legend"><span>$(esc(lo))</span><i></i><span>$(esc(hi))</span>$(isempty(label) ? "" : "<em>$(esc(label))</em>")</div>"""

"""A figure: panels side by side, then a legend and a caption."""
figure(panels...; caption = "", legend = "", cols = length(panels)) =
    Fig("""<figure class="pp-fig"><div class="pp-fig-row" style="--cols:$cols">$(join(panels))</div>$legend$(isempty(caption) ? "" : "<figcaption>$(esc(caption))</figcaption>")</figure>""")

# many marks of one class as one path: dots (round caps on zero-length segments) or segments
function dots(pts, c, r)
    isempty(pts) && return ""
    d = join(("M$(r1(x)) $(r1(y))h0" for (x, y) in pts))
    return """<path class="$c" d="$d" stroke="currentColor" stroke-width="$(r1(2r))" stroke-linecap="round" fill="none"/>"""
end
function segs(ss, c, w; op = 1)
    isempty(ss) && return ""
    d = join(("M$(r1(a)) $(r1(b))L$(r1(x)) $(r1(y))" for (a, b, x, y) in ss))
    return """<path class="$c" d="$d" stroke="currentColor" stroke-width="$w" stroke-linecap="round" fill="none" opacity="$op"/>"""
end
group(f, items) = (D = Dict{String, Vector{Any}}(); for it in items; push!(get!(D, f(it), Any[]), it); end; D)

"""Fit points into a `w` × `h` box with a margin, keeping the aspect."""
function fit(x, y, w, h; pad = 14, flip = true)
    x0, x1 = extrema(x); y0, y1 = extrema(y)
    k = min((w - 2pad) / max(x1 - x0, eps()), (h - 2pad) / max(y1 - y0, eps()))
    ox = (w - k * (x1 - x0)) / 2; oy = (h - k * (y1 - y0)) / 2
    X = ox .+ k .* (x .- x0); Y = flip ? h .- (oy .+ k .* (y .- y0)) : oy .+ k .* (y .- y0)
    return X, Y
end

"""
A node-link picture. `xy` n × 2 positions, links `src` → `dst` (each pair once is enough). `val` per node in [0, 1]
(NaN: grey); `lw` per link (stroke width); `path` a vertex sequence drawn in the text color; `mark` vertices
circled.
"""
function network(xy, src, dst; val = fill(NaN, size(xy, 1)), lw = nothing, path = Int[], mark = Int[], w = 480, h = 360, r = 3.2)
    X, Y = fit(xy[:, 1], xy[:, 2], w, h; pad = 12)
    body = IOBuffer()
    links = unique(minmax(a, b) for (a, b) in zip(src, dst) if a != b)
    if lw === nothing
        print(body, segs([(X[a], Y[a], X[b], Y[b]) for (a, b) in links], "kl", 0.8))
    else
        W = Dict(minmax(a, b) => l for (a, b, l) in zip(src, dst, lw))
        for (l, ls) in group(e -> string(W[e]), links)
            print(body, segs([(X[a], Y[a], X[b], Y[b]) for (a, b) in ls], "kl", l))
        end
    end
    if length(path) > 1
        print(body, """<polyline class="ki" points="$(join(("$(r1(X[v])),$(r1(Y[v]))" for v in path), " "))" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>""")
    end
    for (c, vs) in group(v -> cls(val[v]), 1:size(xy, 1))
        print(body, dots([(X[v], Y[v]) for v in vs], c, r))
    end
    for v in mark
        print(body, """<circle class="ki" cx="$(r1(X[v]))" cy="$(r1(Y[v]))" r="$(r1(2.4r))" fill="none" stroke="currentColor" stroke-width="2.2"/>""")
    end
    return String(take!(body)), w, h
end

"""
A grid of cells colored by `val` (a matrix of values in [0, 1], NaN grey), as runs of equal color per row. `marks`:
(row, col) cells circled; `path`: (row, col) cells joined in the text color.
"""
function gridmap(val::AbstractMatrix; w = 360, marks = Tuple{Int, Int}[], path = Tuple{Int, Int}[], levels = 32)
    m, n = size(val); c = w / n; h = r1(m * c)
    q(x) = isfinite(x) ? "k$(clamp(round(Int, (levels - 1) * x) * (31 ÷ max(levels - 1, 1)), 0, 31))" : "kg"
    body = IOBuffer()
    runs = Dict{String, Vector{NTuple{3, Int}}}()
    for i in 1:m
        j = 1
        while j <= n
            k = q(val[i, j]); j2 = j
            while j2 < n && q(val[i, j2 + 1]) == k; j2 += 1; end
            push!(get!(runs, k, NTuple{3, Int}[]), (i, j, j2)); j = j2 + 1
        end
    end
    for (k, rs) in runs
        d = join(("M$(r1((j - 1) * c)) $(r1((i - 1) * c))h$(r1((j2 - j + 1) * c + 0.4))v$(r1(c + 0.4))h-$(r1((j2 - j + 1) * c + 0.4))z" for (i, j, j2) in rs))
        print(body, """<path class="$k" d="$d" fill="currentColor"/>""")
    end
    if length(path) > 1
        print(body, """<polyline class="ki" points="$(join(("$(r1((j - 0.5) * c)),$(r1((i - 0.5) * c))" for (i, j) in path), " "))" fill="none" stroke="currentColor" stroke-width="$(r1(max(2, c / 2.2)))" stroke-linejoin="round" stroke-linecap="round"/>""")
    end
    for (i, j) in marks
        print(body, """<circle class="ki" cx="$(r1((j - 0.5) * c))" cy="$(r1((i - 0.5) * c))" r="$(r1(max(5, 1.6c)))" fill="none" stroke="currentColor" stroke-width="2.2"/>""")
    end
    return String(take!(body)), w, h
end

"""A small matrix with its numbers, cells colored by `val` in [0, 1]."""
function matrixfig(M; val = unit(M), fmt = num, cell = 46)
    m, n = size(M); w = n * cell + 26; h = m * cell + 26
    body = IOBuffer()
    for j in 1:n
        print(body, """<text class="pp-fig-ix" x="$(26 + (j - 0.5) * cell)" y="16" text-anchor="middle">$j</text>""")
    end
    for i in 1:m
        print(body, """<text class="pp-fig-ix" x="13" y="$(26 + (i - 0.5) * cell + 4)" text-anchor="middle">$i</text>""")
        for j in 1:n
            x = 26 + (j - 1) * cell; y = 26 + (i - 1) * cell; v = val[i, j]
            print(body, """<rect class="$(cls(v))" x="$(x + 1.5)" y="$(y + 1.5)" width="$(cell - 3)" height="$(cell - 3)" rx="6" fill="currentColor" opacity="$(isfinite(v) ? 0.9 : 0.5)"/>""")
            print(body, """<text class="pp-fig-num" x="$(x + cell / 2)" y="$(y + cell / 2 + 5)" text-anchor="middle">$(isfinite(M[i, j]) ? fmt(M[i, j]) : M[i, j] > 0 ? "∞" : "−∞")</text>""")
        end
    end
    return String(take!(body)), w, h
end

"""A schedule: one bar per task from `start` for `dur`, critical tasks in the accent color, slack as a faint tail."""
function gantt(names, start, dur, critical; slack = zeros(length(names)), w = 640, row = 24)
    keep = [i for i in eachindex(names) if dur[i] > 0]
    total = maximum(start[keep] .+ dur[keep] .+ slack[keep]); lab = 108
    k = (w - lab - 16) / total; h = length(keep) * row + 30
    body = IOBuffer()
    for t in 0:5:total
        x = lab + t * k
        print(body, """<line class="kl" x1="$(r1(x))" x2="$(r1(x))" y1="4" y2="$(h - 22)" stroke="currentColor" stroke-width="0.8"/><text class="pp-fig-ix" x="$(r1(x))" y="$(h - 6)" text-anchor="middle">$(num(t))</text>""")
    end
    for (r, i) in enumerate(keep)
        y = 4 + (r - 1) * row
        print(body, """<text class="pp-fig-lab" x="$(lab - 8)" y="$(y + row / 2 + 4)" text-anchor="end">$(esc(names[i]))</text>""")
        if slack[i] > 0
            print(body, """<rect class="kg" x="$(r1(lab + (start[i] + dur[i]) * k))" y="$(y + 7)" width="$(r1(slack[i] * k))" height="$(row - 14)" rx="3" fill="currentColor" opacity="0.6"/>""")
        end
        print(body, """<rect class="$(critical[i] ? "ka" : "k8")" x="$(r1(lab + start[i] * k))" y="$(y + 3)" width="$(r1(dur[i] * k - 2))" height="$(row - 6)" rx="5" fill="currentColor"/>""")
    end
    return String(take!(body)), w, h
end

"""Horizontal bars, labelled, with their values printed by `fmt`; `hi` marks bars drawn in the accent color."""
function bars(labels, vals; fmt = num, hi = falses(length(vals)), w = 560, row = 30, val = nothing)
    lab = min(250, 8 + 6.6 * maximum(length, string.(labels))); top = maximum(vals); k = (w - lab - 80) / top; h = length(vals) * row + 6
    body = IOBuffer()
    for (r, (l, v)) in enumerate(zip(labels, vals))
        y = 3 + (r - 1) * row
        c = hi[r] ? "ka" : val === nothing ? "k6" : cls(val[r])
        print(body, """<text class="pp-fig-lab" x="$(lab - 10)" y="$(y + row / 2 + 4)" text-anchor="end">$(esc(l))</text>""")
        print(body, """<rect class="$c" x="$lab" y="$(y + 4)" width="$(r1(max(2, v * k)))" height="$(row - 8)" rx="5" fill="currentColor"/>""")
        print(body, """<text class="pp-fig-num2" x="$(r1(lab + max(2, v * k) + 8))" y="$(y + row / 2 + 4)">$(esc(fmt(v)))</text>""")
    end
    return String(take!(body)), w, h
end

"""
A yes/no grid with row and column labels: `M[i, j]` true is a filled cell, in the accent color where `hi[i, j]`.
"""
function checkgrid(rows, cols, M; hi = falses(size(M)), cell = 34)
    lab = min(320, 12 + 7.0 * maximum(length, string.(rows))); top = 30
    w = lab + length(cols) * cell + 4; h = top + length(rows) * cell + 4
    body = IOBuffer()
    for (j, c) in enumerate(cols)
        print(body, """<text class="pp-fig-lab" x="$(r1(lab + (j - 0.5) * cell))" y="20" text-anchor="middle">$(esc(c))</text>""")
    end
    for (i, r) in enumerate(rows)
        y = top + (i - 1) * cell
        print(body, """<text class="pp-fig-lab" x="$(lab - 10)" y="$(r1(y + cell / 2 + 4))" text-anchor="end">$(esc(r))</text>""")
        for j in eachindex(cols)
            x = lab + (j - 1) * cell
            c = M[i, j] ? (hi[i, j] ? "ka" : "k6") : "kg"
            print(body, """<rect class="$c" x="$(r1(x + 3))" y="$(r1(y + 3))" width="$(cell - 6)" height="$(cell - 6)" rx="6" fill="currentColor" opacity="$(M[i, j] ? 0.95 : 0.35)"/>""")
        end
    end
    return String(take!(body)), w, h
end

"""Points colored by `val` in [0, 1]."""
function scatter(x, y, val; w = 420, h = 300, r = 2.2)
    X, Y = fit(x, y, w, h; pad = 10)
    body = IOBuffer()
    for (c, vs) in group(i -> cls(val[i]), eachindex(x))
        print(body, dots([(X[i], Y[i]) for i in vs], c, r))
    end
    return String(take!(body)), w, h
end

end
using .Figs
