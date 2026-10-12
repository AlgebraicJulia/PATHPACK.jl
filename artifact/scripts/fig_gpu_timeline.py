#!/usr/bin/env python3
"""The GPU timeline figure: one untimed apsp_gpu call on fe_tooth on one B200, step by step, every bar to scale.

Usage: fig_gpu_timeline.py [GPU_DATA_DIR] [OUT_TEX] [COMMIT]
       (defaults: data/gpu, build/figures/fig-gpu-timeline.tex, $GPU_COMMIT, else lib-g)

Every number drawn comes from GPU_DATA_DIR/gpu_timeline_trace.csv (the graph's n, and the numbers of the caption of
earlier versions, from gpu_apsp.csv); none is typed by hand.

The call. gpu_timeline_trace.csv is written by the trace job on one B200 at COMMIT (measure.jl and trace.sbatch in
bench/timeline/measure2.jl, target gpu_timeline): apsp_gpu!(D, A) as gpu_apsp.jl's
code ours-auto calls it (output preallocated, julia -t 8,0, cold calls), with no step timer, so nothing in the call
synchronizes. Set "events": the same call with CUDA events recorded on its stream between its steps (apsp_gpu!'s own
code re-issued step by step with the library's functions; results bitwise equal to the plain call's), which give the
GPU's steps. We draw the events call whose total is the median (the lower median for an even count). Host steps are
PATHPACK's trace hook (each @step's start and end on the host, as it ran); all host tasks share the 8 Julia threads.

Bars (start and end as traced, ms from the call's start), one per step, in three lanes:
  CPU   symbolic analysis    "symbolic" (main task)
        plan                 the GPU's plan, main task: "top selection" to "top batch" (else "plan (top)")
        A's entries          "copy entries" (task entries): A's entries into the host factor, on CPU threads
        subtrees             "bottom (CPU)" (task subtrees): the subtrees' numeric factorization
        structure            the solve's structure (task structure): "structure" to "levels"
        solve setup          main task: "solve setup" to "allocate" (setup, operators, memory check, workspace)
        layers               the solve's layers, planned on a host thread while the GPU runs the closure: from the
                             first start to the last end of the steps "layer plan" and "slot plan"
  PCIe  A's top              "A's top entries and index arrays" (device_entries!, during the subtrees)
        (unlabelled)         "solve structure": the copy of the structure step's result (arrow from it)
        factors              "subtrees' factors and updates" (upload_bottom!, after all subtrees)
  GPU   top                  from the CUDA events: the top's factorization (from its issue to the event after it)
        (unlabelled sliver)  "merged factor and operators": the solve's setup on the device
        U*, XL*, relabel     X <- U*, X <- XL*, the relabelling of the result's columns
Not drawn: the host's issue of launches (main task's "top (GPU)", "closure", "relabel columns"), steps under 0.1 ms.
Arrows: a step that starts when another ends (the structure -> its copy; the subtrees -> their factors' copy -> the top).
Colors as in the per-step figure (fig-phases-1col.tex): symbolic analysis black!22 (no split: the trace has no separate ordering
step); the factorization (plan, A's entries, subtrees, top) oiorange; transfer and setup (copies, structure, setups,
relabel) oiorange!38; U* oisky; XL* oiblue; the layers, CPU work inside the closure, black!10.
The caption's numbers are defined here as macros (\\Tl...)."""
import csv, math, os, re, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "gpu")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "build", "figures", "fig-gpu-timeline.tex")
COMMIT = (sys.argv[3] if len(sys.argv) > 3 else "") or os.environ.get("GPU_COMMIT", "") or "lib-g"
# fe_tooth (ROME's test graph G15, a 3-D mesh): its phase shares are near the medians of the benchmark graphs with n > 50,000
GRAPH, KIND = "fe_tooth", "b200"
def warn(s):
    print("fig_gpu_timeline: WARNING: " + s)

# ================= data: the traced calls =================
with open(os.path.join(GDATA, "gpu_timeline_trace.csv"), newline="") as h:
    TR = [r for r in csv.DictReader(h) if r["graph"] == GRAPH and r["commit"] == COMMIT and "B200" in r["gpu"]]
EVT = {int(r["call"]): float(r["call_ms"]) for r in TR if r["set"] == "events"}
if not EVT:
    sys.exit(f"fig_gpu_timeline: no traced calls (set events) of {GRAPH} on a B200 at {COMMIT}")
if len(EVT) % 2 == 0:
    warn("an even number of traced calls: the lower median is drawn")
RC = min(c for c in EVT if EVT[c] == statistics.median_low(EVT.values()))      # the call drawn: the median by total
CALL = [r for r in TR if r["set"] == "events" and int(r["call"]) == RC]
TEND = EVT[RC]
def rows(name, lane):
    return [r for r in CALL if r["step"] == name and r["lane"] == lane]
def step(name, lane="cpu"):
    rs = rows(name, lane)
    if len(rs) != 1:
        sys.exit(f"fig_gpu_timeline: {len(rs)} rows of step '{name}' ({lane}) in traced call {RC}")
    return float(rs[0]["start_ms"]), float(rs[0]["end_ms"])
def span(names, lane="cpu"):
    rs = [r for n in names for r in rows(n, lane)]
    if not rs:
        sys.exit(f"fig_gpu_timeline: no rows of {names} ({lane}) in traced call {RC}")
    return min(float(r["start_ms"]) for r in rs), max(float(r["end_ms"]) for r in rs)
T = {}                                                   # bar -> (start, end), ms from the call's start
T["symbolic"] = step("symbolic")
T["plan"] = (step("top selection")[0], step("top batch")[1] if rows("top batch", "cpu") else step("plan (top)")[1])
T["entries"] = step("copy entries")
T["subtrees"] = step("bottom (CPU)")
T["structure"] = (step("structure")[0], step("levels")[1])
T["setup"] = (step("solve setup")[0], step("allocate")[1])
T["layers"] = span(["layer plan", "slot plan"])
T["atop"] = step("A's top entries and index arrays", "pcie")
T["supload"] = step("solve structure", "pcie")
T["factors"] = step("subtrees' factors and updates", "pcie")
for k, nm in (("top", "top"), ("gsetup", "merged factor and operators"), ("inverse", "X <- U*"), ("solve", "X <- XL*"),
              ("relabel", "relabel")):
    T[k] = step(nm, "gpu")
for a, b in (("structure", "supload"), ("subtrees", "factors"), ("factors", "top")):
    if abs(T[a][1] - T[b][0]) > 0.5:
        warn(f"{b} does not start when {a} ends ({T[a][1]:.2f} vs {T[b][0]:.2f} ms): its arrow is drawn anyway")
if TEND < T["relabel"][1] - 1e-6:
    warn("the call's total is shorter than its last event")

# the graph's n, and the numbers of the caption of earlier versions (timed calls, gpu_apsp.csv), kept while it is in use
APSP = []
with open(os.path.join(GDATA, "gpu_apsp.csv"), newline="") as h:
    APSP = [r for r in csv.DictReader(h) if r["gpu.kind"] == KIND and r["commit"] == COMMIT and r["graph"] == GRAPH
            and r.get("status") == "ok" and str(r.get("check.ok", "")).lower() == "true"]
def one(code):
    rs = [r for r in APSP if r["code"] == code]
    return rs[0] if len(rs) == 1 else None
un, ph = one("ours-auto"), one("ours-auto-phases")
if un is None:
    sys.exit(f"fig_gpu_timeline: no single ours-auto row of {GRAPH} ({KIND}, {COMMIT}) in gpu_apsp.csv")
N = int(un["n"])

# ================= geometry (pt; y up, the picture's top at 0) =================
W = 252.0                          # \columnwidth: the grid background spans it exactly
FX0, FX1 = 1.5, W - 1.5            # the lanes' frames
X0 = FX0 + 3.0                     # time 0
H, P = 10.0, 12.0                  # bar height, row pitch
PAD, GAP = 2.5, 3.0                # frame padding above/below its rows; gap between frames
GAPL = 2.5                         # gap between a bar and its name outside it
BASE = 2.2                         # baseline below a row's center (\footnotesize names centered on the bar)
# lanes: name, frame style, name style, rows
LANES = [("CPU", "sys cpu lane", "sys cpu label", 3), ("PCIe", "sys pcie lane", "sys cpu label", 1),
         ("GPU", "sys gpu lane", "sys gpu label", 1)]
f2 = lambda v: f"{v:.2f}"

# text widths (Times, \footnotesize = 8 pt; math letters as italic), to place names and to check that they fit
CW, CI = {}, {}
for fam, tab in (("ptmr8a", CW), ("ptmri8a", CI)):
    afm = os.popen(f"kpsewhich {fam}.afm 2>/dev/null").read().strip()
    if afm and os.path.isfile(afm):
        for m in re.finditer(r"C (-?\d+) ; WX (\d+) ;", open(afm).read()):
            if int(m.group(1)) > 0: tab[chr(int(m.group(1)))] = int(m.group(2))
def tw(s, size=8.0):
    w = 0.0
    for k, part in enumerate(re.split(r"\$", s)):
        if k % 2:                  # math: letters italic, \gets as 1 em, ^* as a small star
            part = part.replace(r"\gets", "\x01").replace("^*", "\x02").replace(" ", "")
            w += sum(1000 if c == "\x01" else 350 if c == "\x02" else CI.get(c, 600) for c in part)
        else:
            w += sum(CW.get(c, 600) for c in part)
    return w / 1000 * size

# time scale: the relabel's name right of the last bar, inside the frame
RELABEL = "relabel"
S = (FX1 - 2.0 - tw(RELABEL) - GAPL - X0) / TEND
xm = lambda t: X0 + S * t

# rows: (lane, row) -> center y
YC, FR = {}, {}
y = -1.5
for li, (name, _, _, nr) in enumerate(LANES):
    top = y
    for r in range(nr):
        YC[(li, r)] = top - PAD - H / 2 - r * P
    bot = top - 2 * PAD - H - (nr - 1) * P
    FR[li] = (top, bot)
    y = bot - GAP
# bars: key -> (lane, row, fill, text color, name, where the name goes: in / left / right / none)
FAC, XS, SYM, LAY = "oiorange", "oiorange!38", "black!22", "black!10"
BARS = [("symbolic", 0, 0, SYM, "black!88", "symbolic analysis", "in"),
        ("layers", 0, 0, LAY, "black!80", "layers", "in"),
        ("plan", 0, 0, FAC, "black!88", "plan", "right"),
        ("entries", 0, 1, FAC, "black!88", "$A$'s entries", "left"),
        ("subtrees", 0, 1, FAC, "black!88", "subtrees", "right"),
        ("structure", 0, 2, XS, "black!88", "structure", "left"),
        ("setup", 0, 2, XS, "black!88", "solve setup", "right"),
        ("atop", 1, 0, XS, "black!88", "$A$'s top", "left"),
        ("supload", 1, 0, XS, "black!88", "", "none"),
        ("factors", 1, 0, XS, "black!88", "factors", "right"),
        ("top", 2, 0, FAC, "black!88", "top", "left"),
        ("gsetup", 2, 0, XS, "black!88", "", "none"),
        ("inverse", 2, 0, "oisky", "black!88", "$U^*$", "in"),
        ("solve", 2, 0, "oiblue", "white", "$XL^*$", "in"),
        ("relabel", 2, 0, XS, "black!88", RELABEL, "right")]
# ================= names: each where it fits (its side, else in its bar, else the other side), never on another =========
ROWS = [(li, r) for li, (_, _, _, nr) in enumerate(LANES) for r in range(nr)]          # top to bottom
occ = {lr: [] for lr in ROWS}                            # (lane, row) -> [(x0, x1, what)]: bars and names placed
for k, li, r, *_ in BARS:
    occ[(li, r)].append((xm(T[k][0]), xm(T[k][1]), k))
for li, (name, _, _, nr) in enumerate(LANES):           # the lane's name, at the frame's left
    for r in ((1, 2) if nr >= 3 else (0,)):
        occ[(li, r)].append((FX0 + 3.0, FX0 + 3.0 + tw(name), name))
def free(lr, a, b, own):
    return FX0 + 1.0 <= a and b <= FX1 - 1.0 and all(b + 0.8 <= x0 or x1 + 0.8 <= a for x0, x1, k in occ[lr] if k != own)
# the names left of their bars in the CPU and PCIe lanes end on one edge (a column), left of the earliest of those bars
LEFTCOL = min(xm(T[b[0]][0]) for b in BARS if b[6] == "left" and b[1] < 2) - GAPL
PLACE = {}                                               # bar -> (in / left / right, (x0, x1) of its name)
for k, li, r, fill, tc, name, where in BARS:
    if where == "none":
        continue
    x0, x1, w = xm(T[k][0]), xm(T[k][1]), tw(name)
    e = min(LEFTCOL, x0 - GAPL) if li < 2 else x0 - GAPL
    cand = {"in": ((x0 + x1 - w) / 2, (x0 + x1 + w) / 2) if w + 2.0 <= x1 - x0 else None,
            "left": (e - w, e), "right": (x1 + GAPL, x1 + GAPL + w)}
    for pl in [where] + [q for q in ("in", "right", "left") if q != where]:
        if cand[pl] and (pl == "in" or free((li, r), *cand[pl], k)):
            break
    else:
        pl = where if cand[where] else "right"
        warn(f"no free place for the name of {k}: drawn {pl} anyway, over something")
    if pl != where:
        warn(f"the name of {k} goes {pl}, not {where}: look at the figure")
    PLACE[k] = (pl, cand[pl])
    if pl != "in":
        occ[(li, r)].append((*cand[pl], k + " (name)"))
# hand-offs: an arrow from the bottom of a bar at its end to the top of the next bar at its start, through free rows only
ARROWS = [("structure", (0, 2), "supload", (1, 0)), ("subtrees", (0, 1), "factors", (1, 0)), ("factors", (1, 0), "top", (2, 0))]
for a, ra, b, rb in ARROWS:
    x = xm(T[b][0])
    for lr in ROWS[ROWS.index(ra) + 1:ROWS.index(rb)]:
        if any(x0 - 1.5 < x < x1 + 1.5 for x0, x1, _ in occ[lr]):
            warn(f"the arrow {a} -> {b} crosses something in lane {LANES[lr[0]][0]} row {lr[1]}")

# ================= output =================
L = []
out = L.append
out("% generated by scripts/fig_gpu_timeline.py from data/gpu/gpu_timeline_trace.csv: do not edit by hand")
out(f"% {GRAPH} (n = {N}) on one B200, commit {COMMIT}: the untimed call traced with events, call {RC} of {len(EVT)} "
    f"(total {TEND:.1f} ms, the median); {S:.4f} pt per ms")
out("% " + ", ".join(f"{k} {v[0]:.1f}-{v[1]:.1f}" for k, v in T.items()))
out("% the caption's numbers:")
gtex, ntex = GRAPH.replace("_", r"\_"), f"{N:,}".replace(",", "{,}")
out(rf"\gdef\TlGraph{{{gtex}}}\gdef\TlN{{{ntex}}}\gdef\TlTraced{{{TEND:.0f}}}\gdef\TlTracedCalls{{{len(EVT)}}}%")
# numbers of the caption of earlier versions (gpu.tex until it takes the new caption): the timed calls, gpu_apsp.csv
if ph is not None:
    import json
    tot = json.loads(ph["phases_runs.total"][len("json:"):])
    out(rf"\gdef\TlCalls{{{len(tot)}}}\gdef\TlThreads{{{int(ph['provenance.threads'])}}}"
        rf"\gdef\TlSymTimed{{{1e3 * json.loads(ph['phases_runs.symbolic'][len('json:'):])[tot.index(statistics.median(tot))]:.1f}}}"
        rf"\gdef\TlTracedSym{{{T['symbolic'][1] - T['symbolic'][0]:.1f}}}%")
out(r"\begingroup%")
out(r"\input{figures/sysstyle.tex}%")
out(r"\definecolor{tlgrid}{HTML}{FAF6EC}\definecolor{tlgridln}{HTML}{ECE3CF}%")
out(r"\begin{tikzpicture}[sys picture, x=1pt, y=1pt,")
out(r"  bar/.style={sys outline},")
out(r"  name/.style={font=\footnotesize, inner sep=0pt, anchor=base},")
out(r"  axis tick/.style={sys tick},")
out(r"  axis label/.style={font=\scriptsize, text=black!65, inner sep=0pt, anchor=base}]")
ybot = FR[len(LANES) - 1][1] - 2.0 - 2.0 - 8.6           # ticks (2 pt), the tick labels' line
# the grid background, its grid on time 0 and the top
out(rf"\fill[tlgrid] (0,{f2(ybot)}) rectangle ({f2(W)},0);")
CELL = 2.5 * 72.27 / 25.4
xs = [X0 + k * CELL for k in range(math.ceil(-X0 / CELL), math.floor((W - X0) / CELL) + 1) if 0 < X0 + k * CELL < W]
ys = [-k * CELL for k in range(1, math.floor(-ybot / CELL) + 1) if -k * CELL > ybot]
out(r"\draw[tlgridln, line width=0.25pt] " + " ".join(f"({f2(x)},{f2(ybot)})--({f2(x)},0)" for x in xs) + " "
    + " ".join(f"(0,{f2(y)})--({f2(W)},{f2(y)})" for y in ys) + ";")
# the lanes' frames and names (the name at the frame's left, centered on the frame; in the CPU lane, between rows 1 and 2,
# below the symbolic analysis)
for li, (name, fstyle, nstyle, nr) in enumerate(LANES):
    t, b = FR[li]
    out(rf"\path[{fstyle}] ({f2(FX0)},{f2(b)}) rectangle ({f2(FX1)},{f2(t)});")
    yc = (YC[(li, 1)] + YC[(li, 2)]) / 2 if nr >= 3 else (t + b) / 2
    out(rf"\node[{nstyle}, anchor=base west] at ({f2(FX0 + 3.0)},{f2(yc - 2.6)}) {{{name}}};")
# bars and names
for k, li, r, fill, tc, name, where in BARS:
    x0, x1, yc = xm(T[k][0]), xm(T[k][1]), YC[(li, r)]
    out(rf"\path[bar, fill={fill}] ({f2(x0)},{f2(yc - H / 2)}) rectangle ({f2(x1)},{f2(yc + H / 2)});")
    if k in PLACE:
        pl, (n0, n1) = PLACE[k]
        anchor, xn, col = {"in": ("base", (n0 + n1) / 2, tc), "left": ("base east", n1, "black!85"),
                           "right": ("base west", n0, "black!85")}[pl]
        out(rf"\node[name, text={col}, anchor={anchor}] at ({f2(xn)},{f2(yc - BASE)}) {{{name}}};")
for a, ra, b, rb in ARROWS:
    x = xm(T[b][0])
    out(rf"\draw[sys arrow] ({f2(x)},{f2(YC[ra] - H / 2)}) -- ({f2(x)},{f2(YC[rb] + H / 2)});")
# the time axis: ticks every 100 ms, below the GPU lane
ya = FR[len(LANES) - 1][1] - 1.0
ticks = list(range(0, int(TEND) + 1, 100))
for t in ticks:
    out(rf"\draw[axis tick] ({f2(xm(t))},{f2(ya)}) -- ({f2(xm(t))},{f2(ya - 2.0)});")
    lab = f"{t}" + (r"\,ms" if t == ticks[-1] else "")
    out(rf"\node[axis label] at ({f2(xm(t))},{f2(ya - 2.0 - 6.6)}) {{\phantom{{\,ms}}{lab}}};" if t == ticks[-1] else
        rf"\node[axis label] at ({f2(xm(t))},{f2(ya - 2.0 - 6.6)}) {{{lab}}};")
out(r"\end{tikzpicture}%")
out(r"\endgroup\par% end the paragraph: main.tex's line break after \input would add a space to an exactly \columnwidth line")
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w") as h:
    h.write("\n".join(L) + "\n")
print(f"fig_gpu_timeline: wrote {OUT} (call {RC} of {len(EVT)}, {TEND:.1f} ms; {S:.4f} pt/ms; height {-ybot:.1f} pt)")
