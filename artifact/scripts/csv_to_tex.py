#!/usr/bin/env python3
"""Turn the benchmark CSV files into pgfplots data files, LaTeX table fragments, and number macros.

Usage:  python3 csv_to_tex.py [DATA_DIR] [OUT_DIR]          (defaults: data build)
Reads:  DATA_DIR/*.csv (written by collect.py; external.csv for the C++ baselines)
Writes: OUT_DIR/data/*.dat, OUT_DIR/data/macros.tex, OUT_DIR/data/schematic_*.tex, OUT_DIR/tables/*.tex
Nothing here needs the cluster: the CSVs are the only input.
"""
import json, glob, math, os, sys, copy, csv, re, statistics as st
from fnmatch import fnmatch
DATA = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "build")

# ---- the CSV files, loaded as a virtual file system: source path -> record (as the raw JSON was) ----
SCALE_FAMILIES = {"scaling", "scaling_factor", "scaling_rows"}
NOT_RESULTS = {"external", "superfw_published", "regions"}
def _val(v):
    if v == "null": return None
    if v in ("true", "false"): return v == "true"
    if v.startswith("json:"): return json.loads(v[5:])
    if re.fullmatch(r"-?\d+", v): return int(v)
    try: return float(v)
    except ValueError: return v
def _unflat(row):
    rec = {}
    for k, v in row.items():
        if k in ("file", "source", "commit", "machine", "date", "job", "_row") or v == "": continue
        *path, last = k.split("."); d = rec
        for p in path: d = d.setdefault(p, {})
        d[last] = _val(v)
    return rec
VFS = {}; KIND = {}; REL = {}
for _p in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
    _fam = os.path.basename(_p)[:-4]
    if _fam in NOT_RESULTS: continue
    _grp = {}
    for _r in csv.DictReader(open(_p)): _grp.setdefault(_r["source"], []).append(_r)
    for _s, _rs in _grp.items():
        VFS[_s] = [_unflat(r) for r in sorted(_rs, key=lambda r: int(r["_row"]))] if _rs[0].get("_row") not in (None, "") else _unflat(_rs[0])
        KIND[_s] = "scale" if _fam in SCALE_FAMILIES else "results"; REL[_s] = _rs[0]["file"]
SCALE = any(k == "scale" for k in KIND.values()) or None
def J(f): return copy.deepcopy(VFS[f])
def G(pattern):                    # results files matching pattern, in the order the raw files sorted
    return sorted(s for s, k in KIND.items() if k == "results" and fnmatch(REL[s], pattern))
def SG(pattern):                   # scaling files matching pattern
    return sorted(s for s, k in KIND.items() if k == "scale" and fnmatch(REL[s], pattern))
os.makedirs(os.path.join(OUT, "data"), exist_ok=True); os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.chdir(OUT)
REGIONS = os.path.join(DATA, "regions.csv")
# The baseline for queries and closures: on each graph, the faster of GAP's Delta-stepping (one core, best Delta of a
# sweep; external.csv) and our Dijkstra. GAP is faster on all but a few graphs.
GAP1 = {}
_ext = os.path.join(DATA, "external.csv")
if os.path.isfile(_ext):
    for _r in csv.DictReader(open(_ext)):
        if _r["tool"] == "gap" and _r["placement"] == "core" and _r["metric"] == "latency_s":
            GAP1[_r["graph"]] = min(GAP1.get(_r["graph"], float("inf")), float(_r["value"]))
def BASE(g, own):
    return min(own, GAP1.get(g, float("inf")))
def ld(pattern, key="graph"):
    return {J(f)[key]: J(f) for f in G(pattern)}

NAMES = {"USA-t-NY": "USA-road-t NY", "USA-t-BAY": "USA-road-t BAY", "USA-t-COL": "USA-road-t COL", "USA-t-FLA": "USA-road-t FLA", "USA-t-NW": "USA-road-t NW", "USA-t-NE": "USA-road-t NE", "USA-t-CAL": "USA-road-t CAL", "USA-t-E": "USA-road-t E", "USA-t-W": "USA-road-t W"}
def tex(g): return NAMES.get(g, g).replace("_", r"\_")
def fmt_n(n): return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.1f}k" if n >= 1e4 else f"{n:,}".replace(",", "{,}"))
def fmt_ms(x):                        # milliseconds, three significant digits, trailing zeros kept
    v = 1e3 * x
    if v != v: return "--"
    if v == 0: return "0"
    d = max(0, 2 - int(math.floor(math.log10(abs(v)))))
    return f"{v:.{d}f}"
    t *= 1e3
    return f"{t:.3g}" if t < 100 else f"{t:.0f}"
def fmt_x(x): return str(x) if isinstance(x, int) else (f"{x:.1f}" if x < 10 else f"{x:.0f}")
def rng(v):
    v = sorted(v)
    if not v: return "??"
    a, b = fmt_x(v[0]), fmt_x(v[-1])
    return a if a == b else f"{a}--{b}"
def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
macros = {}

# ---------------- kernels ----------------
k = J(G("exp1_kernels.json")[-1])
dg = max(r["gmuladd"] for r in k if r["semiring"] == "OpenBLAS dgemm"); clock = dg / 16
macros.update(DgemmRate=f"{dg:.1f}", MinPlusBound=f"{dg / 2:.1f}")
kr = lambda sr, T, n=1024: next(r["gmuladd"] for r in k if r["semiring"] == sr and r["T"] == T and r["n"] == n)
# kernel rates relative to OpenBLAS dgemm on the same core (Float32: relative to twice the dgemm rate)
macros.update(MinPlusRate=f"{kr('MinPlus','Float64'):.1f}", MinPlusPct=f"{100*kr('MinPlus','Float64')/dg:.0f}",
              MinPlusSPRate=f"{kr('MinPlus','Float32'):.1f}", MinPlusSPPct=f"{100*kr('MinPlus','Float32')/(2*dg):.0f}",
              KernelPeak=f"{kr('MinPlus','Float64'):.1f}")

# ---------------- queries ----------------
q3 = ld("exp3_*.json"); q3b = ld("exp3b_*.json")
e11 = ld("exp11_*.json"); e13 = ld("exp13_*.json"); e12 = {}
e16 = {}; e16p = {}                    # elementary (fast-path) single queries: one thread, and all threads
E18_all = {}                            # scaling runs (exp18), loaded early for the external comparisons; later dirs override
for _f in SG("exp18_*.json"):
    _r = J(_f); E18_all.setdefault(_r["graph"], {})[_r["nt"]] = _r
for f in G("exp16_*.json"):
    r = J(f)
    (e16 if r["nt"] == 1 else e16p)[r["graph"]] = r
for _g, _r in e16p.items():            # the 16-thread runs also time one-core queries and Dijkstra, alone in a region
    e16[_g] = _r
for f in G("exp12_*.json"):
    r = J(f)
    if r["nt"] == 1 and (r["graph"] not in e12 or r["ordering"] == "AMF"): e12[r["graph"]] = r
src = "elimination coordinates, medians" if e11 else ("exp3b (medians)" if q3b else "exp3 (means)")
def qrow(g):
    if g in e16 and g in e13:              # final run: fast-path runs (Dijkstra, single query) + exp13 (factor, block-size sweep)
        z = e13[g]; q = e16[g]; sym = z.get("symmetric", True)
        td = BASE(g, q["dijkstra"]); t1 = q["elem_target_1"]["median"]
        ks = {1: t1}; ratio = {}
        for kk, v in z["sweep"].items():
            if int(kk) > 1: ks[int(kk)] = min(v["column"], v["row"]) if sym else v["column"]
            if sym: ratio[int(kk)] = v["column"] / v["row"]
        cands = [(t, kk) for kk, t in ks.items() if kk > 1] or [(t1, 1)]
        tb, kb = min(cands)
        tsym, tnum = z["t_sym"], z["t_num"]
        be = ("never", "never") if td <= tb else ((tsym + tnum) / (td - tb), tnum / (td - tb))
        return dict(n=z["n"], m=z["m"], fill=z["fill_ratio"], tsym=tsym, tnum=tnum, td=td, t1=t1, tb=tb, kb=kb,
                    be=be, directed=not sym, ks=ks, nnz=z["nnzLU"], ratio=ratio)
    if g in e11:
        r = e11[g]; z = e13.get(g); w = e12.get(g)
        td = min(r["dijkstra_original"]["median"], r["dijkstra_elimination_order"]["median"])
        t1 = r["solve_elimination_coords"]["median"]
        if g in e16:                   # fast path: elementary solve; its preallocated Dijkstra is the baseline
            td = e16[g]["dijkstra"]; t1 = e16[g]["elem_target_1"]["median"]
        sym = (z or w or {}).get("symmetric", True)
        ks = {1: t1}; ratio = {}
        # (k = 1 in the blocking figure is the single query, fast path when available)
        if z:
            for kk, v in z["sweep"].items():
                if int(kk) > 1: ks[int(kk)] = min(v["column"], v["row"]) if sym else v["column"]
                if sym: ratio[int(kk)] = v["column"] / v["row"]
        cands = [(r["blocked_elimination_coords_per_query"], r["k"])] + [(t, kk) for kk, t in ks.items() if kk > 1]
        if w:
            cands.append((w["blocked_column_layout_per_query"], w["k"]))
            if sym: cands.append((w["blocked_row_layout_per_query"], w["k"]))
        tb, kb = min(cands)
        tsym, tnum = (z["t_sym"], z["t_num"]) if z else (None, None)
        be = ("never", "never") if td <= tb else (((tsym + tnum) / (td - tb), tnum / (td - tb)) if z else (None, None))
        return dict(n=r["n"], m=r["m"], fill=z["fill_ratio"] if z else None, tsym=tsym, tnum=tnum, td=td, t1=t1, tb=tb, kb=kb,
                    be=be, directed=not sym, ks=ks, nnz=z["nnzLU"] if z else None, ratio=ratio)
    if g in q3b:
        r = q3b[g]; td = r["dijkstra"]["median"]; t1 = r["solve1"]["median"]; kb = r["k_blocked"]
        tb = r["blocked_per_query_median"].get(str(kb), t1)
        return dict(n=r["n"], m=r["m"], fill=r["nnzLU"] / r["m"], tsym=r["t_sym"], tnum=r["t_num"], td=td, t1=t1, tb=tb, kb=kb,
                    be=(r.get("breakeven_total"), r.get("breakeven_numeric")), directed=r.get("class", "").endswith("(dir.)"),
                    ks={int(kk): v for kk, v in r["blocked_per_query_median"].items()} | {1: t1}, nnz=r["nnzLU"])
    if g in q3:
        r = q3[g]; ts = r["t_solve"]; kb = max(int(x) for x in ts); td = r["t_dijkstra"]
        be = (r["breakeven"], r["breakeven_num"]) if td > ts[str(kb)] / kb else (None, None)
        return dict(n=r["n"], m=r["m"], fill=r["nnzLU"] / r["m"], tsym=r["t_sym"], tnum=r["t_num"], td=td, t1=ts["1"], tb=ts[str(kb)] / kb,
                    kb=kb, be=be, directed=r["directed"], ks={int(kk): v / int(kk) for kk, v in ts.items()}, nnz=r["nnzLU"])
    return None
ROWS = ["roadNet-PA", "roadNet-CA", "belgium_osm", "luxembourg_osm", "USA-t-NY", "USA-t-FLA", "rgg2d-1M", "finan512", "c-69",
        "grid2d-1000", "oilpan", "wing", "grid3d-40", "email-Enron", "p2p-Gnutella31", "soc-Epinions1"]
lines = []
for g in ROWS:
    r = qrow(g)
    if not r: continue
    s1, sb = r["td"] / r["t1"], r["td"] / r["tb"]
    b0 = r["be"][0]
    be = "never" if b0 == "never" or (b0 is None and s1 < 1 and sb < 1) else ("--" if b0 is None else f"{math.ceil(r['be'][0])} / {math.ceil(r['be'][1])}")
    name = tex(g) + (r" (dir.)" if r["directed"] else "")
    fillc = "--" if r["fill"] is None else fmt_x(r["fill"])
    setup = "--" if r["tsym"] is None else f"{r['tsym']:.2f} + {r['tnum']:.2f}"
    lines.append(f"{name} & {fmt_n(r['n'])} & {fillc} & {setup} & {fmt_ms(r['td'])} & {fmt_ms(r['t1'])} & "
                 f"{fmt_ms(r['tb'])} ({r['kb']}) & {s1:.1f}$\\times$ / {sb:.1f}$\\times$ & {be} \\\\")
open("tables/tab-queries.tex", "w").write(f"% generated by csv_to_tex.py from {src}\n" + "\n".join(lines) + "\n")
ROADS = ["roadNet-PA", "roadNet-TX", "roadNet-CA", "luxembourg_osm", "belgium_osm", "USA-t-NY", "USA-t-BAY", "USA-t-COL", "USA-t-FLA"]
rr = [qrow(g) for g in ROADS if qrow(g)]
macros.update(NRoads=str(len(rr)), RoadSingle=rng([r["td"] / r["t1"] for r in rr]), RoadBlocked=rng([r["td"] / r["tb"] for r in rr]),
              RoadBreakEven=rng([math.ceil(r["be"][0]) for r in rr if isinstance(r["be"][0], float)]),
              RoadBreakEvenNum=rng([math.ceil(r["be"][1]) for r in rr if isinstance(r["be"][1], float)]),
              QuerySource=src)
SFW = ["c-69", "delaunay_n16", "email-Enron", "fe_tooth", "finan512", "lpl1", "luxembourg_osm", "net4-1", "oilpan", "onera_dual", "t60k", "wing"]
sf = [qrow(g) for g in SFW if qrow(g)]
def best_rate(g):
    r = qrow(g)
    return max(r["nnz"] * kk / (r["ks"][kk] * kk) / 1e9 for kk in r["ks"]) if r and r.get("nnz") else None
RR = [best_rate(g) for g in ROADS if best_rate(g)]
BIG = [best_rate(g) for g in ["oilpan", "wing", "grid3d-40", "email-Enron", "fe_tooth", "onera_dual"] if best_rate(g)]
def col_rate(g):
    z = e13.get(g)
    return max(z["nnzLU"] / v["column"] / 1e9 for v in z["sweep"].values()) if z else None
macros.update(RoadFillHalf=rng([qrow(g)["fill"] / 2 for g in ROADS if qrow(g) and qrow(g).get("fill")]))
macros.update(RoadFill=rng([qrow(g)["fill"] for g in ROADS if qrow(g) and qrow(g).get("fill")]))
macros.update(RoadRate=rng(RR), BigRate=rng(BIG), RoadRateCol=rng([col_rate(g) for g in ROADS if col_rate(g)]),
              BigPeakPct=rng([100 * b / kr('MinPlus', 'Float64') for b in BIG]))
LARGE = [g for g in ["USA-t-NW", "USA-t-NE", "USA-t-CAL", "USA-t-E", "USA-t-W"] if g in e16 or g in e11]
macros.update(LargeRoadSingle=rng([qrow(g)["td"] / qrow(g)["t1"] for g in LARGE]), NLargeRoads=str(len(LARGE)),
              LargeRoadMaxN=f"{max((e16.get(g) or e11.get(g))['n'] for g in LARGE) / 1e6:.1f}" if LARGE else "??")
if e16p:
    RG8 = [g for g in ROADS if g in e16p and g in e16]
    macros.update(ElemP=str(max(r["nt"] for r in e16p.values())),
        ElemParSpeedup=rng([e16[g]["elem_target_1"]["median"] / e16p[g]["elem_target_nt"]["median"] for g in RG8]),
        ElemParVsDijkstra=rng([e16p[g]["dijkstra"] / e16p[g]["elem_target_nt"]["median"] for g in RG8]))
if "wiki-Vote" in e16:                 # the fast path on a directed graph with a small factor
    w16 = e16["wiki-Vote"]
    macros.update(WikiFastGain=f'{w16["dense_single"]["median"] / w16["elem_target_1"]["median"]:.0f}',
                  WikiVsDij=f'{w16["elem_target_1"]["median"] / BASE("wiki-Vote", w16["dijkstra"]):.0f}')
macros.update(SFWCount=str(len(sf)), SFWBlocked=rng([r["td"] / r["tb"] for r in sf]), SFWSingleWins=str(sum(r["td"] / r["t1"] > 1.05 for r in sf)),
              SFWSingleWinsText="all" if sum(r["td"] / r["t1"] > 1.05 for r in sf) == len(sf) else str(sum(r["td"] / r["t1"] > 1.05 for r in sf)))
SER = ["roadNet-PA", "USA-t-FLA", "luxembourg_osm", "finan512", "oilpan", "wing", "grid3d-40", "email-Enron", "p2p-Gnutella31", "ER-16k"]
for g in SER:
    r = qrow(g)
    if not r: continue
    with open(f"data/qk_{g}.dat", "w") as f:
        f.write("k speedup rate ratio\n")
        for kk in sorted(r["ks"]):
            rate = r["nnz"] / r["ks"][kk] / 1e9 if r.get("nnz") else float("nan")
            f.write(f"{kk} {r['td'] / r['ks'][kk]:.4g} {rate:.4g} {r.get('ratio', {}).get(kk, float('nan')):.4g}\n")
with open("data/qk_plots_speedup.tex", "w") as f1, open("data/qk_plots_rate.tex", "w") as f2, open("data/qk_plots_ratio.tex", "w") as f3:
    for g in SER:
        if not qrow(g): continue
        f1.write(f"\\addplot table[x=k, y=speedup] {{data/qk_{g}.dat}};\n\\addlegendentry{{{tex(g)}}}\n")
        f2.write(f"\\addplot table[x=k, y=rate, unbounded coords=discard] {{data/qk_{g}.dat}};\n")
        f3.write(f"\\addplot table[x=k, y=ratio, unbounded coords=discard] {{data/qk_{g}.dat}};\n")
allq = {g: qrow(g) for g in set(q3) | set(q3b) | set(e11) | set(e16) if qrow(g) and not g.endswith("+sfw")}   # +sfw: SuperFW table only
with open("data/fillratio.dat", "w") as f:
    f.write("fill speedup directed label anchor\n")
    LAB = {"grid3d-40": "west", "ER-16k": "east", "wiki-Vote": "west", "soc-Epinions1": "east", "p2p-Gnutella31": "east"}
    for g, r in sorted(allq.items()):
        if r["fill"] is None: continue
        lab = tex(g) if g in LAB else "{}"
        f.write(f"{r['fill']:.4g} {r['td'] / r['t1']:.4g} {int(r['directed'])} {lab} {LAB.get(g, 'west')}\n")
fr_pts = [(r["fill"], r["td"] / r["t1"]) for r in allq.values() if r["fill"] is not None]
und = [(r["fill"], r["td"] / r["t1"]) for r in allq.values() if r["fill"] is not None and not r["directed"]]
# every undirected graph up to wmax wins, every one from lmin on loses; graphs in between form a band near parity
_u = sorted(und)
wmax = max([f for f, _ in _u if all(sp >= 1 for g_, sp in _u if g_ <= f)], default=float("nan"))
lmin = min([f for f, _ in _u if all(sp < 1 for g_, sp in _u if g_ >= f)], default=float("nan"))
_band = [(f, sp) for f, sp in _u if wmax < f < lmin]
_next = min([f for f, _ in _u if f > wmax], default=lmin); _prev = max([f for f, _ in _u if f < lmin], default=wmax)
macros.update(FillWinMax=f"{wmax:.0f}", FillLoseMin=f"{lmin:.0f}", FillSeparable="0" if _band else "1",
              FillLine=f"{math.sqrt(wmax * _next):.1f}", FillLineB=f"{math.sqrt(_prev * lmin):.1f}",
              FillBandN=str(len(_band)), FillBandSpeed=rng([sp for _, sp in _band]) if _band else "")
xs = [math.log(a) for a, _ in fr_pts]; ys = [math.log(b) for _, b in fr_pts]
macros.update(HasLayoutData="1" if any(qrow(g) and qrow(g).get("ratio") for g in SER) else "0")
macros.update(FillCorr=f"{corr(xs, ys):.2f}".replace("-", "$-$"), NQueryGraphs=str(len(allq)))
_own = [g for g in sorted(allq) if g in e16 and g in GAP1 and e16[g]["dijkstra"] < GAP1[g]]
macros.update(BaseOwnGraphs=" and ".join(tex(g) for g in _own) if len(_own) <= 2 else ", ".join(tex(g) for g in _own[:-1]) + ", and " + tex(_own[-1]),
              NBaseOwn=str(len(_own)))

# ---------------- factorization intensity ----------------
rows = [J(f) for f in G(os.path.join("exp2_rows", "*.json"))]
ORD = {"AMF": "amf", "MMD": "mmd", "METIS": "metis", "ND": "nd", "SafeRules(METIS)": "srmetis", "Compression(SafeRules(METIS))": "csrmetis", "SafeRules(ND)": "srnd"}
pts = {o: [] for o in ORD}
for r in rows:
    if "asmwords" in r and isinstance(r.get("t_num"), (int, float)) and r["t_num"] and r["muladds"] > 5e8:
        pts[r["ordering"]].append((r["muladds"] / r["asmwords"], r["muladds"] / r["t_num"] / 1e9))
allp = [p for v in pts.values() for p in v]
# nested-dissection orderings vs. minimum degree (AMF): factorization time on complex networks
_tn = {}
for r in rows:
    if isinstance(r.get("t_num"), (int, float)) and r["t_num"]: _tn.setdefault(r["graph"], {})[r["ordering"]] = r["t_num"]
_nd = []
for _g in ("email-Enron", "ER-16k"):
    _d = _tn.get(_g, {})
    if "AMF" in _d: _nd += [v / _d["AMF"] for o, v in _d.items() if "METIS" in o or "ND" in o]
if _nd: macros["OrderNDSlow"] = rng(_nd)
for o, v in pts.items():
    with open(f"data/intensity_{ORD[o]}.dat", "w") as f:
        f.write("intensity rate\n"); [f.write(f"{a:.4g} {b:.4g}\n") for a, b in v]
macros.update(IntensityCorr=f"{corr([math.log(a) for a, _ in allp], [math.log(b) for _, b in allp]):.2f}", NIntensity=str(len(allp)))

# ---------------- all pairs ----------------
a = ld("exp4_*.json")
for g, r in a.items():                  # give n x Dijkstra the better of the two labelings (exp11 per-graph ratio)
    src_l = e16.get(g) or e11.get(g)
    if src_l:
        rr = src_l["dijkstra_elimination_order"]["median"] / src_l["dijkstra_original"]["median"]
        r["t_dijkstra_all"] *= min(1.0, rr)
    if g in GAP1: r["t_dijkstra_all"] = min(r["t_dijkstra_all"], r["n"] * GAP1[g])    # n runs of the faster baseline
    if g in e13: r["t_sym"] = e13[g]["t_sym"]      # single-ordering symbolic time (exp4 ran two orderings)
AP = ["USpowerGrid", "ca-GrQc", "grid2d-90", "rgg2d-8k", "c-42", "G67", "delaunay_n14", "fe_sphere", "ca-HepTh", "BA-8k", "nd6k", "ER-8k", "hypercube-13", "wiki-Vote"]
apg = list(reversed([g for g in AP if g in a]))
with open("data/apsp.dat", "w") as f:
    f.write("idx factor closure perm speedup\n")
    for i, g in enumerate(apg):
        r = a[g]; d = r["t_dijkstra_all"]; tot = r["t_sym"] + r["t_num"] + r["t_closure"]
        f.write(f"{i} {(r['t_sym'] + r['t_num']) / d:.4g} {r['t_closure'] / d:.4g} {r['t_perm'] / d:.4g} {d / tot:.3g}\n")
xmax = max((a[g]["t_sym"] + a[g]["t_num"] + a[g]["t_closure"]) / a[g]["t_dijkstra_all"] for g in apg)
with open("data/apsp_labels.tex", "w") as f:      # speedup labels at the exact end of each stacked bar
    for i, g in enumerate(apg):
        r = a[g]; d = r["t_dijkstra_all"]; tot = r["t_sym"] + r["t_num"] + r["t_closure"]
        f.write(f"\\node[apsplabel] at (axis cs:{tot / d:.4g},{i}) {{{d / tot:.1f}$\\times$}};\n")
macros.update(APSPXmax=f"{math.ceil((xmax + 0.35) * 5) / 5:.1f}")
open("data/apsp_style.tex", "w").write("\\pgfplotsset{apsp labels/.style={ytick={0,...," + str(len(apg) - 1) + "}, yticklabels={"
    + ",".join("{" + tex(g) + "}" for g in apg) + "}, ymin=-0.6, ymax=" + f"{len(apg) - 0.4}" + "}}\n")
small = [g for g in AP[:10] if g in a]
macros.update(APSPRange=rng([a[g]["t_dijkstra_all"] / (a[g]["t_sym"] + a[g]["t_num"] + a[g]["t_closure"]) for g in small]),
              APSPRate=rng([a[g]["n"] * a[g]["nnzLU"] / a[g]["t_closure"] / 1e9 for g in small]),
              APSPPermMax=f'{max(100 * a[g]["t_perm"] / (a[g]["t_sym"] + a[g]["t_num"] + a[g]["t_closure"]) for g in small):.0f}',
              APSPLosers=", ".join(tex(g) for g in AP[10:] if g in a and a[g]["t_dijkstra_all"] / (a[g]["t_sym"] + a[g]["t_num"] + a[g]["t_closure"]) < 3),
              APSPDense=rng([a[g]["t_dense"] / a[g]["t_closure"] for g in a if a[g].get("t_dense")]))

# ---------------- dense right-hand sides and difference constraints ----------------
d9 = [J(f) for f in G("exp9_*.json")]
de = {J(f)["graph"]: J(f) for f in G("exp5e_smooth_*.json")}
dr = {J(f)["graph"]: J(f) for f in G("exp5e_random_*.json")}
DG = ["roadNet-PA", "USA-t-FLA", "USA-t-NY", "luxembourg_osm"]; lines = []
for g in DG:
    rs = {r["family"]: r for r in d9 if r["graph"] == g}
    if not rs: continue
    best = "\\,/\\,".join(f"{rs[fam]['speedup_vs_best_dense_baseline']:.1f}" if fam in rs else "--" for fam in ("offsets", "facilities", "smooth"))
    ss = rng([r["speedup_vs_single_source"] for r in rs.values()])
    if g in de:
        e = de[g]; ns = e["t_num"] + e["t_solve"]; tot = e["t_sym"] + ns
        dc = f"{fmt_x(e['t_spfa'] / ns)}\\,/\\,{e['t_spfa'] / tot:.1f} & {fmt_x(e['t_bf'] / ns)}\\,/\\,{e['t_bf'] / tot:.1f}"
    else: dc = "-- & --"
    lines.append(f"{tex(g)} & {best} & {ss} & {dc} \\\\")
open("tables/tab-dense.tex", "w").write("% generated by csv_to_tex.py\n" + "\n".join(lines) + "\n")
macros.update(DenseBest=rng([r["speedup_vs_best_dense_baseline"] for r in d9]), DenseSingle=rng([r["speedup_vs_single_source"] for r in d9]),
              DCspfa=rng([e["t_spfa"] / (e["t_num"] + e["t_solve"]) for e in de.values()]),
              DCspfaMax=fmt_x(max(e["t_spfa"] / (e["t_num"] + e["t_solve"]) for e in de.values())),
              DCjacobi=rng([e["t_bf"] / (e["t_num"] + e["t_solve"]) for e in de.values()]),
              DCspfaTotal=rng([e["t_spfa"] / (e["t_sym"] + e["t_num"] + e["t_solve"]) for e in de.values() if e["n"] > 2e5]),
              DCspfaTotalMax=fmt_x(max(e["t_spfa"] / (e["t_sym"] + e["t_num"] + e["t_solve"]) for e in de.values())),
              DCjacobiTotal=rng([e["t_bf"] / (e["t_sym"] + e["t_num"] + e["t_solve"]) for e in de.values()]),
              DCsweeps=rng([e["bf_iters"] for e in de.values()]).replace(".0", ""),
              DCrandomSPFA=fmt_x(max([(e["t_num"] + e["t_solve"]) / e["t_spfa"] for e in dr.values() if e.get("spfa_finished")], default=0)))

# ---------------- other semirings ----------------
c5 = ld("exp5c_*.json"); m5 = ld("exp5b_*.json"); lines = []
wa = {}
for f in G("exp14a_*.json"): r = J(f); wa[(r["graph"], r["variant"])] = r
wb = ld("exp14b_*.json"); wc = ld("exp14c_*.json")
def blk(title, base, items):
    return [f"{title if i == 0 else ''} & {tex(g)} & {base if i == 0 else ''} & {s1} & {sb} \\\\" for i, (g, s1, sb) in enumerate(items)]
WG = ["roadNet-PA", "USA-t-FLA", "belgium_osm", "grid2d-1000"]
if wa:
    s1w = lambda r: r.get("solve1_elem") or r["solve1"]
    lines += blk("widest, directed", "Dijkstra", [(g, fmt_x(wa[(g, "asym")]["dijkstra_target"] / s1w(wa[(g, "asym")])),
        fmt_x(wa[(g, "asym")]["dijkstra_source"] / wa[(g, "asym")]["blocked_row_per_query"])) for g in WG if (g, "asym") in wa])
    lines.append("\\midrule")
    best = lambda r: min(r["blocked_column_per_query"], r["blocked_row_per_query"])
    lines += blk("widest, undirected", "spanning tree", [(g, fmt_x(wa[(g, "sym")]["mst_query"] / s1w(wa[(g, "sym")])),
        fmt_x(wa[(g, "sym")]["mst_query"] / best(wa[(g, "sym")]))) for g in WG if (g, "sym") in wa])
    lines.append("\\midrule")
    RO = [g for (g, v) in wa if v == "sym" and g not in ("grid2d-1000", "rgg2d-1M")]
    macros.update(HasWidest="1",
        WidestDirSingle=rng([r["dijkstra_target"] / s1w(r) for (g, v), r in wa.items() if v == "asym"]),
        WidestDirBlocked=rng([r["dijkstra_source"] / r["blocked_row_per_query"] for (g, v), r in wa.items() if v == "asym"]),
        MSTOverSingle=rng([s1w(r) / r["mst_query"] for (g, v), r in wa.items() if v == "sym"]),
        MSTBlockedRoad=rng([wa[(g, "sym")]["mst_query"] / best(wa[(g, "sym")]) for g in RO]))
if wb:
    RG = ["cit-HepPh", "email-EuAll", "wiki-Vote", "p2p-Gnutella31", "soc-Epinions1"]
    lines += blk("reachability", "MS-BFS", [(g, "--", f'{wb[g]["msbfs256_target"] / wb[g]["relation_target_per_query"]:.1f} / {wb[g]["msbfs256_source"] / wb[g]["relation_source_per_query"]:.1f}')
                                          for g in RG if g in wb])
    lines.append("\\midrule")
    bestr = {g: max(wb[g]["msbfs256_target"] / wb[g]["relation_target_per_query"], wb[g]["msbfs256_source"] / wb[g]["relation_source_per_query"]) for g in wb}
    macros.update(ReachN=str(len(wb)), ReachWins=str(sum(v > 1.05 for v in bestr.values())), ReachTies=str(sum(0.95 <= v <= 1.05 for v in bestr.values())),
                  ReachFactorMax=f'{max(r["t_factor"] for r in wb.values()):.1f}',
                  ReachWorst=fmt_x(max([1 / v for v in bestr.values() if v < 0.95], default=1.0)))
    _src = [wb[g]["msbfs256_source"] / wb[g]["relation_source_per_query"] for g in wb]
    _tgt = [wb[g]["msbfs256_target"] / wb[g]["relation_target_per_query"] for g in wb]
    macros.update(ReachSrcWins=str(sum(v > 1 for v in _src)), ReachSrcRange=rng(_src),
                  ReachTgtLosses=str(sum(v < 1 for v in _tgt)), ReachTgtWorst=fmt_x(max([1 / v for v in _tgt if v < 1], default=1.0)))
if m5:                                   # reals: our generic engine vs. a production numerical direct solver (solves)
    lines += blk("M-matrix LU", "UMFPACK", [(g, fmt_x(m5[g]["t_umf1"] / m5[g]["t_solve1"]), fmt_x((m5[g]["t_umfk"] / m5[g]["k"]) / (m5[g]["t_solvek"] / m5[g]["k"])))
                                            for g in ["roadNet-PA", "grid3d-40", "email-Enron"] if g in m5])
while lines and lines[-1] == "\\midrule": lines.pop()
if m5:
    macros.update(MMFactor=rng([m5[g]["t_umf"] / (m5[g]["t_sym"] + m5[g]["t_num"]) for g in m5]),
                  CholFactor=rng([(m5[g]["t_sym"] + m5[g]["t_num"]) / m5[g]["t_chol"] for g in m5 if m5[g].get("t_chol")]))
macros.setdefault("HasWidest", "0")
open("tables/tab-semirings.tex", "w").write("% generated by csv_to_tex.py\n" + "\n".join(lines) + "\n")

# ---------------- external baselines (GAP, PHAST, CCH) on the DIMACS road networks, from external.csv ----------------
# One core (alone in a NUMA region) and one 16-core region. Node figures are 8 regions; ratios between them cancel the 8.
P = 16
EXT = os.path.join(DATA, "external.csv")
if os.path.isfile(EXT):
    xr = list(csv.DictReader(open(EXT)))
    def xv(tool, g, placement, metric, variant=None):
        v = [float(r["value"]) for r in xr if r["tool"] == tool and r["graph"] == g and r["placement"] == placement
             and r["metric"] == metric and (variant is None or r["variant"] == variant)]
        return v
    XG = [g for g in ["USA-t-NY", "USA-t-BAY", "USA-t-COL", "USA-t-FLA", "USA-t-NW", "USA-t-NE", "USA-t-CAL", "USA-t-E", "USA-t-W", "USA-t-CTR", "USA-t-USA"]
          if xv("gap", g, "core", "latency_s") and qrow(g)]
    gap1 = {g: min(xv("gap", g, "core", "latency_s")) for g in XG}
    gapP = {g: min(xv("gap", g, "region", "latency_s") or [float("nan")]) for g in XG}
    gapQ = {g: max(xv("gap", g, "region-processes", "qps") or [float("nan")]) for g in XG}
    def ph(g, placement, kind):
        m = {"single": ["rank_k1_s", "level_k1_s"], "blocked": ["rank_k16_s", "rank_k64_s", "level_k16_s", "level_k64_s"], "tree": ["tree_s"]}[kind]
        v = [x for mm in m for x in xv("phast", g, placement, mm)]
        return min(v) if v else float("nan")
    ours16 = lambda g: E18_all[g][P] if g in E18_all and P in E18_all[g] else None
    best_qps = lambda r: max(1 / r[k] for k in ("tp_row_per_query", "lib_row_per_query", "tp_col_per_query", "lib_col_per_query"))
    rows = []; s1 = []; sb = []; sbP = []; sc = []; ge1 = []; geP = []; po = []; pl = []; pbk = []; gthr = []; pthr = []; gft = []
    for g in XG:
        r = qrow(g); full = g in e13
        e1 = e16[g]["elem_target_1"]["median"] if g in e16 else r["t1"]
        eP = e16p[g]["elem_target_nt"]["median"] if g in e16p else float("nan")
        p1, pb1, pP = ph(g, "core", "single"), ph(g, "core", "blocked"), ph(g, "region", "tree")
        short = g.replace("USA-t-", "")
        rows.append(f"{short} & {fmt_ms(gap1[g])} / {fmt_ms(gapP[g])} & {fmt_ms(p1)} / {fmt_ms(pP)} & {fmt_ms(e1)} / {fmt_ms(eP)} & "
                    f"{fmt_ms(pb1)} & " + (fmt_ms(r["tb"]) if full else "--") + " \\\\")
        s1.append(gap1[g] / r["t1"]); sc.append(gap1[g] / gapP[g]); ge1.append(gap1[g] / e1); geP.append(gapP[g] / eP)
        po.append(e1 / p1); pl.append(eP / pP)
        if full: sb.append(gap1[g] / r["tb"]); sbP.append(gapP[g] / r["tb"]); pbk.append(r["tb"] / pb1)
        o = ours16(g)
        if o:
            gthr.append(best_qps(o) / gapQ[g])
            pthr.append((1 / ph(g, "region", "blocked")) / best_qps(o))
        if eP == eP: gft.append((1 / eP) / gapQ[g])
    chb = [x for g in XG for x in xv("phast", g, "core", "ch_build_s")]
    _cch = [(sum(xv("cch", g, "core", mm)[0] for mm in ("order_s", "build_s", "customize_s")) / (e13[g]["t_sym"] + e13[g]["t_num"]))
            for g in XG if g in e13 and all(xv("cch", g, "core", mm) for mm in ("order_s", "build_s", "customize_s"))]
    if _cch: macros["CCHPrep"] = rng(_cch)
    cc = {g: (xv("cch", g, "core", "customize_s"), xv("cch", g, "core", "query_s")) for g in XG}
    open("tables/tab-external.tex", "w").write("% generated by csv_to_tex.py\n" + "\n".join(rows) + "\n")
    macros.update(HasExternal="1", HasPhastX="1", NExternal=str(len(XG)), ExtP=str(P),
                  GAPSingle=rng(s1), GAPSingleElem=rng(ge1), GAPParElem=rng(geP), GAPScale=rng(sc),
                  GAPBlocked=rng(sb), GAPBlockedPar=rng(sbP), GAPBlockPar=rng(gthr) if gthr else "??", GAPFastThru=rng(gft) if gft else "??",
                  PHASTOverUs=rng(po), PHASTLatRatio=rng(pl), PHASTParRatio=rng(pl), PHASTBlockedOverUs=rng(pbk),
                  PHASTBlockPar=rng(pthr) if pthr else "??",
                  CHBuild=rng(chb).replace(".0", "") if chb else "??",
                  OurSetupDIMACS=rng([e13[g]["t_sym"] + e13[g]["t_num"] for g in XG if g in e13]),
                  OurNumericDIMACS=rng([e13[g]["t_num"] * 1e3 for g in XG if g in e13]).replace(".0", ""),
                  RKCustomize=rng([cc[g][0][0] * 1e3 for g in XG if cc[g][0]]).replace(".0", "") if any(cc[g][0] for g in XG) else "??",
                  RKQuery=rng([cc[g][1][0] * 1e6 for g in XG if cc[g][1]]).replace(".0", "") if any(cc[g][1] for g in XG) else "??",
                  ElemFastGain=rng([e16[g]["dense_single"]["median"] / e16[g]["elem_target_1"]["median"] for g in ROADS if g in e16]),
                  GAPvsDij=rng([e16[g]["dijkstra"] / gap1[g] for g in XG if g in e16]))
else:
    macros.update(HasExternal="0")

# ---------------- comparison with the results reported for SuperFW (Sao et al. 2020) ----------------
# SuperFW's Fig. 6 labels: small graphs, speedups of SuperFW and of Dijkstra-APSP over BlockedFW; large graphs, speedup of
# SuperFW over Dijkstra-APSP. 32 Haswell cores, METIS preprocessing excluded. n/|S| from their Table 3.
SFW_SMALL = {"USpowerGrid": (32, 8.01, 6.2e2), "c-42": (15, 13, 1.5e2), "G67": (42, 16, 5.0e1), "delaunay_n14": (97, 26, 1.7e2),
             "fe_sphere": (65, 24, 8.5e1), "nd6k": (4.85, 1.86, 5.8), "OPF_6000": (123, 32, 1.4e3),
             "hypercube_14": (4.11, 13, 5.0), "rgg2d_14": (12, 1.96, 1.6e1), "rgg3d_14": (1.92, 0.30, 2.57),
             "EB_8192_256": (1.12, 0.93, 2.5), "EB_16384_64": (1.17, 3.91, 2.6)}    # EB labels overlap in their Fig. 6a: approximate
SFW_LARGE = {"c-69": 1.0, "delaunay_n16": 2.9, "email-Enron": 0.2, "fe_tooth": 1.7, "finan512": 7.2, "lpl1": 2.8,
             "luxembourg_osm": 1.2, "net4-1": 52, "oilpan": 5.1, "onera_dual": 1.0, "t60k": 3.4, "wing": 0.8}
SFW_LARGE_NS = {"c-69": 2.0e2, "delaunay_n16": 1.7e2, "email-Enron": 52, "fe_tooth": 88, "finan512": 1.5e3, "lpl1": 4.8e2,
                "luxembourg_osm": 6.7e3, "net4-1": 2.9e3, "oilpan": 1.7e2, "onera_dual": 1.5e2, "t60k": 1.1e3, "wing": 1.0e2}   # n/|S|, their Table 3
sfwname = lambda g, have: g + "+sfw" if (g + "+sfw") in have else g     # prefer runs with SuperFW's weight convention
rows = []; small_rows = []
n_sfw_weights = 0
for g, (sf, dj, ns) in SFW_SMALL.items():
    gg = sfwname(g, a)
    if gg not in a: continue
    n_sfw_weights += gg != g
    r = a[gg]; ours = r["t_dijkstra_all"] / (r["t_sym"] + r["t_num"] + r["t_closure"])
    wr = r["n"] ** 2 * (r["n"] / ns) / (r["n"] * r["nnzLU"])
    rows.append(f"{tex(g)} & {fmt_x(sf / dj)} & {fmt_x(ours)} & {wr:.1f} \\\\")
    small_rows.append((g, ours, sf / dj, wr))
rows.append("\\midrule")
wins = 0; nl = 0
large_work = {}; large_ours = {}
for g, sf in SFW_LARGE.items():
    gg = g + "+sfw" if qrow(g + "+sfw") else g
    r = qrow(gg)
    if not r: continue
    n_sfw_weights += gg != g
    ours = r["td"] / r["tb"]; nl += 1; wins += ours > sf
    wr = r["n"] ** 2 * (r["n"] / SFW_LARGE_NS[g]) / (r["n"] * r["nnz"]) if r.get("nnz") else float("nan")
    large_work[g] = wr; large_ours[g] = ours
    rows.append(f"{tex(g)} & {fmt_x(sf)} & {fmt_x(ours)} & {wr:.1f} \\\\")
open("tables/tab-superfw.tex", "w").write("% generated by csv_to_tex.py\n" + "\n".join(rows) + "\n")
win = [r for r in small_rows if r[1] > r[2]]; lose = [r for r in small_rows if r[1] <= r[2]]
macros.update(SFWSmallN=str(len(small_rows)), SFWSmallWins=str(len(win)),
              SFWAdv=rng([o / t for _, o, t, _ in win]), SFWWork=rng([w for _, _, _, w in win if w > 1]),
              SFWSmallLosers=(", ".join(tex(g) for g, *_ in lose) or "none"),
              SFWWorkAbove=(", ".join(tex(g) for g, _, _, w in small_rows if w <= 1) or "none"),
              SFWAdvSmallSep=rng([o / t for g, o, t, _ in small_rows if g in ("USpowerGrid", "c-42", "G67", "delaunay_n14", "fe_sphere")]),
              SFWAdvLarge=rng([large_ours[g] / SFW_LARGE[g] for g in large_ours if large_ours[g] > SFW_LARGE[g]]),
              SFWLargeWins=str(wins), SFWLargeN=str(nl), SFWWeightsTheirs=str(n_sfw_weights),
              SFWNetWork=f'{1 / large_work["net4-1"]:.1f}' if "net4-1" in large_work else "??",
              SFWLargeWorkBelow=str(sum(w > 1 for w in large_work.values())))

# ---------------- scaling (optional): exp7c (factor, column-layout solves, threaded Dijkstra, closure) + exp12 (row layout)
if SCALE:
    sc = [J(f) for f in SG("exp7c_*.json")]
    sd = [J(f) for f in SG("exp12_*.json")]
    E18 = {}                                  # subtree-parallel scaling runs (exp18), keyed by graph then threads
    for f in SG("exp18_*.json"):
        r = J(f); E18.setdefault(r["graph"], {})[r["nt"]] = r
    C = {}; Dd = {}
    for r in sc: C.setdefault((r["graph"], r["ordering"]), {})[r["nt"]] = r
    for r in sd: Dd.setdefault((r["graph"], r["ordering"]), {})[r["nt"]] = r
    QG = [g for g in ["USA-t-FLA", "roadNet-PA", "belgium_osm"] if (g, "AMF") in C or g in E18]
    if E18 and not any(g in E18 for g in QG): QG = sorted(E18)[:3]
    QGC = [g for g in QG if (g, "AMF") in C]              # graphs with exp7c data (factorization, closure, old solves)
    with open("data/scal_q_plots.tex", "w") as f:
        for ci, g in enumerate(QG):
            if g in E18:
                rs = E18[g]
                with open(f"data/scal_q_{g}.dat", "w") as h:
                    h.write("nt dijkstra col row\n")      # row := best blocked path; col := single queries, one at a time
                    for t in sorted(rs):
                        best = max(1 / rs[t][k] for k in ("tp_row_per_query", "lib_row_per_query", "tp_col_per_query", "lib_col_per_query"))
                        h.write(f"{t} {rs[t]['qps_dijkstra']:.4g} {1 / rs[t]['fast_single']:.4g} {best:.4g}\n")
                col = f"c{ci}"
                f.write(f"\\addplot[{col}, mark=*, mark size=1.3pt, line width=0.9pt] table[x=nt, y=row] {{data/scal_q_{g}.dat}};\\addlegendentry{{{tex(g)}}}\n")
                f.write(f"\\addplot[{col}, mark=o, mark size=1.3pt, densely dotted, line width=0.8pt, forget plot] table[x=nt, y=col] {{data/scal_q_{g}.dat}};\n")
                f.write(f"\\addplot[{col}, mark=none, dashed, line width=0.8pt, forget plot] table[x=nt, y=dijkstra] {{data/scal_q_{g}.dat}};\n")
                continue
            rs = C[(g, "AMF")]; ds = Dd.get((g, "AMF"), {})
            with open(f"data/scal_q_{g}.dat", "w") as h:
                h.write("nt dijkstra col row\n")
                for t in sorted(rs):
                    row = 1 / ds[t]["blocked_row_layout_per_query"] if t in ds else float("nan")
                    h.write(f"{t} {rs[t]['qps_dijkstra']:.4g} {rs[t]['qps_solve']:.4g} {row:.4g}\n")
            col = f"c{ci}"
            f.write(f"\\addplot[{col}, mark=*, mark size=1.3pt, line width=0.9pt] table[x=nt, y=col] {{data/scal_q_{g}.dat}};\\addlegendentry{{{tex(g)}}}\n")
            f.write(f"\\addplot[{col}, mark=o, mark size=1.3pt, densely dotted, line width=0.8pt, forget plot] table[x=nt, y=row, unbounded coords=discard] {{data/scal_q_{g}.dat}};\n")
            f.write(f"\\addplot[{col}, mark=none, dashed, line width=0.8pt, forget plot] table[x=nt, y=dijkstra] {{data/scal_q_{g}.dat}};\n")
    SG = [("roadNet-PA", "t_num", "factor.: roadNet-PA", "c0", "solid"), ("grid2d-1000", "t_num", "factor.: grid2d-1000", "c1", "solid"),
          ("grid3d-40", "t_num", "factor.: grid3d-40", "c2", "solid"), ("email-Enron", "t_num", "factor.: email-Enron", "c3", "solid"),
          ("c-42", "t_closure", "closure: c-42", "c4", "dashed"), ("delaunay_n14", "t_closure", "closure: delaunay\\_n14", "c5", "dashed")]
    with open("data/scal_s_plots.tex", "w") as f:
        for g, key, lab, col, sty in [(g, k2, l2, c2, s2) for g in ["USA-t-FLA", "roadNet-PA"] if g in E18 and 1 in E18[g]
                                      for (k2, l2, c2, s2) in (("fast_single", f"one query: {tex(g)}", "c6", "densely dashdotted"),
                                                               ("dense_single_tp", f"dense rhs: {tex(g)}", "c7", "densely dotted"))][:2]:
            rs = E18[g]; b = rs[1]
            with open(f"data/scal_s_{g}_{key}.dat", "w") as h:
                h.write("nt speedup\n")
                for t in sorted(rs): h.write(f"{t} {b[key] / rs[t][key]:.4g}\n")
            f.write(f"\\addplot[{col}, mark=square*, mark size=1.2pt, {sty}, line width=0.9pt] table[x=nt, y=speedup] {{data/scal_s_{g}_{key}.dat}};\\addlegendentry{{{lab}}}\n")
        for g, key, lab, col, sty in SG:
            if (g, "AMF") not in C: continue
            rs = C[(g, "AMF")]; b = rs[min(rs)]
            if not b.get(key): continue
            with open(f"data/scal_s_{g}_{key}.dat", "w") as h:
                h.write("nt speedup\n")
                for t in sorted(rs): h.write(f"{t} {b[key] / rs[t][key]:.4g}\n")
            f.write(f"\\addplot[{col}, mark=*, mark size=1.2pt, {sty}, line width=0.9pt] table[x=nt, y=speedup] {{data/scal_s_{g}_{key}.dat}};\\addlegendentry{{{lab}}}\n")
    macros.update(ScaleSolidLabel="blocked, best layout" if E18 else "blocked, column layout",
                  ScaleDottedLabel="single queries, one at a time" if E18 else "blocked, row layout")
    P = 16                                       # the cores of one NUMA region
    if E18:
        EG = [g for g in ["USA-t-FLA", "roadNet-PA", "belgium_osm"] if g in E18 and P in E18[g] and 1 in E18[g]]
        bq = lambda r: max(1 / r[k] for k in ("tp_row_per_query", "lib_row_per_query", "tp_col_per_query", "lib_col_per_query"))
        row1 = lambda r: max(1 / r["tp_row_per_query"], 1 / r["lib_row_per_query"])
        macros.update(TPRatio=rng([bq(E18[g][P]) / E18[g][P]["qps_dijkstra"] for g in EG]),
                      TPRowOne=rng([row1(E18[g][1]) / E18[g][1]["qps_dijkstra"] for g in EG]),
                      TPBestGain=rng([bq(E18[g][P]) / bq(E18[g][1]) for g in EG]),
                      TPFastVsDij=rng([(1 / E18[g][P]["fast_single"]) / E18[g][P]["qps_dijkstra"] for g in EG]),
                      TPRowScale=rng([E18[g][1]["tp_row_per_query"] / E18[g][P]["tp_row_per_query"] for g in EG]),
                      ScaleBlockK=rng([E18[g][1]["k"] for g in EG]).replace(".0", ""),
                      NYRowScale=fmt_x(E18["USA-t-NY"][1]["tp_row_per_query"] / E18["USA-t-NY"][P]["tp_row_per_query"]) if "USA-t-NY" in E18 and P in E18["USA-t-NY"] else "??",
                      NYBlockK=str(E18["USA-t-NY"][1]["k"]) if "USA-t-NY" in E18 else "??",
                      EnronRowScale=fmt_x(E18["email-Enron"][1]["tp_row_per_query"] / E18["email-Enron"][P]["tp_row_per_query"]) if "email-Enron" in E18 and P in E18["email-Enron"] else "??",
                      TPFastScale=rng([E18[g][1]["fast_single"] / E18[g][P]["fast_single"] for g in EG]),
                      TPDenseScale=rng([E18[g][1]["dense_single_tp"] / E18[g][P]["dense_single_tp"] for g in EG]))
    at = lambda g, t: C[(g, "AMF")][t]
    macros.update(ScaleP=str(P),
        ScaleRatio=rng([at(g, P)["qps_solve"] / at(g, P)["qps_dijkstra"] for g in QGC]),
        ScaleSolve=rng([at(g, P)["qps_solve"] / at(g, 1)["qps_solve"] for g in QGC]),
        ScaleDijkstra=rng([at(g, P)["qps_dijkstra"] / at(g, 1)["qps_dijkstra"] for g in QGC]),
        ScaleRowOne=rng([1 / Dd[(g, "AMF")][1]["blocked_row_layout_per_query"] / at(g, 1)["qps_dijkstra"] for g in QGC if (g, "AMF") in Dd]),
        ScaleRowP=rng([1 / Dd[(g, "AMF")][P]["blocked_row_layout_per_query"] / at(g, P)["qps_dijkstra"] for g in QGC if (g, "AMF") in Dd]),
        ScaleFactorRoad=rng([max(at(g, 1)["t_num"] / at(g, t)["t_num"] for t in C[(g, "AMF")] if t <= P) for g in QGC]),
        ScaleFactorMesh=rng([max(at(g, 1)["t_num"] / at(g, t)["t_num"] for t in C[(g, "AMF")] if t <= P) for g in ["grid2d-1000", "grid3d-40"] if (g, "AMF") in C]),
        ScaleFactorEnron=fmt_x(max(at("email-Enron", 1)["t_num"] / at("email-Enron", t)["t_num"] for t in C[("email-Enron", "AMF")] if t <= P)) if ("email-Enron", "AMF") in C else "??",
        ScaleClosure=rng([at(g, 1)["t_closure"] / at(g, P)["t_closure"] for g in ["c-42", "delaunay_n14"] if (g, "AMF") in C and at(g, 1).get("t_closure")]),
        ScaleRoadFactorTime=rng([at(g, 1)["t_num"] for g in QGC]))

# ---------------- schematic (7x7 grid, nested dissection) ----------------
kk = 7; V = [(x, y) for y in range(kk) for x in range(kk)]; vid = {v: i for i, v in enumerate(V)}
adj = {i: set() for i in range(kk * kk)}
for (x, y) in V:
    for dx, dy in ((1, 0), (0, 1)):
        u = (x + dx, y + dy)
        if u in vid: adj[vid[(x, y)]].add(vid[u]); adj[vid[u]].add(vid[(x, y)])
groups = [("cA", [(x, y) for x in range(3) for y in range(3)]), ("cA", [(x, y) for x in range(3) for y in range(4, 7)]),
          ("cA", [(x, 3) for x in range(3)]), ("cB", [(x, y) for x in range(4, 7) for y in range(3)]),
          ("cB", [(x, y) for x in range(4, 7) for y in range(4, 7)]), ("cB", [(x, 3) for x in range(4, 7)]), ("cR", [(3, y) for y in range(7)])]
# classes: the independent subtrees A (A1, A2, A3) and B (B1, B2, B3) of the clique tree, and its root R
order = [vid[v] for _, g in groups for v in g]; pos = {v: i for i, v in enumerate(order)}
cls = {vid[v]: c for c, g in groups for v in g}
N = kk * kk; nb = {i: set(adj[i]) for i in adj}; P = set()
for v in order:
    later = [u for u in nb[v] if pos[u] > pos[v]]
    for u in later:
        P.add((pos[u], pos[v]))
        for w in later:
            if w != u: nb[u].add(w)
gname = ["Aone", "Atwo", "SA", "Bone", "Btwo", "SB", "S"]; bounds = []; start = 0
for _, gv in groups: bounds.append((start, start + len(gv))); start += len(gv)
with open("data/schematic_sizes.tex", "w") as f:
    f.write("% separator (a) and supernode (n) sizes of the 7x7 example, generated by csv_to_tex.py\n")
    for nm, (lo, hi) in zip(gname, bounds):
        a_ = len({i for (i, j) in P if lo <= j < hi and i >= hi})
        f.write(f"\\def\\n{nm}{{{hi - lo}}}\\def\\a{nm}{{{a_}}}\n")
with open("data/schematic_grid.tex", "w") as f:
    for i in adj:
        for j in adj[i]:
            if i < j: f.write(f"\\draw[edge] ({V[i][0]},{V[i][1]}) -- ({V[j][0]},{V[j][1]});\n")
    for i, (x, y) in enumerate(V): f.write(f"\\node[vtx,fill={cls[i]}] at ({x},{y}) {{}};\n")
with open("data/schematic_pattern.tex", "w") as f:
    for i in range(N):
        f.write(f"\\fill[{cls[order[i]]}] ({i},{-i}) rectangle ++(1,-1);\n")
    for (i, j) in sorted(P):     # i > j: L entry (i, j) and U entry (j, i)
        c = cls[order[j]]
        f.write(f"\\fill[{c}] ({j},{-i}) rectangle ++(1,-1);\n\\fill[{c}!45] ({i},{-j}) rectangle ++(1,-1);\n")
    f.write(f"\\draw[black!60, line width=0.3pt] (0,0) rectangle ({N},{-N});\n")

# ---------------- methodology numbers ----------------
_rg = e16.get("rgg2d-1M")
if _rg: macros["RelabelRgg"] = fmt_x(_rg["dijkstra_original"]["median"] / _rg["dijkstra_elimination_order"]["median"])
# region independence: eight regions at once (regions.csv) vs one region alone (scaling.csv), same graph and threads
if os.path.isfile(REGIONS):
    _best = lambda r: max(1 / float(r[k]) for k in ("tp_row_per_query", "lib_row_per_query", "tp_col_per_query", "lib_col_per_query"))
    _rr = list(csv.DictReader(open(REGIONS)))
    if _rr:
        _g, _t = _rr[0]["graph"], int(_rr[0]["nt"])
        _alone = E18_all.get(_g, {}).get(_t)
        if _alone:
            _a = _best({k: _alone[k] for k in ("tp_row_per_query", "lib_row_per_query", "tp_col_per_query", "lib_col_per_query")})
            _dev = max(abs(_best(r) / _a - 1) for r in _rr)
            macros.update(RegionDev=f"{100 * _dev:.1f}", RegionGraph=tex(_g))

with open("data/macros.tex", "w") as f:
    f.write("% generated by csv_to_tex.py -- do not edit\n")
    for key, v in sorted(macros.items()): f.write(f"\\newcommand{{\\{key}}}{{{v}}}\n")
print("\n".join(f"{k_} = {v}" for k_, v in sorted(macros.items())))

