#!/usr/bin/env python3
"""Derived numbers (LaTeX macros), tables, and figure data, from the CPU benchmark CSVs (data/*.csv).

Usage: cpu_data.py DATA_DIR OUT_DIR [COMMIT]
Writes OUT_DIR/data/results.tex (macros), OUT_DIR/data/*.dat (figure data), OUT_DIR/tables/*.tex.
Only rows measured at COMMIT (default: the most common commit in apsp.csv) are used; anything missing is reported."""
import os, sys, math
import pandas as pd, numpy as np

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "data"), exist_ok=True); os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
warn = lambda m: print("cpu_data: WARNING:", m, file=sys.stderr)
read = lambda f: pd.read_csv(os.path.join(DATA, f)) if os.path.isfile(os.path.join(DATA, f)) else None

# ---------------- the graph sets, in report order ----------------
APSP = ["USpowerGrid", "delaunay_n13", "G67", "c-42", "CurlCurl_0", "fe_sphere", "nd6k", "delaunay_n14", "OPF_6000",
        "EB_8192_256", "EB_16384_64", "rgg2d_14", "rgg3d_14", "hypercube_14", "lpl1", "email-Enron", "t60k", "wing",
        "delaunay_n16", "c-69", "oilpan", "finan512", "fe_tooth", "onera_dual", "net4-1", "luxembourg_osm"]
tex = lambda g: g.replace("_", r"\_")

# ---------------- all-pairs shortest paths ----------------
a = read("apsp.csv")
a["nt"] = a["file"].str.extract(r"_nt(\d+)\.json").astype(int)
COMMIT = sys.argv[3] if len(sys.argv) > 3 else a["commit"].mode()[0]
COMMITS = set(COMMIT.split(","))                                   # several: commits that differ only in excluded phases
other = a[~a.commit.isin(COMMITS)]
if len(other): warn(f"apsp.csv: ignoring {len(other)} rows not at {COMMIT}: " + ", ".join(sorted({f'{g}@nt{n}' for g, n in zip(other.graph, other.nt)})))
if len(COMMITS) > 1: warn(f"apsp.csv: rows from {len(COMMITS)} commits treated as equivalent: " + ", ".join(f"{c} ({(a.commit == c).sum()} rows)" for c in sorted(COMMITS)))
a = a[a.commit.isin(COMMITS)].copy()
# timing convention: the output buffer is provided before timing, as for the baselines (our Dijkstra writes into a matrix
# allocated before its timer; GAP fills its own per-run vector): first-touch allocation (t_alloc) and renumbering (t_perm)
# are excluded from totals and reported apart
a["ours"] = a.t_sym + a.t_num + a.t_closure

e = read("external.csv")
ga = e[(e.tool == "gap_apsp") & (e.metric == "latency_s")].copy()
ga["value"] = ga.value.astype(float)
ga["full"] = ga.variant.fillna("").str.contains("runs")                      # n runs measured (not a sample)
ga = ga[ga.full | ~ga.groupby(["graph", "placement"]).full.transform("any")]   # prefer full runs where they exist
if (~ga.full).any(): warn("gap_apsp: sampled (not full) runs used for: " + ", ".join(sorted(set(ga[~ga.full].graph))))
gap1 = ga[ga.placement == "core"].groupby("graph").value.mean()                # one core: time per source
gapP = ga[ga.placement == "processes"].groupby(["graph", "threads"]).value.mean()  # P concurrent single-thread runs
gfull = e[(e.tool == "gap_apsp") & (e.metric == "total_s")].copy(); gfull["value"] = gfull.value.astype(float)
gfull = gfull.groupby(["graph", "threads"]).value.min()                      # all n runs, writing the n x n output
old_gap = set()
def gap_all(g, n, nt):                                                       # n single-source runs on nt cores
    if (g, nt) in gfull.index: return gfull[(g, nt)]
    old_gap.add(g)                                                           # older runs: no output matrix written
    v = gap1.get(g, np.nan) if nt == 1 else gapP.get((g, nt), np.nan) / nt
    return v * n
a["gap"] = [gap_all(g, n, nt) for g, n, nt in zip(a.graph, a.n, a.nt)]
if old_gap: warn("gap_apsp: no full-matrix run (output not written) for: " + ", ".join(sorted(old_gap)))

rows = []
for g in APSP:
    r1 = a[(a.graph == g) & (a.nt == 1)]; r16 = a[(a.graph == g) & (a.nt == 16)]
    if r1.empty or r16.empty: warn(f"apsp: {g} missing at 1 or 16 threads (skipped)"); continue
    r1, r16 = r1.iloc[0], r16.iloc[0]
    rows.append(dict(graph=g, n=int(r1.n), fill=r1.fill, ncc=int(r1.ncc),
                     ours1=r1.ours, ours16=r16.ours, dij1=r1.t_dijkstra_all, dij16=r16.t_dijkstra_all, gap1=r1.gap, gap16=r16.gap,
                     inv1=r1.t_inverse, sol1=r1.t_solve, clo1=r1.t_closure, clo16=r16.t_closure, alloc1=r1.t_alloc, sym1=r1.t_sym, num1=r1.t_num, perm1=r1.t_perm))
A = pd.DataFrame(rows)
for b in ("dij", "gap"):
    for t in ("1", "16"): A[f"x{b}{t}"] = A[f"{b}{t}"] / A[f"ours{t}"]

macros = {}
fx = lambda x: f"{x:.1f}" if x < 9.95 else f"{x:.0f}"                         # 2.4, 13
rng = lambda s: f"{fx(s.min())}--{fx(s.max())}" if len(s) else "--"
macros["ApspGraphs"] = str(len(A))
for b, name in (("gap", "Delta"), ("dij", "Dij")):
    for t, tn in (("1", "One"), ("16", "Sixteen")):
        x = A[f"x{b}{t}"].dropna(); w = x[x > 1]
        macros[f"Apsp{tn}{name}Wins"] = str(len(w)); macros[f"Apsp{tn}{name}Of"] = str(len(x))
        macros[f"Apsp{tn}{name}Range"] = rng(w)
        macros[f"Apsp{tn}{name}Median"] = fx(x.median()); macros[f"Apsp{tn}{name}Max"] = fx(x.max())
big = A[(A.n > 50000) & (A.ncc == 1)]
macros["ClosureScalingRange"] = rng(big.clo1 / big.clo16)
macros["SolveShareRange"] = f"{100 * (big.sol1 / big.clo1).min():.0f}--{100 * (big.sol1 / big.clo1).max():.0f}"
s32 = a[a.nt == 32]
if len(s32):                                                                  # what allocation would add at 32 threads
    sh = s32.t_alloc / (s32.ours + s32.t_alloc)
    macros["AllocShareThirtyTwo"] = f"{100 * sh.min():.0f}--{100 * sh.max():.0f}"

# scaling (SuperFW's strong-scaling graphs): time against threads, ours and both baselines on the same cores
SCALE = ["finan512", "net4-1", "email-Enron", "wing"]
up, reg2, lead = [], [], []
for i, G in enumerate(SCALE):
    sg = a[a.graph == G].sort_values("nt")
    if sg.empty: warn(f"scaling: {G} missing"); continue
    with open(os.path.join(OUT, "data", f"scaling_{i}.dat"), "w") as f:
        f.write("nt ours closure dij gap\n")
        for r in sg.itertuples(): f.write(f"{r.nt} {r.ours:.4g} {r.t_closure:.4g} {r.t_dijkstra_all:.4g} {r.gap:.4g}\n")
    c = sg.set_index("nt")
    if 1 in c.index and 16 in c.index: up.append(c.t_closure[1] / c.t_closure[16])
    if 16 in c.index and 32 in c.index: reg2.append(c.t_closure[16] / c.t_closure[32])
    if 64 in c.index: lead.append((G, c.gap[64] / c.ours[64]))
macros["ScalingNames"] = ",".join(tex(G) for G in SCALE)
if up: macros["ScaleOneRegion"] = f"{fx(min(up))}--{fx(max(up))}"
if reg2: macros["ScaleSecondRegion"] = f"{min(reg2):.1f}--{max(reg2):.1f}"
g64 = [c for c in []]
for G, x in lead: macros["LeadSixtyFour" + "".join(ch for ch in G.title() if ch.isalpha())] = fx(x)
gs = [(a[(a.graph == G) & (a.nt == 1)].gap.iloc[0] / a[(a.graph == G) & (a.nt == 64)].gap.iloc[0]) for G in SCALE if ((a.graph == G) & (a.nt == 64)).any()]
if gs: macros["GapScaleSixtyFour"] = f"{fx(min(gs))}--{fx(max(gs))}"

# figure data: speedups over n single-source runs, per graph, x = position
with open(os.path.join(OUT, "data", "apsp_speedup.dat"), "w") as f:
    # one yardstick for every bar: n runs of Dijkstra on one core, divided by each method's time
    f.write("x graph ours1 gap1 ours16 gap16\n")
    for i, r in A.reset_index(drop=True).iterrows():
        q = lambda t: f"{r.dij1 / t:.4g}" if np.isfinite(t) and t > 0 else "nan"
        f.write(f"{i} {tex(r.graph)} {q(r.ours1)} {q(r.gap1)} {q(r.ours16)} {q(r.gap16)}\n")
top = np.nanmax(np.concatenate([(A.dij1 / A[c]).values for c in ("ours1", "gap1", "ours16", "gap16")]))
macros["ApspYmax"] = f"{top * 1.6:.3g}"                                           # axis limits follow the data
bot = np.nanmin(np.concatenate([(A.dij1 / A[c]).values for c in ("ours1", "gap1", "ours16", "gap16")]))
macros["ApspYmin"] = f"{min(bot / 1.6, 0.5):.2g}"
macros["ApspXticks"] = ",".join(str(i) for i in range(len(A)))
macros["ApspXlabels"] = ",".join("{" + tex(g) + "}" for g in A.graph)

# STREAM triad: one core and one NUMA region
st = e[(e.tool == "stream") & (e.metric == "triad_GBps")].copy(); st["value"] = st.value.astype(float)
for t, name in ((1, "StreamOne"), (16, "StreamSixteen")):
    v = st[st.threads == t].value
    if len(v): macros[name] = f"{v.mean():.1f}"

# ---------------- bottleneck paths ----------------
bt = read("bottleneck.csv")
if bt is not None:
    bt = bt[bt.commit.isin(COMMITS)] if "commit" in bt else bt
    bt["ours"] = bt.t_sym + bt.t_num + bt.t_closure                            # same convention as above
    bo = e[(e.tool == "boost_widest") & (e.metric == "latency_s")].copy(); bo["value"] = bo.value.astype(float)
    boost = bo.groupby("graph").value.mean()
    chk = e[(e.tool == "boost_widest") & (e.metric == "check_pass")]
    if len(chk) and (chk.value.astype(float) < 1).any(): warn("Boost's widest paths disagree with our closure on: " + ", ".join(chk[chk.value.astype(float) < 1].graph))
    B = []
    for g in APSP:
        r = bt[bt.graph == g]
        if r.empty: continue
        r = r.iloc[0]; bb = boost.get(g, np.nan) * r.n
        tro = A[A.graph == g]
        B.append(dict(graph=g, n=int(r.n), ours=r.ours, dij=r.t_dijkstra_all, boost=bb, trop=tro.ours1.iloc[0] if len(tro) else np.nan))
    B = pd.DataFrame(B); B["xdij"] = B.dij / B.ours; B["xboost"] = B.boost / B.ours
    w = B[B.xdij > 1]
    macros.update(BottleneckGraphs=str(len(B)), BottleneckWins=str(len(w)), BottleneckRange=rng(w.xdij), BottleneckBoostRange=rng(B[B.xboost > 1].xboost.dropna()))
    macros.update(BottleneckMedian=fx(B.xdij.median()), BottleneckMax=fx(B.xdij.max()), BottleneckBoostMax=fx(B.xboost.max()))
    big = B[B.n > 50000]
    dev = (big.ours / big.trop - 1).abs()
    macros["BottleneckVsTropical"] = f"{100 * dev.max():.0f}"
    # table: two halves side by side
    lines = ["% generated by scripts/cpu_data.py", r"\begin{tabular}{@{}lrrrr@{\hspace{1.2em}}lrrrr@{}}", r"\toprule",
             r"Graph & ours & Dijk. & Boost & $\times$ & Graph & ours & Dijk. & Boost & $\times$ \\", r"\midrule"]
    t = lambda x: "--" if not np.isfinite(x) else (f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}")
    half = math.ceil(len(B) / 2); L = B.iloc[:half].reset_index(drop=True); R = B.iloc[half:].reset_index(drop=True)
    for i in range(half):
        cell = lambda r: f"{tex(r.graph)} & {t(r.ours)} & {t(r.dij)} & {t(r.boost)} & {fx(r.xdij)}"
        lines.append(cell(L.loc[i]) + " & " + (cell(R.loc[i]) if i < len(R) else "& & & &") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tables", "tab-bottleneck.tex"), "w").write("\n".join(lines) + "\n")

# ---------------- queries: DIMACS road networks ----------------
q = read("queries.csv")
if q is not None:
    q = q[q.commit.isin(COMMITS)]
    gq = e[(e.tool == "gap_query") & (e.metric == "latency_s")].copy(); gq["value"] = gq.value.astype(float)
    if gq.commit.isin(COMMITS).any(): gq = gq[gq.commit.isin(COMMITS)]
    gapq = gq.groupby(["graph", "threads"]).value.mean()
    ph = e[e.tool == "phast"].copy(); ph["value"] = ph.value.astype(float)
    php = ph.pivot_table(index=["graph", "threads"], columns="metric", values="value")
    def phast(g, t, k):                                          # PHAST per tree: the better of its two vertex orders
        try: return min(php.loc[(g, t), f"rank_k{k}_s"], php.loc[(g, t), f"level_k{k}_s"])
        except KeyError: return np.nan
    cch = e[e.tool == "cch"].copy(); cch["value"] = cch.value.astype(float)
    cchp = cch.pivot_table(index="graph", columns="metric", values="value")
    roads = q[q.graph.str.startswith("USA-t-")].sort_values("n")
    def best_block(r, nt):
        v = [float(r[c]) for c in r.index if c.startswith("blocks.") and c.split(".")[1].endswith(f"_nt{nt}") and pd.notna(r[c])]
        return min(v) if v else np.nan
    Q = []
    for _, r in roads.iterrows():
        g = r.graph
        Q.append(dict(graph=g, name=g[6:], n=int(r.n), ours1=r["single.source_nt1.median"], ours16=r["single.source_nt16.median"],
                      gap1=gapq.get((g, 1), np.nan), gap16=gapq.get((g, 16), np.nan), ph1=phast(g, 1, 1), ph16=phast(g, 16, 1),
                      blk1=best_block(r, 1), blk16=best_block(r, 16), phb1=min(phast(g, 1, 16), phast(g, 1, 64)), phb16=min(phast(g, 16, 16), phast(g, 16, 64)),
                      prep=r.t_sym + r["t_num.1"], ch=php.loc[(g, 1), "ch_build_s"] if (g, 1) in php.index else np.nan,
                      cch=cchp.loc[g, ["order_s", "build_s", "customize_s"]].sum() if g in cchp.index else np.nan))
    Q = pd.DataFrame(Q)
    r_ = lambda x: f"{fx(x.min())}--{fx(x.max())}"
    macros.update(QueryNets=str(len(Q)), QueryOneDeltaRange=r_(Q.gap1 / Q.ours1), QuerySixteenDeltaRange=r_(Q.gap16 / Q.ours16),
                  QueryBlockDeltaRange=r_(Q.gap1 / Q.blk1), PhastSingleRange=r_(Q.ours1 / Q.ph1), PhastSixteenRange=r_(Q.ours16 / Q.ph16),
                  PhastBlockRange=r_(Q.blk1 / Q.phb1), PrepCHRange=r_(Q.ch / Q.prep), PrepCCHRange=r_(Q.cch / Q.prep))
    ms = lambda x: "--" if not np.isfinite(x) else (f"{1e3 * x:.2f}" if x < 0.01 else f"{1e3 * x:.1f}" if x < 0.1 else f"{1e3 * x:.0f}")
    sec = lambda x: "--" if not np.isfinite(x) else (f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}")
    lines = ["% generated by scripts/cpu_data.py", r"\begin{tabular}{@{}lr@{\hspace{1em}}rrrrrr@{\hspace{1em}}rrrr@{\hspace{1em}}rrr@{}}", r"\toprule",
             r" & & \multicolumn{6}{c}{One query (ms)} & \multicolumn{4}{c}{Per query in a block (ms)} & \multicolumn{3}{c}{Preprocessing (s)} \\",
             r"\cmidrule(lr){3-8}\cmidrule(lr){9-12}\cmidrule(l){13-15}",
             r" & & \multicolumn{2}{c}{ours} & \multicolumn{2}{c}{$\Delta$-step.} & \multicolumn{2}{c}{PHAST} & \multicolumn{2}{c}{ours} & \multicolumn{2}{c}{PHAST} & & & \\",
             r"Network & $n$ & 1 & 16 & 1 & 16 & 1 & 16 & 1 & 16 & 1 & 16 & ours & CH & CCH \\", r"\midrule"]
    for _, r in Q.iterrows():
        lines.append(f"{r['name']} & {r.n / 1e6:.2f}M & " + " & ".join(ms(r[c]) for c in ("ours1", "ours16", "gap1", "gap16", "ph1", "ph16", "blk1", "blk16", "phb1", "phb16"))
                     + " & " + " & ".join(sec(r[c]) for c in ("prep", "ch", "cch")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tables", "tab-queries.tex"), "w").write("\n".join(lines) + "\n")
    # reuse figure: speedup per query over one Delta-stepping run on the same cores, against the block size k
    ks = [1, 8, 32, 128, 256, 512, 1024]
    with open(os.path.join(OUT, "data", "reuse.dat"), "w") as f:
        f.write("k " + " ".join(f"s1_{i} s16_{i}" for i in range(len(Q))) + "\n")
        for k in ks:
            vals = []
            for i, (_, r) in enumerate(roads.iterrows()):
                for nt, gref in ((1, Q.gap1.iloc[i]), (16, Q.gap16.iloc[i])):
                    c = [float(r[col]) for col in r.index if col.startswith(f"blocks.k{k}_nt{nt}.") and pd.notna(r[col])]
                    if k == 1: c.append(float(r[f"single.source_nt{nt}.median"]))   # one query: the single-query fast path
                    vals.append(f"{gref / min(c):.4g}" if c else "nan")
            f.write(f"{k} " + " ".join(vals) + "\n")
    macros["ReuseLast"] = str(len(Q) - 1)
    macros["PhastRefOne"] = f"{(Q.gap1 / Q.phb1).median():.3g}"; macros["PhastRefSixteen"] = f"{(Q.gap16 / Q.phb16).median():.3g}"

# ---------------- dense kernels ----------------
k = read("kernels.csv")
if k is not None:
    k = k[k.commit.isin(COMMITS)] if "commit" in k else k
    rate = lambda s, T, n: k[(k.semiring == s) & (k["T"] == T) & (k.n == n)].gmuladd.mean()
    mp, sp, dg = rate("MinPlus", "Float64", 1024), rate("MinPlus", "Float32", 1024), rate("dgemm", "Float64", 1024)
    macros.update(MinPlusRate=f"{mp:.1f}", MinPlusSPRate=f"{sp:.1f}", DgemmRate=f"{dg:.1f}", MinPlusPct=f"{100 * mp / dg:.0f}")
    sg_ = rate("sgemm", "Float32", 1024)
    if np.isfinite(sg_): macros.update(SgemmRate=f"{sg_:.1f}", MinPlusSPPct=f"{100 * sp / sg_:.0f}")
    macros["KernelsHaveSgemm"] = "1" if (k.semiring == "sgemm").any() else "0"  # sgemm: measured from the next run on
    plotted = k[k.semiring.isin(["dgemm", "MinPlus", "MaxMin"]) & (k["T"] == "Float64")]          # panel (a): Float64 only
    macros["KernelYmax"] = f"{10 * math.ceil(plotted.gmuladd.max() * 1.1 / 10):g}"
    with open(os.path.join(OUT, "data", "kernels.dat"), "w") as f:              # rate against matrix size, one core
        f.write("n minplus64 minplus32 maxmin64 dgemm sgemm\n")
        for n in sorted(k.n.unique()):
            f.write(f"{n} {rate('MinPlus', 'Float64', n):.3f} {rate('MinPlus', 'Float32', n):.3f} {rate('MaxMin', 'Float64', n):.3f} {rate('dgemm', 'Float64', n):.3f} {rate('sgemm', 'Float32', n):.3f}\n")

macros["DataCommit"] = COMMIT
with open(os.path.join(OUT, "data", "results.tex"), "w") as f:
    f.write("% generated by scripts/cpu_data.py from " + DATA + f" (library commit {COMMIT})\n")
    for kk, v in macros.items(): f.write(f"\\def\\{kk}{{{v}}}\n")
print(f"cpu_data: {len(macros)} macros, {len(A)} all-pairs graphs, commit {COMMIT}")
for kk, v in macros.items():
    if not kk.startswith("ApspX"): print(f"  {kk} = {v}")
