#!/usr/bin/env python3
"""Figure data for the all-pairs comparison (fig-apsp-combined): all-pairs shortest paths on CPUs and on one GPU, on
one yardstick, the speedup over n runs of Dijkstra's algorithm on one core.

Usage: [GPU_ROME_RULE=final|best] fig_apsp_combined.py [DATA_DIR [OUT_DIR [GPU_COMMIT [CPU_COMMITS]]]]
  defaults: data, build/figures, lib-g, lib-b,lib-a (the Makefile's GPU_COMMIT and CPU_COMMITS); GPU_HEAD from the
  environment, default ours-auto (as the Makefile exports it); GPU_ROME_RULE from the environment, default final (below;
  scripts/gpu_data.py reads the same variable: give both the same value). Writes OUT_DIR/fig-apsp-combined.dat (one row
  per graph) and OUT_DIR/fig-apsp-combined-num.tex (the axis limits and ticks). If build/data exists (make numbers),
  checks every CPU speedup against build/data/apsp_speedup.dat and the ratios against the macros
  (build/data/results.tex, gpu_results.tex: 44 checks); exits with status 1 on any mismatch. Prints ROME's time and the
  ratio per graph under both rules.

Each bar: n runs of Dijkstra's algorithm on one core (data/apsp.csv, rows _nt1, t_dijkstra_all: all n targets, into an
output allocated before the timer; bench/apsp.jl) divided by the method's time:
  Delta-stepping and ours, 1 and 16 cores   as scripts/cpu_data.py computes apsp_speedup.dat (the same code, below)
  ROME, one B200   per graph, t_match.median of rome-default, rome-leaf400 or rome-leaf800, by GPU_ROME_RULE:
                   final  the fastest passing build among the rows at GPU_COMMIT (the final job, which ran ROME right
                          after ours on the same node), as gpu_data.py selects baselines; other commits only for a graph
                          with no passing ROME row at GPU_COMMIT (none of the 25)
                   best   the fastest passing row over every B200 job and commit (conservative: in the final job, ROME's
                          runs are bimodal on several small graphs, c-42: 25-30 ms and 52-57 ms, and the median falls on
                          the slow mode)
  ours, one B200   code GPU_HEAD at GPU_COMMIT, t_cold.median
  passing: status ok and check.ok true. Graphs: the 25 of the CPU runs, the SuperFW and ROME test sets without luxembourg_osm (no
CPU run: its distance matrix exceeds a NUMA region's memory), by n, then m. The label over each B200 pair is ROME's time
over ours."""
import math, os, re, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "build", "figures")
COMMIT = sys.argv[3] if len(sys.argv) > 3 else "lib-g"
CPU_COMMITS = set((sys.argv[4] if len(sys.argv) > 4 else "lib-b,lib-a").split(","))
HEAD = os.environ.get("GPU_HEAD", "ours-auto")
RULE = os.environ.get("GPU_ROME_RULE", "final")
if RULE not in ("final", "best"): sys.exit(f"fig_apsp_combined: GPU_ROME_RULE must be final or best, not {RULE!r}")
ROMEB = ["rome-default", "rome-leaf400", "rome-leaf800"]
APSP = ["USpowerGrid", "delaunay_n13", "G67", "c-42", "CurlCurl_0", "fe_sphere", "nd6k", "delaunay_n14", "OPF_6000",
        "EB_8192_256", "EB_16384_64", "rgg2d_14", "rgg3d_14", "hypercube_14", "lpl1", "email-Enron", "t60k", "wing",
        "delaunay_n16", "c-69", "oilpan", "finan512", "fe_tooth", "onera_dual", "net4-1", "luxembourg_osm"]  # report order
warn = lambda m: print("fig_apsp_combined: WARNING:", m, file=sys.stderr)
tex = lambda g: g.replace("_", r"\_")
geo = lambda s: float(np.exp(np.log(pd.Series(s, dtype=float).dropna()).mean()))
fx = lambda x: f"{x:.1f}" if x < 9.95 else f"{x:.0f}"                         # the macros' format: 2.4, 13

# ---- CPUs: scripts/cpu_data.py's all-pairs code, unchanged in substance (apsp_speedup.dat is checked below)
a = pd.read_csv(os.path.join(DATA, "apsp.csv"))
a["nt"] = a["file"].str.extract(r"_nt(\d+)\.json").astype(int)
a = a[a.commit.isin(CPU_COMMITS)].copy()
a["ours"] = a.t_sym + a.t_num + a.t_closure                                     # first touch and renumbering excluded
e = pd.read_csv(os.path.join(DATA, "external.csv"))
ga = e[(e.tool == "gap_apsp") & (e.metric == "latency_s")].copy()
ga["value"] = ga.value.astype(float)
ga["full"] = ga.variant.fillna("").str.contains("runs")                        # n runs measured (not a sample)
ga = ga[ga.full | ~ga.groupby(["graph", "placement"]).full.transform("any")]
gap1 = ga[ga.placement == "core"].groupby("graph").value.mean()
gapP = ga[ga.placement == "processes"].groupby(["graph", "threads"]).value.mean()
gfull = e[(e.tool == "gap_apsp") & (e.metric == "total_s")].copy(); gfull["value"] = gfull.value.astype(float)
gfull = gfull.groupby(["graph", "threads"]).value.min()                        # all n runs, writing the n x n output
estimated = set()
def gap_all(g, n, nt):                                                         # n single-source runs on nt cores
    if (g, nt) in gfull.index: return gfull[(g, nt)]
    estimated.add((g, nt))
    v = gap1.get(g, np.nan) if nt == 1 else gapP.get((g, nt), np.nan) / nt
    return v * n
rows, omitted = [], []
for g in APSP:
    r1 = a[(a.graph == g) & (a.nt == 1)]; r16 = a[(a.graph == g) & (a.nt == 16)]
    if r1.empty or r16.empty: omitted.append(g); continue
    r1, r16 = r1.iloc[0], r16.iloc[0]
    if not (pd.isna(r1.dijkstra_sampled) and r1.dijkstra_targets == r1.n):
        warn(f"{g}: the one-core Dijkstra run is not all n targets")
    rows.append(dict(graph=g, n=int(r1.n), m=int(r1.m), tdij=r1.t_dijkstra_all, tgap1=gap_all(g, r1.n, 1), tours1=r1.ours,
                     tgap16=gap_all(g, r16.n, 16), tours16=r16.ours))
if omitted: warn(f"no CPU run at 1 or 16 threads, omitted: {omitted}")
if estimated: warn(f"Delta-stepping from per-run samples (no full-matrix run): {sorted(estimated)}")
D = pd.DataFrame(rows)

# ---- one B200: ours and ROME, selected as scripts/gpu_data.py selects them
b = pd.read_csv(os.path.join(DATA, "gpu", "gpu_apsp.csv"), low_memory=False)
ok = lambda d: d[(d.status == "ok") & (d["check.ok"].astype(str).str.lower() == "true")]
latest = lambda d, by: d.sort_values("date").groupby(by, dropna=False).tail(1)
b = b[~b.code.str.startswith("ours") | (b.commit == COMMIT)]                    # our rows: at COMMIT only
fams = {"ROME": ROMEB}
fams.update({c: [c] for c in b.code.unique() if not str(c).startswith("ours") and not str(c).startswith("rome-")})
drop = pd.Series(False, index=b.index)
for codes in fams.values():                                                     # baselines: rows at COMMIT where the
    f = b[b.code.isin(codes)]                                                   # family has passing ones, else any commit
    at = ok(f[f.commit.astype(str) == COMMIT])
    if at.empty: continue
    have = set(zip(at.graph, at["gpu.kind"]))
    drop |= pd.Series([(k in have) and (str(c) != COMMIT) for k, c in zip(zip(f.graph, f["gpu.kind"]), f.commit)],
                      index=f.index).reindex(b.index, fill_value=False)
good = ok(latest(b[~drop], ["graph", "gpu.kind", "code"]))
og = good[(good.code == HEAD) & (good["gpu.kind"] == "b200")].set_index("graph")
fastest = lambda d: d[d.code.isin(ROMEB) & (d["gpu.kind"] == "b200")].sort_values("t_match.median").groupby("graph").head(1).set_index("graph")
RG = {"final": fastest(good),                                                  # as gpu_data.py: rows at COMMIT, latest
      "best": fastest(ok(b))}                                                   # every passing row, any job and commit
rg = RG[RULE]
D["tgpu"] = D.graph.map(og["t_cold.median"]); D["trome"] = D.graph.map(rg["t_match.median"])
D["rome_build"] = D.graph.map(rg.code).str.replace("rome-", "")
if D[["tgpu", "trome"]].isna().any().any(): warn(f"no passing B200 run: {list(D.graph[D[['tgpu', 'trome']].isna().any(axis=1)])}")
for k, r in RG.items():                                                          # ROME's time per graph under both rules
    D[f"trome_{k}"] = D.graph.map(r["t_match.median"]); D[f"xrome_{k}"] = D[f"trome_{k}"] / D.tgpu
    D[f"rome_{k}_at"] = D.graph.map(r.code.str.replace("rome-", "") + " " + r.commit.astype(str) + " " + r.job.astype(str))

# ---- speedups over n runs of Dijkstra on one core, and the ratios
for s in ("gap1", "ours1", "gap16", "ours16", "rome", "gpu"): D[s] = D.tdij / D["t" + s]
D["xrome"] = D.trome / D.tgpu                                                  # the label: ROME's time over ours
D["xcpu16"] = D.tours16 / D.tgpu; D["xcpu1"] = D.tours1 / D.tgpu               # one B200 over 16 cores, over one core
D = D.sort_values(["n", "m"]).reset_index(drop=True)
S = ["gap1", "ours1", "gap16", "ours16", "rome", "gpu"]
v = lambda x: f"{x:.4g}" if np.isfinite(x) else "nan"
with open(os.path.join(OUT, "fig-apsp-combined.dat"), "w") as f:
    f.write("% generated by scripts/fig_apsp_combined.py: speedup over n runs of Dijkstra's algorithm on one core;\n"
            f"% lrome: ROME's time over ours on one B200 (GPU_ROME_RULE={RULE}), printed at height ylab\n")
    f.write("x graph n " + " ".join(S) + " lrome ylab\n")
    for i, r in D.iterrows():
        f.write(f"{i} {tex(r.graph)} {r.n} " + " ".join(v(r[s]) for s in S)
                + f" {fx(r.xrome) if np.isfinite(r.xrome) else 'nan'} {v(np.fmax(r.rome, r.gpu))}\n")

# axis limits and ticks from the data. Bottom: the power of ten below the smallest bar (the bars grow from it). Ticks: the
# powers of ten from there to the tallest bar. Top: room above the top tick for the legend (two rows, 20pt: 1.3 decades of
# this panel, 15.5pt each): 40 times the top tick (1.6 decades), so that the legend's bottom edge clears the top grid line
# by about 3.5pt; and at least a factor of 16 (1.2 decades) over the tallest bar, for the labels over the tallest bars
nice_above = lambda x: min(k * 10 ** math.floor(math.log10(x)) for k in (1, 2, 5, 10) if k * 10 ** math.floor(math.log10(x)) >= x)
allv = D[S].values
e0, e1 = math.floor(math.log10(np.nanmin(allv) / 1.1)), math.floor(math.log10(np.nanmax(allv)))
num = dict(ApspCombYmin=f"{10.0 ** e0:g}", ApspCombYmax=f"{max(nice_above(np.nanmax(allv) * 16), 40 * 10.0 ** e1):g}",
           ApspCombYticks=",".join(f"{10.0 ** e:g}" for e in range(e0, e1 + 1)), ApspCombLast=str(len(D) - 1),
           # not printed in the figure (the macros carry them): its summary numbers, for reference
           ApspCombGpuCpuGeo=fx(geo(D.xcpu16)), ApspCombGpuCpuOf=str(int(D.xcpu16.notna().sum())),
           ApspCombRomeGeo=fx(geo(D.xrome)), ApspCombRomeWins=str(int((D.xrome > 1).sum())), ApspCombRomeOf=str(int(D.xrome.notna().sum())))
with open(os.path.join(OUT, "fig-apsp-combined-num.tex"), "w") as f:
    f.write(f"% generated by scripts/fig_apsp_combined.py (GPU_ROME_RULE={RULE}): the figure's axis limits and ticks, and its\n"
            "% summary numbers over its graphs (geometric means)\n")
    for k, s in num.items(): f.write(f"\\def\\{k}{{{s}}}%\n")

# ---- report, and checks against the CPU speedup data and the macros (build/, from make numbers)
pd.set_option("display.width", 250)
print(D[["graph", "n", "tdij", "tgap1", "tours1", "tgap16", "tours16", "trome", "rome_build", "tgpu"] + S + ["xrome", "xcpu16"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("figure numbers:", ", ".join(f"{k}={s}" for k, s in num.items()))
print(f"ROME under both rules (the figure uses GPU_ROME_RULE={RULE}); time in s, build commit job, ratio ROME/ours:")
print(D[["graph", "tgpu", "trome_final", "rome_final_at", "xrome_final", "trome_best", "rome_best_at", "xrome_best"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
R16 = ["USpowerGrid", "delaunay_n13", "G67", "c-42", "CurlCurl_0", "fe_sphere", "nd6k", "lpl1", "email-Enron", "t60k",
       "wing", "delaunay_n16", "c-69", "oilpan", "finan512", "fe_tooth"]                       # ROME's test set (gpu_data.py)
for k in RG:
    x, x16 = D[f"xrome_{k}"], D[D.graph.isin(R16)][f"xrome_{k}"]
    print(f"  {k:5s}: geometric mean {geo(x):.3f} over the {x.notna().sum()} graphs (ROME's 16: {geo(x16):.3f}), "
          f"range {x.min():.2f}--{x.max():.2f}, faster than ROME on {(x > 1).sum()}")
bad = []
def check(name, have, want):
    if want is None: return
    print(f"  check {name}: here {have}, macro {want}" + ("" if have == want else "   <-- MISMATCH"))
    if have != want: bad.append(name)
sp = os.path.join(ROOT, "build", "data", "apsp_speedup.dat")
if os.path.isfile(sp):                                                         # the CPU speedups, value by value
    A3 = pd.read_csv(sp, sep=r"\s+")
    for _, r in A3.iterrows():
        g = r.graph.replace(r"\_", "_"); h = D.set_index("graph").loc[g]
        for c in ("ours1", "gap1", "ours16", "gap16"):
            if f"{h[c]:.4g}" != f"{float(r[c]):.4g}": check(f"apsp_speedup.dat {g} {c}", f"{h[c]:.4g}", f"{float(r[c]):.4g}")
    check("apsp_speedup.dat: same graphs", str(sorted(D.graph) == sorted(A3.graph.str.replace(r"\_", "_", regex=False))), "True")
    print(f"  check apsp_speedup.dat: {4 * len(A3)} speedups compared")
else: warn(f"{sp} missing: run make numbers to check the CPU speedups against it")
def macros(p):
    p = os.path.join(ROOT, "build", "data", p)
    return dict(re.findall(r"\\def\\(\w+)\{([^}]*)\}", open(p).read())) if os.path.isfile(p) else None
pm, gm = macros("results.tex"), macros("gpu_results.tex")
if pm:
    check("ApspGraphs", str(len(D)), pm.get("ApspGraphs"))
    for col, nt in (("1", "One"), ("16", "Sixteen")):
        x = D[f"tgap{col}"] / D[f"tours{col}"]; w = x[x > 1]
        check(f"Apsp{nt}DeltaWins", str(len(w)), pm.get(f"Apsp{nt}DeltaWins"))
        check(f"Apsp{nt}DeltaMedian", fx(x.median()), pm.get(f"Apsp{nt}DeltaMedian"))
        check(f"Apsp{nt}DeltaMax", fx(x.max()), pm.get(f"Apsp{nt}DeltaMax"))
    check("ApspOneDijWins", str(int((D.ours1 > 1).sum())), pm.get("ApspOneDijWins"))
    check("ApspOneDijMedian", fx(D.ours1.median()), pm.get("ApspOneDijMedian"))
if gm:
    WORDS = "Zero One Two Three Four Five Six Seven Eight Nine".split()
    cname = lambda g: "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z]+", re.sub(r"\d", lambda m: WORDS[int(m.group())] + " ", g)) if w)
    for g, x in zip(D.graph, D.xrome): check(f"GpuRomeX{cname(g)}", fx(x), gm.get(f"GpuRomeX{cname(g)}"))
    agree = {k: sum(fx(x) == gm.get(f"GpuRomeX{cname(g)}") for g, x in zip(D.graph, D[f"xrome_{k}"])) for k in RG}
    print(f"  the macros' ROME ratios agree with GPU_ROME_RULE=final on {agree['final']}, best on {agree['best']} of {len(D)} graphs")
    if agree[RULE] < len(D) and max(agree.values()) == len(D):
        print(f"  (build/data/gpu_results.tex was made under the other rule: give make pdf and this script the same GPU_ROME_RULE)")
    for c, nm in (("xcpu16", "Sixteen"), ("xcpu1", "One")):
        check(f"GpuVsCpu{nm}Geo", fx(geo(D[c])), gm.get(f"GpuVsCpu{nm}Geo"))
        check(f"GpuVsCpu{nm}Of", str(int(D[c].notna().sum())), gm.get(f"GpuVsCpu{nm}Of"))
        check(f"GpuVsCpu{nm}Median", fx(D[c].median()), gm.get(f"GpuVsCpu{nm}Median"))
        check(f"GpuVsCpu{nm}Range", f"{fx(D[c].min())}--{fx(D[c].max())}", gm.get(f"GpuVsCpu{nm}Range"))
else: warn("build/data/gpu_results.tex missing: run make numbers to check the ratios against the macros")
if bad: print("fig_apsp_combined: MISMATCH with the macros: " + ", ".join(bad), file=sys.stderr); sys.exit(1)
