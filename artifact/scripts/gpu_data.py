#!/usr/bin/env python3
"""Derived GPU numbers (LaTeX macros), tables and figure data, from the GPU benchmark CSVs (data/gpu/*.csv, written by bench/gpu_*.jl).

Usage: gpu_data.py GPU_DATA_DIR CPU_DATA_DIR OUT_DIR [COMMIT] [CPU_COMMITS]
Writes OUT_DIR/data/gpu_results.tex (macros \\Gpu...), OUT_DIR/data/gpu_*.dat (figure data), OUT_DIR/tables/tab-gpu-*.tex.
COMMIT: the library commit of OUR rows (default: the most common one among them); baseline rows (ROME, Fast-APSP, ...)
are used whatever commit their harness recorded. Rows are deduplicated per measurement, the latest date winning.
Conventions (see bench/gpu_apsp.jl, gpu_multigpu.jl, gpu_kernels.jl):
  ours      code ours-amf (AMF ordering, hybrid factorization), time t_cold.median (output preallocated); ours-bestfill
            is reported apart; phases.* from the HEAD-phases rows (each the median of 7 synchronized cold calls), else
            from the headline row's one synchronized call; the per-step figure and every phase macro: all graphs with phase measurements
  ROME      time t_match.median (its tree, allocation minus the n^2 output, and computing), one passing row (status ok
            and check.ok) per graph and GPU among rome-default, rome-leaf400, rome-leaf800, chosen by GPU_ROME_RULE:
              final (default)  the fastest build in the rows kept for ROME: at COMMIT where ROME passed there (the final
                               job, run right after ours), else the latest row per build of any commit
              best             the fastest passing row over every job and commit of that GPU (ROME's best measurement)
            Every ROME number follows the rule: the \\GpuRome... macros, \\GpuRtxRome..., Tables V and VI, gpu_apsp.dat
  Fast-APSP multi-GPU rows, its canonical scope (conversion, METIS, preconditioning, solve; output in host memory)
  semirings gpu_semirings.csv, gpu_semirings_kernels.csv (bench/semirings/gpu_semirings.jl, target gpu_semirings): rows at COMMIT,
            else at GPU_SR_COMMIT (default lib-f); see the section "other semirings"
Only real, verified measurements are used: status ok, check.ok true, and not extrapolated (a baseline row with extrapolated
== True timed a sample of sources and scaled the time to n: never used, for any code, in any number, table or figure)."""
import os, sys, math, re
import pandas as pd, numpy as np

GDATA, CDATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
COMMIT = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
CPU_COMMITS = set(sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else None
os.makedirs(os.path.join(OUT, "data"), exist_ok=True); os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
warn = lambda m: print("gpu_data: WARNING:", m, file=sys.stderr)
def read(d, f):
    p = os.path.join(d, f)
    if not os.path.isfile(p): warn(f"{p} missing"); return None
    return pd.read_csv(p, low_memory=False)
tex = lambda g: g.replace("_", r"\_")
fx = lambda x: f"{x:.1f}" if x < 9.95 else f"{x:.0f}"                        # the CPU tables' format: 2.4, 13
rng = lambda s: f"{fx(s.min())}--{fx(s.max())}" if len(s) else "--"
geo = lambda s: float(np.exp(np.log(s).mean()))
pct = lambda x: f"{100 * x:.0f}"
thou = lambda x: f"{int(x):,}".replace(",", "{,}")
def fxk(x):                      # speedups that reach thousands: as fx below 1,000 (2.4, 13, 612); from 1,000 on with a
    if x < 999.5: return fx(x)   # thousands separator as in the other macros (1{,}189); from 10,000 on, three significant
    if x < 9999.5: return thou(round(x))   # digits (10{,}600), not five digits of noise
    return thou(round(x, 2 - int(math.floor(math.log10(x)))))
WORDS = "Zero One Two Three Four Five Six Seven Eight Nine".split()
def cname(g):                                                                   # graph name -> macro suffix (letters only)
    s = re.sub(r"\d", lambda m: WORDS[int(m.group())] + " ", g)
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z]+", s) if w)
macros = {}
def sampled(d):                                                                 # rows whose time was extrapolated from a sample
    return d["extrapolated"].astype(str).str.lower() == "true" if "extrapolated" in d else pd.Series(False, index=d.index)
def ok(d):                                                                      # passing rows: real, verified measurements
    return d[(d.status == "ok") & (d["check.ok"].astype(str).str.lower() == "true") & ~sampled(d)]
def latest(d, by):                                                              # one row per measurement: the latest
    return d.sort_values("date").groupby(by, dropna=False).tail(1)
def pick_commit(d, codes):
    global COMMIT
    ours = d[codes(d)]
    if COMMIT is None and len(ours): COMMIT = ours.commit.mode()[0]
    other = ours[ours.commit != COMMIT]
    if len(other): warn(f"ignoring {len(other)} of our rows not at {COMMIT}: commits {sorted(set(other.commit))}")
    return d[~codes(d) | (d.commit == COMMIT)]

ROME16 = ["USpowerGrid", "delaunay_n13", "G67", "c-42", "CurlCurl_0", "fe_sphere", "nd6k", "lpl1", "email-Enron", "t60k",
          "wing", "delaunay_n16", "c-69", "oilpan", "finan512", "fe_tooth"]                    # ROME's Table 2, G0--G15
APSP = ["USpowerGrid", "delaunay_n13", "G67", "c-42", "CurlCurl_0", "fe_sphere", "nd6k", "delaunay_n14", "OPF_6000",
        "EB_8192_256", "EB_16384_64", "rgg2d_14", "rgg3d_14", "hypercube_14", "lpl1", "email-Enron", "t60k", "wing",
        "delaunay_n16", "c-69", "oilpan", "finan512", "fe_tooth", "onera_dual", "net4-1", "luxembourg_osm"]   # report order
FAP = ["598a", "luxembourg_osm", "web-sk-2005", "usroads-48", "fe_ocean"]                    # Fast-APSP's graphs, by n
ROADS = ["USA-t-NY", "USA-t-BAY", "USA-t-COL"]                                               # larger than one GPU
KINDS = {"b200": "", "rtx": "Rtx"}                                                           # macro suffix per GPU
HEAD = os.environ.get("GPU_HEAD", "ours-amf")                                                # our headline variant
ORDNAMES = {"amf": "AMF", "auto": "AutoOrder", "bestfill": "BestFill(AMF, METIS)"}
HEADORD = HEAD.split("-", 1)[1] if "-" in HEAD else "amf"                       # ours-amf -> amf, ours-auto -> auto
GPU_MEM_SPEC_GB = 180          # NVIDIA's B200 specification (HGX/DGX B200: 8 x 180 GB); CUDA reports gpu.memory_gb

# ================= one GPU: all-pairs, against ROME and the other baselines =================
a = read(GDATA, "gpu_apsp.csv")
a_raw = a.copy()                                                                # every row as read (ROME's rule "best" below)
a_all = latest(a, ["graph", "gpu.kind", "code", "commit"])                    # (every commit: the RTX note's fallback)
# only real measurements: the baselines' rows timed on a sample of sources and scaled to n (extrapolated == True) are not
# passing rows (ok() above), so no number, table or figure uses them; logged here per code and GPU
for (code_, kind_), d_ in a[sampled(a)].groupby(["code", "gpu.kind"]):
    warn(f"extrapolated (a sample of sources, scaled to n), never used: {code_} on {kind_}, {len(d_)} rows: {sorted(set(d_.graph))}")
a = pick_commit(a, lambda d: d.code.str.startswith("ours"))
# baselines: prefer rows at our COMMIT (a run that tags them with it ran them in the same job as ours, right after, on the
# same node); per family (ROME's three builds count as one) and per graph and GPU, rows of other commits are used only
# where the family has no passing row at COMMIT, and those graphs are logged
BASEFAM = {"ROME": ["rome-default", "rome-leaf400", "rome-leaf800"]}
BASEFAM.update({c: [c] for c in a.code.unique() if not str(c).startswith("ours") and not str(c).startswith("rome-")})
def prefer_commit(d):
    drop = pd.Series(False, index=d.index)
    for fam, codes in BASEFAM.items():
        f = d[d.code.isin(codes)]
        at = ok(f[f.commit.astype(str) == str(COMMIT)])
        if at.empty: continue                                                   # no rows at COMMIT: any commit, as before
        have = set(zip(at.graph, at["gpu.kind"]))
        key = list(zip(f.graph, f["gpu.kind"]))
        drop |= pd.Series([(k in have) and (str(c) != str(COMMIT)) for k, c in zip(key, f.commit)], index=f.index).reindex(d.index, fill_value=False)
        fb = sorted({f"{g} ({k})" for g, k in key if (g, k) not in have})
        warn(f"{fam}: rows at {COMMIT} for {len(have)} (graph, GPU); other commits for {fb}" if fb else f"{fam}: all rows at {COMMIT}")
    return d[~drop]
a = prefer_commit(a)
a = latest(a, ["graph", "gpu.kind", "code"])
good = ok(a)
ours = {k: good[(good.code == HEAD) & (good["gpu.kind"] == k)].set_index("graph") for k in KINDS}
# the reports use the B200; the RTX PRO 6000 appears in one number and was not rerun: if it has no rows at this commit and
# variant, its numbers come from GPU_RTX_HEAD at GPU_RTX_COMMIT (default: ours-amf at lib-d, the last RTX run)
if ours["rtx"].empty:
    rh, rc = os.environ.get("GPU_RTX_HEAD", "ours-amf"), os.environ.get("GPU_RTX_COMMIT", "lib-d")
    ours["rtx"] = latest(ok(a_all[(a_all.code == rh) & (a_all.commit == rc) & (a_all["gpu.kind"] == "rtx")]), ["graph"]).set_index("graph")
    warn(f"rtx: no {HEAD} rows at {COMMIT}; the RTX note uses {rh} at {rc} ({len(ours['rtx'])} graphs)")
oursbf = {k: good[(good.code == "ours-bestfill") & (good["gpu.kind"] == k)].set_index("graph") for k in KINDS}
for k in KINDS:                                                                 # BestFill was run at an earlier commit only
    if oursbf[k].empty:
        bf = ok(a_all[(a_all.code == "ours-bestfill") & (a_all["gpu.kind"] == k)])
        if len(bf): oursbf[k] = latest(bf, ["graph"]).set_index("graph"); warn(f"{k}: BestFill rows from {sorted(set(bf.commit))}")
# ROME: one passing row per graph and GPU, by GPU_ROME_RULE (see the top): "final", the fastest build among the rows kept
# above (at COMMIT where ROME passed there: the final job; else the latest row per build); "best", the fastest passing row
# over every job and commit of that GPU (one row per job: no row is a duplicate of another measurement)
ROMEB = ["rome-default", "rome-leaf400", "rome-leaf800"]
ROME_RULE = os.environ.get("GPU_ROME_RULE", "final").strip().lower() or "final"
if ROME_RULE not in ("final", "best"): sys.exit(f"gpu_data: GPU_ROME_RULE must be final or best, not {ROME_RULE!r}")
a_job = latest(a_raw, ["graph", "gpu.kind", "code", "job"])
def rome_pick(kind, rule=ROME_RULE):
    r = good if rule == "final" else ok(a_job)
    r = r[r.code.isin(ROMEB) & (r["gpu.kind"] == kind)].sort_values("t_match.median")
    return r.groupby("graph").head(1).set_index("graph")
rome = {k: rome_pick(k) for k in KINDS}
macros["GpuRomeRule"] = ROME_RULE
_rf, _rb = rome_pick("b200", "final"), rome_pick("b200", "best")                # the log: where the two rules differ (B200)
_d = [g for g in _rf.index if g in _rb.index and _rf.loc[g, "t_match.median"] != _rb.loc[g, "t_match.median"]]
print(f"gpu_data: ROME rule {ROME_RULE}; on the B200 the rule best differs from final on {len(_d)} of {len(_rf)} graphs:")
for g in _d:
    print(f"    {g:15s} final {_rf.loc[g, 'code'][5:]:8s} {_rf.loc[g, 'commit']} {_rf.loc[g, 't_match.median']:.4g} s"
          f" | best {_rb.loc[g, 'code'][5:]:8s} {_rb.loc[g, 'commit']} job {_rb.loc[g, 'job']} {_rb.loc[g, 't_match.median']:.4g} s")
cmp = {}
for k, suf in KINDS.items():
    o, r = ours[k], rome[k]
    gs = [g for g in o.index if g in r.index]
    C = pd.DataFrame(dict(n=o.loc[gs, "n"], ours=o.loc[gs, "t_cold.median"], bf=oursbf[k]["t_cold.median"].reindex(gs),
                          rome=r.loc[gs, "t_match.median"], rome_full=r.loc[gs, "t_full.median"], rome_own=r.loc[gs, "t_rome.median"],
                          build=r.loc[gs, "code"].str.replace("rome-", "")))
    C["x"] = C.rome / C.ours; C["xbf"] = C.rome / C.bf
    cmp[k] = C
    t16 = C.reindex([g for g in ROME16 if g in C.index])
    miss = [g for g in ROME16 if g not in C.index]
    if miss: warn(f"{k}: ROME test graphs without a passing pair (ours, ROME): {miss}")
    w = t16[t16.x > 1]
    macros.update({f"GpuRomeGeo{suf}": fx(geo(t16.x)), f"GpuRomeWins{suf}": str(len(w)), f"GpuRomeOf{suf}": str(len(t16)),
                   f"GpuRomeRange{suf}": rng(w.x), f"GpuRomeMin{suf}": fx(t16.x.min()), f"GpuRomeMax{suf}": fx(t16.x.max()),
                   f"GpuRomeMedian{suf}": fx(t16.x.median()), f"GpuRomeGeoBestFill{suf}": fx(geo(t16.xbf.dropna())),
                   f"GpuRomeWinsBestFill{suf}": str(int((t16.xbf > 1).sum()))})
    if len(t16): macros[f"GpuRomeMaxGraph{suf}"] = tex(t16.x.idxmax()); macros[f"GpuRomeMinGraph{suf}"] = tex(t16.x.idxmin())
    allg = C.reindex([g for g in APSP if g in C.index])                        # the 26 all-pairs graphs
    lo = allg[allg.x < 1].sort_values("x")
    macros.update({f"GpuRomeAllGeo{suf}": fx(geo(allg.x)), f"GpuRomeAllWins{suf}": str(int((allg.x > 1).sum())),
                   f"GpuRomeAllOf{suf}": str(len(allg)),
                   f"GpuRomeLosses{suf}": ", ".join(f"{tex(g)} ({x:.2f}\\x)" for g, x in lo.x.items()) or "none"})
    for g, x in sorted(C.x.items()): macros[f"GpuRomeX{cname(g)}{suf}"] = fx(x)
# the orderings at a fixed commit (B200, ROME's test set): AMF alone against ROME, the meta-ordering against AMF, and
# the better of AMF and METIS by fill against AMF; each pair from one commit, so that only the ordering differs
def _at(code, commit, kind="b200"):
    d = ok(a_all[(a_all.code == code) & (a_all.commit == commit) & (a_all["gpu.kind"] == kind)])
    return latest(d, ["graph"]).set_index("graph") if len(d) else d
_A04, _U04 = _at("ours-amf", "lib-e"), _at("ours-auto", "lib-e")
_A25, _B25 = _at("ours-amf", "lib-d"), _at("ours-bestfill", "lib-d")
if len(_A04) and len(_U04) and len(_A25) and len(_B25):
    _r = rome["b200"]
    _g = [g for g in ROME16 if g in _A04.index and g in _r.index]
    _x = _r.loc[_g, "t_match.median"] / _A04.loc[_g, "t_cold.median"]
    _gu = [g for g in ROME16 if g in _A04.index and g in _U04.index]
    _gb = [g for g in ROME16 if g in _A25.index and g in _B25.index]
    _fill = [abs(float(_B25.loc[g, "nnzLU"]) / float(_A25.loc[g, "nnzLU"]) - 1) for g in _gb]
    macros.update(GpuAmfCommit="lib-e", GpuRomeGeoAmf=fx(geo(_x)), GpuRomeWinsAmf=str(int((_x > 1).sum())),
                  GpuRomeOfAmf=str(len(_g)),
                  GpuAutoVsAmfPct=f"{100 * (1 - geo(_U04.loc[_gu, 't_cold.median'] / _A04.loc[_gu, 't_cold.median'])):.0f}",
                  GpuBestFillVsAmf=fx(geo(_B25.loc[_gb, "t_cold.median"] / _A25.loc[_gb, "t_cold.median"])),
                  GpuBestFillFillPct=f"{100 * max(_fill):.0f}")
    print(f"gpu_data: orderings: AMF@lib-e vs ROME {macros['GpuRomeGeoAmf']}x ({macros['GpuRomeWinsAmf']}/{len(_g)}); "
          f"auto vs AMF @lib-e {macros['GpuAutoVsAmfPct']}% faster; bestfill vs AMF @lib-d {macros['GpuBestFillVsAmf']}x; "
          f"fill within {macros['GpuBestFillFillPct']}%")
else:
    warn("orderings: missing rows for the fixed-commit ordering comparison")

# the RTX PRO 6000 note: per part, B200-node time over RTX-node time (geometric mean over ROME's graphs both ran)
both = [g for g in ROME16 if g in rome["b200"].index and g in rome["rtx"].index and g in ours["b200"].index and g in ours["rtx"].index]
if both:
    cpuph = lambda o: o["phases.symbolic"].astype(float) + o["phases.numeric_setup"].fillna(0).astype(float) + o["phases.numeric_cpu"].fillna(0).astype(float)
    macros.update(GpuRtxRomeTreeX=fx(geo(rome["b200"].loc[both, "t_tree.median"] / rome["rtx"].loc[both, "t_tree.median"])),
                  GpuRtxRomeCompX=fx(geo(rome["b200"].loc[both, "t_computing.median"] / rome["rtx"].loc[both, "t_computing.median"])),
                  GpuRtxOursCpuX=fx(geo(cpuph(ours["b200"].loc[both]) / cpuph(ours["rtx"].loc[both]))),
                  GpuRtxOursX=fx(geo(ours["b200"].loc[both, "t_cold.median"] / ours["rtx"].loc[both, "t_cold.median"])))

# figure data: one B200 (and one RTX PRO 6000), time per graph of ROME's test set
with open(os.path.join(OUT, "data", "gpu_apsp.dat"), "w") as f:
    f.write("x graph ours rome oursrtx romertx\n")
    for i, g in enumerate(ROME16):
        v = lambda k, c: f"{cmp[k].loc[g, c]:.4g}" if g in cmp[k].index else "nan"
        f.write(f"{i} {tex(g)} {v('b200', 'ours')} {v('b200', 'rome')} {v('rtx', 'ours')} {v('rtx', 'rome')}\n")
vals = pd.concat([cmp[k].reindex(ROME16)[["ours", "rome"]].stack() for k in KINDS]).dropna()
macros["GpuApspYmin"] = f"{10 ** math.floor(math.log10(vals.min())):.3g}"
macros["GpuApspYmax"] = f"{10 ** math.ceil(math.log10(vals.max())):.3g}"

# ROME's other two timing scopes (a table caption's): its own (elimination tree and computation, as its authors report)
# and full (with the allocation of its n^2 output), same build as the matched scope, over ROME's test set
for k, suf in KINDS.items():
    t16 = cmp[k].reindex([g for g in ROME16 if g in cmp[k].index])
    if len(t16):
        macros.update({f"GpuRomeOwnGeo{suf}": fx(geo(t16.rome_own / t16.ours)), f"GpuRomeOwnWins{suf}": str(int((t16.rome_own > t16.ours).sum())),
                       f"GpuRomeFullGeo{suf}": fx(geo(t16.rome_full / t16.ours)), f"GpuRomeFullWins{suf}": str(int((t16.rome_full > t16.ours).sum()))})

# one B200, larger graphs (a table): every graph with n >= 100,000 on which both codes pass the check (ROME: its best
# passing build, matched scope, as above), by n; graphs where every ROME build fails the check are not compared
LARGE_N = 100000
CL = cmp["b200"][cmp["b200"].n >= LARGE_N].sort_values("n")
if len(CL):
    macros.update(GpuRomeLargeGeo=fx(geo(CL.x)), GpuRomeLargeRange=rng(CL.x), GpuRomeLargeOf=str(len(CL)),
                  GpuRomeLargeWins=str(int((CL.x > 1).sum())), GpuRomeLargeNMin=thou(CL.n.min()), GpuRomeLargeNMax=thou(CL.n.max()),
                  GpuRomeLargeOwnGeo=fx(geo(CL.rome_own / CL.ours)), GpuRomeLargeFullGeo=fx(geo(CL.rome_full / CL.ours)))
    bigfail = [g for g in ours["b200"].index if ours["b200"].loc[g, "n"] >= LARGE_N and g not in CL.index]
    macros["GpuRomeLargeOmitted"] = str(len(bigfail))
    print(f"gpu_data: large graphs vs ROME: {list(CL.index)}; ours alone (no passing ROME build): {bigfail}")
    sec3 = lambda t: f"{t:.3f}" if t < 1 else (f"{t:.2f}" if t < 10 else f"{t:.1f}")
    lines = ["% generated by scripts/gpu_data.py", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Graph & $n$ & ours (s) & ROME (s) & $\times$ \\", r"\midrule"]
    for g, b in CL.iterrows():
        lines.append(f"{tex(g)} & {thou(b.n)} & {sec3(b.ours)} & {sec3(b.rome)} & {fx(b.x)} \\\\")
    lines += [r"\midrule", f"geo.\\ mean & & & & {macros['GpuRomeLargeGeo']} \\\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tables", "tab-gpu-rome.tex"), "w").write("\n".join(lines) + "\n")

# the L4 (Ada) note: ours (GPU_HEAD at COMMIT) against ROME's best passing build, matched scope, on ROME's test set
l4o = good[(good.code == HEAD) & (good["gpu.kind"] == "nvidial4")].set_index("graph")
if l4o.empty:                                                                   # the L4 note was run at an earlier commit
    l4all = ok(a_all[(a_all.code == HEAD) & (a_all["gpu.kind"] == "nvidial4")])
    if len(l4all):
        l4o = latest(l4all, ["graph"]).set_index("graph"); warn(f"nvidial4: our rows from {sorted(set(l4all.commit))}")
l4r = rome_pick("nvidial4")                                                     # (ROME's rows: GPU_ROME_RULE, as above)
l4g = [g for g in ROME16 if g in l4o.index and g in l4r.index]
if l4g:
    x = l4r.loc[l4g, "t_match.median"] / l4o.loc[l4g, "t_cold.median"]
    macros.update(GpuRomeGeoLfour=fx(geo(x)), GpuRomeWinsLfour=str(int((x > 1).sum())), GpuRomeOfLfour=str(len(x)))
    print(f"gpu_data: L4 vs ROME on {l4g}")

# the other one-GPU baselines (each in its own scope, see bench/gpu_baselines.jl), on the graphs both ran with real, verified
# measurements (ok(): a run timed on a sample of sources and scaled to n is never used). \GpuVs<name>Sampled: the graphs of
# ours on which the code passed only with such a run, left out of every number (B200: cuGraph 1, ADDS 6, Gunrock 18)
NAMES = {"cugraph": "Cugraph", "gunrock": "Gunrock", "adds": "Adds", "fw": "Fw", "ecl": "Ecl", "fastapsp1": "FastApspOne"}
passing = lambda d: d[(d.status == "ok") & (d["check.ok"].astype(str).str.lower() == "true")]   # (sampled or not)
for code, nm in NAMES.items():
    for k, suf in KINDS.items():
        b = good[(good.code == code) & (good["gpu.kind"] == k)].set_index("graph")
        sm = passing(a[(a.code == code) & (a["gpu.kind"] == k)])
        sm = sorted(g for g in sm[sampled(sm)].graph if g in ours[k].index)
        gs = [g for g in ours[k].index if g in b.index]
        if not gs: continue
        x = b.loc[gs, "t.median"] / ours[k].loc[gs, "t_cold.median"]
        macros.update({f"GpuVs{nm}Geo{suf}": fxk(geo(x)), f"GpuVs{nm}Min{suf}": fxk(x.min()), f"GpuVs{nm}Of{suf}": str(len(x)),
                       f"GpuVs{nm}Wins{suf}": str(int((x > 1).sum())), f"GpuVs{nm}Sampled{suf}": str(len(sm))})
        if sm: print(f"gpu_data: {code} on {k}: {len(x)} graphs compared; sampled runs left out on {len(sm)}: {sm}")
# ADDS: the graphs it did not complete within the harness's limit, 900 s (one B200; latest row per graph, of ours' graphs):
# killed at the limit (\GpuAddsKilled), or run on a sample of sources only, with a time scaled to n beyond the limit
# (\GpuAddsOverLimit: the full run would have been killed; counted as not completed). Graphs it ran on a sample with a scaled
# time under the limit (the harness samples when its calibration predicts a long run) are not counted here, nor are runs
# that crashed or were not verified; \GpuVsAddsSampled counts every sampled graph (over the limit or not)
LIMIT_S = 900
ad = a[(a.code == "adds") & (a["gpu.kind"] == "b200") & a.graph.isin(ours["b200"].index)]
killed = sorted(ad[ad.status.astype(str).str.contains("timeout")].graph)
adp = passing(ad); over = sorted(adp[sampled(adp) & (adp["t.median"].astype(float) >= LIMIT_S)].graph)
macros.update(GpuAddsTimeouts=str(len(killed) + len(over)), GpuAddsKilled=str(len(killed)), GpuAddsOverLimit=str(len(over)))
print(f"gpu_data: ADDS not completed within {LIMIT_S} s: killed {killed}; sampled, scaled time over the limit {over}")

# table: speedup of ours over each GPU code on one B200, on the graphs both completed with real, verified measurements
# (ROME: its row per graph by GPU_ROME_RULE, as everywhere; the others: their passing rows, never a sampled run)
BL = [("ROME", None, "t_match.median"),
      ("Fast-APSP", ["fastapsp1"], "t.median"), ("cuGraph", ["cugraph"], "t.median"),
      ("ECL-APSP", ["ecl"], "t.median"), ("ADDS", ["adds"], "t.median"), ("Gunrock", ["gunrock"], "t.median"),
      ("Floyd--Warshall", ["fw"], "t.median")]
# each code's macro name (\GpuVs<name>Max, \GpuVs<name>MaxGraph: its largest speedup in the table and that graph). For ROME
# it is \GpuVsRomeAllMax: over every graph both codes ran (the table's row), not ROME's 16 test graphs (\GpuRomeMax)
BLMAC = {"ROME": "RomeAll", "Fast-APSP": "FastApspOne", "cuGraph": "Cugraph", "ECL-APSP": "Ecl", "ADDS": "Adds",
         "Gunrock": "Gunrock", "Floyd--Warshall": "Fw"}
rows = []
for name, codes, col in BL:
    passed = rome["b200"] if codes is None else \
        ok(a[a.code.isin(codes) & (a["gpu.kind"] == "b200")]).sort_values(col).groupby("graph").head(1).set_index("graph")
    gs = [g for g in ours["b200"].index if g in passed.index]
    if not gs: continue
    x = passed.loc[gs, col] / ours["b200"].loc[gs, "t_cold.median"]
    rows.append((geo(x), name, len(x), x.median(), x.max(), x.idxmax()))
    macros.update({f"GpuVs{BLMAC[name]}Max": fxk(x.max()), f"GpuVs{BLMAC[name]}MaxGraph": tex(x.idxmax())})
# columns: code, graphs, then our speedup: geometric mean, median, and the largest with its graph (in parentheses).
# 245pt wide with \tabcolsep 3pt (measured with the real-only rows), set here so that the table never runs past the column
# (with main.tex's 6pt it would be 269pt); main.tex's \setlength{\tabcolsep}{...} before the \input should say 3pt too
lines = ["% generated by scripts/gpu_data.py",
         r"\setlength{\tabcolsep}{3pt}% 245pt wide (with 6pt: 269pt, wider than the column); main.tex should say 3pt too",
         r"\begin{tabular}{@{}lrrrr@{\hspace{3pt}}l@{}}", r"\toprule",
         r"Code & graphs & \multicolumn{4}{c@{}}{our speedup} \\", r"\cmidrule(l){3-6}",
         r" & & geo.\ mean & median & \multicolumn{2}{c@{}}{max} \\", r"\midrule"]
for gm, name, cnt, med, mx, mg in sorted(rows):
    lines.append(f"{name} & {cnt} & {fxk(gm)}\\x & {fxk(med)}\\x & {fxk(mx)}\\x & ({tex(mg)}) \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tables", "tab-gpu-baselines.tex"), "w").write("\n".join(lines) + "\n")

# ================= phases of the hybrid algorithm (B200) =================
PH = [("sym", ["symbolic"]), ("numcpu", ["numeric_setup", "numeric_cpu"]), ("numgpu", ["numeric_gpu"]), ("xfer", ["transfer"]),
      ("setup", ["solve_setup", "relabel", "other"]), ("inv", ["inverse"]), ("sol", ["solve"])]
hyb = ours["b200"]
# the shares: from the HEAD + "-phases" rows (each phases.* the median of 7 cold synchronized calls) where they exist at
# this commit, else from the HEAD rows' single synchronized call; total times (t_cold.median) always from the HEAD rows
phrows = good[(good.code == HEAD + "-phases") & (good["gpu.kind"] == "b200")].set_index("graph")
def phsrc(g): return phrows.loc[g] if g in phrows.index else hyb.loc[g]
fromph = [g for g in hyb.index if g in phrows.index]
warn(f"phases: shares from {HEAD}-phases (7-call medians) for {len(fromph)} of {len(hyb)} graphs"
     + (f"; from {HEAD} (one call) for {sorted(set(hyb.index) - set(fromph))}" if len(fromph) < len(hyb) else ""))
fnum = lambda r, c: float(r.get("phases." + c, 0) or 0) if pd.notna(r.get("phases." + c, np.nan)) else 0.0
def phases(g):
    r = phsrc(g)
    v = {k: sum(fnum(r, c) for c in cs) for k, cs in PH}
    v["alloc"] = float(hyb.loc[g, "t_alloc"])
    # finer parts (the per-step figure): the symbolic analysis split into the ordering and the rest (where phases.ordering exists)
    v["ord"] = min(fnum(r, "ordering"), v["sym"]); v["symrest"] = v["sym"] - v["ord"]
    v["nset"] = fnum(r, "numeric_setup"); v["ncpu"] = fnum(r, "numeric_cpu")
    return v
cpuf = good[(good.code == HEAD + "-cpufactor") & (good["gpu.kind"] == "b200")].set_index("graph")
# one set of graphs for the per-step figure and for every phase macro (\GpuPhaseGraphs, \GpuFactorShare*, \GpuClosureShare*,
# ...): every graph of our headline rows on one B200 with phase measurements (the 7-call phases rows, else the headline
# row's own synchronized call), ordered by n * fill (closure work n * nnz(L+U) over symbolic work nnz(A))
NF = lambda g: float(hyb.loc[g, "n"]) * float(hyb.loc[g, "fill"])
PG = [g for g in hyb.index if pd.notna(phsrc(g).get("phases.symbolic", np.nan))]
if len(PG) < len(hyb): warn(f"phases: no phase measurements for {sorted(set(hyb.index) - set(PG))}")
PGF = sorted(PG, key=NF)
GROUPS = [(0, 2e5, "$n \\cdot$fill $< 2\\cdot10^5$"), (2e5, 1e6, "$2\\cdot10^5$ to $10^6$"), (1e6, 1e12, "$\\ge 10^6$")]
# the parts of panel (a), in stacking order (bottom to top): (column, style, legend entry); the symbolic analysis splits
# into the ordering and the rest only when the rows carry phases.ordering
HASORD = any(phases(g)["ord"] > 0 for g in PGF)
PARTS = ([("ord", "ph ord", "ordering (CPU)"), ("symrest", "ph symrest", "rest of symbolic analysis (CPU)")] if HASORD else
         [("symrest", "ph sym", "symbolic analysis (CPU)")]) + \
        [("fac", "ph fac", "factorization (CPU + GPU)"), ("inv", "ph inv", "inverse $U^*$ (GPU)"), ("sol", "ph sol", "solve $XL^*$ (GPU)")]
def parts(v):
    return dict(ord=v["ord"], symrest=v["symrest"], fac=v["numcpu"] + v["numgpu"] + v["xfer"] + v["setup"], inv=v["inv"], sol=v["sol"])
with open(os.path.join(OUT, "data", "gpu_phases.dat"), "w") as f:
    f.write("x graph sym fac clo ms nfill " + " ".join(c for c, *_ in PARTS if c != "fac") + "\n")
    for i, g in enumerate(PGF):
        v = phases(g); fac_ = v["numcpu"] + v["numgpu"] + v["xfer"] + v["setup"]; clo_ = v["inv"] + v["sol"]
        tot = v["sym"] + fac_ + clo_; ms_ = 1e3 * float(hyb.loc[g, "t_cold.median"]); pv = parts(v)
        f.write(f"{i} {tex(g)} {v['sym'] / tot:.4f} {fac_ / tot:.4f} {clo_ / tot:.4f} {ms_:.3g} {NF(g):.4g} "
                + " ".join(f"{pv[c] / tot:.4f}" for c, *_ in PARTS if c != "fac") + "\n")
P = ["% generated by scripts/gpu_data.py: the per-step figure, one stacked part per line, then the legend (one row)"]
P += [f"  \\addplot[{st}, forget plot] table[x=x, y={c}] {{data/gpu_phases.dat}};" for c, st, _ in PARTS]
P += [f"  \\addlegendimage{{{st}, area legend}}\\addlegendentry{{{lab}}}" for c, st, lab in PARTS]
open(os.path.join(OUT, "data", "gpu_phases_plots.tex"), "w").write("\n".join(P) + "\n")
macros["GpuPhaseLegendCols"] = str(len(PARTS))
clsh = lambda g: (lambda v: (v["inv"] + v["sol"]) / sum(v[k] for k, _ in PH))(phases(g))
for k, (lo, hi, lab) in enumerate(GROUPS):
    idx = [i for i, g in enumerate(PGF) if lo <= NF(g) < hi]
    L_ = "ABC"[k]
    macros[f"GpuPhaseGroup{L_}Lo"] = f"{min(idx) - 0.5:.1f}" if idx else "0"
    macros[f"GpuPhaseGroup{L_}Hi"] = f"{max(idx) + 0.5:.1f}" if idx else "0"
    macros[f"GpuPhaseGroup{L_}Mid"] = f"{(min(idx) + max(idx)) / 2:.1f}" if idx else "0"
    macros[f"GpuPhaseGroup{L_}Label"] = lab
    if idx: macros[f"GpuPhaseGroup{L_}Closure"] = pct(float(np.median([clsh(PGF[i]) for i in idx])))
# the closure's share against n * fill: Spearman's rank correlation, and a logistic fit (least squares on the
# logit of the share against log10(n * fill)), written as a curve
xs = np.array([NF(g) for g in PGF]); ys = np.array([clsh(g) for g in PGF])
rk = lambda v: pd.Series(v).rank().to_numpy()
rho = float(np.corrcoef(rk(xs), rk(ys))[0, 1])
lx = np.log10(xs); yy = np.clip(ys, 1e-3, 1 - 1e-3); b1, b0 = np.polyfit(lx, np.log(yy / (1 - yy)), 1)
fit = lambda x: 1 / (1 + np.exp(-(b0 + b1 * np.log10(x))))
with open(os.path.join(OUT, "data", "gpu_phases_scatter.dat"), "w") as f:
    f.write("nfill clo\n")
    for i in range(len(PGF)): f.write(f"{xs[i]:.4g} {ys[i]:.4f}\n")
with open(os.path.join(OUT, "data", "gpu_phases_fit.dat"), "w") as f:
    f.write("nfill clo\n")
    for x in np.logspace(np.log10(xs.min()), np.log10(xs.max()), 41): f.write(f"{x:.4g} {fit(x):.4f}\n")
macros.update(GpuPhaseRho=f"{rho:.2f}", GpuPhaseFitLo=pct(fit(xs.min())), GpuPhaseFitHi=pct(fit(xs.max())),
              GpuPhaseNfillMin=f"{10 ** math.floor(math.log10(xs.min())):.0f}", GpuPhaseNfillMax=f"{10 ** math.ceil(math.log10(xs.max())):.0f}")
print(f"gpu_data: phases: Spearman {rho:.2f}, fit {pct(fit(xs.min()))}--{pct(fit(xs.max()))}%")
# figure: the time of each phase against n (log-log), one point per graph and phase
with open(os.path.join(OUT, "data", "gpu_phases_abs.dat"), "w") as f:
    f.write("n graph sym fac clo\n")
    for g in PGF:
        v = phases(g); fac_ = v["numcpu"] + v["numgpu"] + v["xfer"] + v["setup"]
        f.write(f"{int(hyb.loc[g, 'n'])} {tex(g)} {v['sym']:.5g} {fac_:.5g} {v['inv'] + v['sol']:.5g}\n")
macros["GpuPhaseLast"] = str(len(PGF) - 1)
macros["GpuPhaseXticks"] = ",".join(str(i) for i in range(len(PGF)))
macros["GpuPhaseXlabels"] = ",".join("{" + tex(g) + "}" for g in PGF)
# The per-step figure in one column (fig-phases-1col.tex, data/gpu_phases_1col.tex): one horizontal bar per graph with
# phase measurements (PGF, all of them), by n * fill, in the three n * fill groups above. 100% = the sum of the parts'
# times in the synchronized phases calls (each the median of 7 cold with_phases calls; each step synchronizes the device
# and the CPU-GPU overlaps of an untimed call are off, so the sum exceeds the call's time, t_cold.median: macros
# \GpuPhaseSyncRatio*); the ordering is the median of 7 cold calls of the ordering alone, and the rest of the symbolic
# analysis is phases.symbolic minus it. Six parts, in the call's order (PATHPACK's phases(), runtime/config.jl): ordering
# and the rest of the symbolic analysis (CPU); the factorization (numeric_setup: the factor's storage, A's entries copied
# in, the plan, CPU; numeric_cpu and numeric_gpu: the numeric factorization's bottom on CPU threads and top on the GPU);
# transfer and setup (transfer: the factors' upload; solve_setup: the solve's structure and factor on the device, the
# dense operators of large fronts, the memory check, workspace allocation; relabel: the result's columns into the labels
# of A; other: the workspace's release); the closure's inverse U* and solve XL* (GPU). Written as TeX calls: per group
# \PhGroup{label}{median share of the closure, %}{graphs}, then per graph \PhBar{graph}{share/label} x 6, share in %
# (2 decimals) and label its integer, the six labels of a bar summing to 100 (largest remainder)
GROUPS1 = ["$n\\cdot\\mathrm{fill} < 2\\cdot10^5$", "$2\\cdot10^5 \\le n\\cdot\\mathrm{fill} < 10^6$", "$n\\cdot\\mathrm{fill} \\ge 10^6$"]
PARTS1 = ("ord", "symrest", "num", "xs", "inv", "sol")
def parts1(v):                                                                  # the six parts of a bar, seconds
    return dict(ord=v["ord"], symrest=v["symrest"], num=v["numcpu"] + v["numgpu"], xs=v["xfer"] + v["setup"], inv=v["inv"], sol=v["sol"])
def lr_round(p):                                                                # shares (%, sum 100) -> integers summing to 100
    f = [math.floor(x + 1e-9) for x in p]; k = 100 - sum(f)                     # largest remainder; ties: the larger share
    for i in sorted(range(len(p)), key=lambda i: (p[i] - f[i], p[i]), reverse=True)[:max(k, 0)]: f[i] += 1
    return f
L = [f"% generated by scripts/gpu_data.py: the per-step figure in one column, all {len(PGF)} graphs with phase measurements, by n * fill;"
     " shares of the synchronized parts' sum, %",
     "% \\PhGroup{label}{median share of the closure (inverse + solve)}{graphs}, then per graph \\PhBar{graph} and per part"
     " {share/label}: ordering, rest of symbolic, factorization, transfer and setup, inverse, solve"]
for j, (lo, hi, _) in enumerate(GROUPS):
    gg = [g for g in PGF if lo <= NF(g) < hi]
    if not gg: continue
    L.append(f"\\PhGroup{{{GROUPS1[j]}}}{{{pct(float(np.median([clsh(g) for g in gg])))}}}{{{len(gg)}}}")
    for g in gg:
        pv = parts1(phases(g)); tot_ = sum(pv.values()); sh = [100 * pv[c] / tot_ for c in PARTS1]; lab = lr_round(sh)
        if sum(lab) != 100: warn(f"phases: {g}: the labels sum to {sum(lab)}")
        L.append(f"\\PhBar{{{tex(g)}}}" + "".join(f"{{{s_:.2f}/{l_}}}" for s_, l_ in zip(sh, lab)))
open(os.path.join(OUT, "data", "gpu_phases_1col.tex"), "w").write("\n".join(L) + "\n")
S = pd.DataFrame([phases(g) for g in PG], index=PG)
tot = S[[k for k, _ in PH]].sum(axis=1)                 # the parts' sum, synchronized (allocation outside; see the per-step figure)
fac = (S.sym + S.numcpu + S.numgpu) / tot; cpu = (S.sym + S.numcpu) / tot; clo = (S.inv + S.sol) / tot
macros.update(GpuFactorShareMedian=pct(fac.median()), GpuFactorShareMin=pct(fac.min()), GpuFactorShareMax=pct(fac.max()),
              GpuCpuShareMedian=pct(cpu.median()), GpuCpuShareMax=pct(cpu.max()), GpuSymShareMedian=pct((S.sym / tot).median()),
              GpuSymShareMax=pct((S.sym / tot).max()), GpuSymShareMaxGraph=tex((S.sym / tot).idxmax()),
              GpuClosureShareMedian=pct(clo.median()), GpuClosureShareMin=pct(clo.min()), GpuClosureShareMax=pct(clo.max()),
              GpuAllocShareMedian=pct((S.alloc / (tot + S.alloc)).median()), GpuAllocShareMax=pct((S.alloc / (tot + S.alloc)).max()),
              GpuPhaseGraphs=str(len(PG)))
big = [g for g in PG if hyb.loc[g, "n"] > 50000]                              # as the CPU section: n > 50,000
if big:
    macros.update(GpuFactorShareBigMedian=pct(fac[big].median()), GpuClosureShareBigMedian=pct(clo[big].median()))
# the same shares through one function, over the same graphs (PG, the per-step figure's): it must reproduce the macros above, and adds
# \GpuSetupShare* (transfer and setup: transfer, solve_setup, relabel, other), \GpuNumShareMedian (the numeric
# factorization alone), \GpuPhaseColClosure A--C (the closure's median share per n * fill group, the strips of the per-step figure) and
# \GpuPhaseColGraphs A--C (the groups' sizes), \GpuPhaseSyncRatio* (the synchronized parts' sum over the call's time,
# t_cold.median: the overhead of timing each step alone)
def phase_shares(gs):
    S_ = pd.DataFrame([phases(g) for g in gs], index=gs); t_ = S_[[k for k, _ in PH]].sum(axis=1)
    f_ = (S_.sym + S_.numcpu + S_.numgpu) / t_; c_ = (S_.inv + S_.sol) / t_; x_ = (S_.xfer + S_.setup) / t_
    m = dict(GpuPhaseGraphs=str(len(gs)), GpuFactorShareMedian=pct(f_.median()), GpuFactorShareMin=pct(f_.min()),
             GpuFactorShareMax=pct(f_.max()), GpuSymShareMedian=pct((S_.sym / t_).median()), GpuClosureShareMedian=pct(c_.median()),
             GpuClosureShareMin=pct(c_.min()), GpuClosureShareMax=pct(c_.max()), GpuSetupShareMedian=pct(x_.median()),
             GpuSetupShareMin=pct(x_.min()), GpuSetupShareMax=pct(x_.max()), GpuSetupShareMaxGraph=tex(x_.idxmax()),
             GpuNumShareMedian=pct(((S_.numcpu + S_.numgpu) / t_).median()))
    b_ = [g for g in gs if hyb.loc[g, "n"] > 50000]
    if b_: m.update(GpuFactorShareBigMedian=pct(f_[b_].median()), GpuClosureShareBigMedian=pct(c_[b_].median()))
    for k_, (lo, hi, _) in enumerate(GROUPS):
        gg = [g for g in gs if lo <= NF(g) < hi]
        if gg: m.update({f"GpuPhaseColClosure{'ABC'[k_]}": pct(float(np.median([clsh(g) for g in gg]))), f"GpuPhaseColGraphs{'ABC'[k_]}": str(len(gg))})
    sync = t_ / hyb.loc[gs, "t_cold.median"].astype(float)
    m.update(GpuPhaseSyncRatioMedian=fx(sync.median()), GpuPhaseSyncRatioMin=fx(sync.min()), GpuPhaseSyncRatioMax=fx(sync.max()))
    return m
for key, val in phase_shares(PG).items():
    if key in macros and macros[key] != val: warn(f"phases: {key} = {val} by phase_shares, {macros[key]} above")
    macros.setdefault(key, val)
cg = [g for g in PG if g in cpuf.index]                                         # (the phase macros' graphs: the same graphs)
if len(cg) < len(PG): warn(f"phases: no {HEAD}-cpufactor rows for {sorted(set(PG) - set(cg))}")
if cg:
    slow = cpuf.loc[cg, "t_cold.median"] / hyb.loc[cg, "t_cold.median"]
    macros.update(GpuCpuFactorSlowMedian=fx(slow.median()), GpuCpuFactorSlowMax=fx(slow.max()), GpuCpuFactorSlowMaxGraph=tex(slow.idxmax()),
                  GpuCpuFactorSlowOf=str(len(cg)))                            # (the graphs with both runs)

# ================= GPU against CPU (our CPU rows, data/apsp.csv: Float64, t_sym + t_num + t_closure) =================
c = read(CDATA, "apsp.csv")
if c is not None:
    c["nt"] = c["file"].str.extract(r"_nt(\d+)\.json").astype(int)
    if CPU_COMMITS: c = c[c.commit.isin(CPU_COMMITS)]
    c["ours_cpu"] = c.t_sym + c.t_num + c.t_closure
    for nt, nm in ((1, "One"), (16, "Sixteen"), (64, "SixtyFour")):
        cc = c[c.nt == nt].set_index("graph")
        gs = [g for g in APSP if g in cc.index and g in hyb.index]
        if not gs: continue
        x = cc.loc[gs, "ours_cpu"] / hyb.loc[gs, "t_cold.median"]
        macros.update({f"GpuVsCpu{nm}Geo": fx(geo(x)), f"GpuVsCpu{nm}Median": fx(x.median()), f"GpuVsCpu{nm}Range": rng(x),
                       f"GpuVsCpu{nm}Of": str(len(x))})
        bx = x[[g for g in gs if hyb.loc[g, "n"] > 50000]]
        if len(bx): macros[f"GpuVsCpu{nm}BigRange"] = rng(bx)

    # table: our closure on CPUs (1 and 16 cores) and on one B200, per graph, by number of vertices
    med = c.groupby(["graph", "nt"]).ours_cpu.median()
    gs = sorted([g for g in APSP if g in hyb.index and (g, 1) in med.index and (g, 16) in med.index],
                key=lambda g: (int(hyb.loc[g, "n"]), int(hyb.loc[g, "m"])))
    sec = lambda t: f"{t:.3f}" if t < 1 else (f"{t:.2f}" if t < 10 else (f"{t:.1f}" if t < 100 else f"{t:.0f}"))
    lines = ["% generated by scripts/gpu_data.py", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{CPU (s)} & B200 & \multicolumn{2}{c}{B200 speedup} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(l){6-7}",
             r"Graph & $n$ & 1 core & 16 cores & (s) & vs.\ 1 & vs.\ 16 \\", r"\midrule"]
    for g in gs:
        t1, t16, tg = med[(g, 1)], med[(g, 16)], float(hyb.loc[g, "t_cold.median"])
        lines.append(f"{tex(g)} & {thou(hyb.loc[g, 'n'])} & {sec(t1)} & {sec(t16)} & {sec(tg)} & {fx(t1 / tg)} & {fx(t16 / tg)} \\\\")
    lines += [r"\midrule", f"geo.\\ mean & & & & & {fx(geo(pd.Series([med[(g, 1)] / float(hyb.loc[g, 't_cold.median']) for g in gs])))} & "
              f"{fx(geo(pd.Series([med[(g, 16)] / float(hyb.loc[g, 't_cold.median']) for g in gs])))} \\\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tables", "tab-gpu-cpu.tex"), "w").write("\n".join(lines) + "\n")
    cc = c[c.graph.isin(gs) & c.nt.isin([1, 16])]                              # the library version of the CPU rows used
    macros["GpuCpuCommit"] = ", ".join(sorted(set(cc.commit.astype(str))))

# ================= kernels (one GPU, Float32) =================
k = read(GDATA, "gpu_kernels.csv")
if k is not None:
    keep = k.impl != "ours"                                                     # our rows: per GPU kind, at COMMIT if present
    for kind in k["gpu.kind"].unique():
        ko = k[(k.impl == "ours") & (k["gpu.kind"] == kind)]
        if ko.empty: continue
        kc = os.environ.get("GPU_KERNEL_COMMIT") or (COMMIT if (ko.commit == COMMIT).any() else ko.commit.mode()[0])
        if kc != COMMIT: warn(f"kernels ({kind}): our rows at {kc} (none at {COMMIT})")
        keep |= (k.impl == "ours") & (k["gpu.kind"] == kind) & (k.commit == kc)
    k = k[keep]
    k = latest(k, ["gpu.kind", "op", "impl", "semiring", "n"])
    def kr(kind, op, impl, sem, n):
        v = k[(k["gpu.kind"] == kind) & (k.op == op) & (k.impl == impl) & (k.semiring == sem) & (k.n == n)].gmuladd
        return float(v.iloc[0]) if len(v) else np.nan
    for kind, suf in KINDS.items():
        kk = k[k["gpu.kind"] == kind]
        if kk.empty: continue
        N = int(kk[kk.impl == "ours"].n.max())
        mp, mm, sg = kr(kind, "gemm", "ours", "MinPlus", N), kr(kind, "gemm", "ours", "MaxMin", N), kr(kind, "gemm", "cublas", "PlusTimes", N)
        so, sc = kr(kind, "trsmL", "ours", "MinPlus", N), kr(kind, "trsmL", "cublas", "PlusTimes", N)
        to, tc = kr(kind, "trtri", "ours", "MinPlus", N), kr(kind, "trtri", "cusolver", "PlusTimes", N)
        if kind == "b200" and not (sg >= sc >= mp >= so >= tc >= to):              # the kernel figure names its curves in this order
            warn(f"kernels: the curves' order at n = {N} is not SGEMM, STRSM, min-plus, solve, STRTRI, inverse; "
                 "check the names at the right of fig-kernels-gpu.tex")
        macros.update({f"GpuKernNNum{suf}": str(N), f"GpuKernN{suf}": thou(N), f"GpuMinPlusRate{suf}": f"{mp / 1e3:.1f}", f"GpuMaxMinRate{suf}": f"{mm / 1e3:.1f}",
                       f"GpuSgemmRate{suf}": f"{sg / 1e3:.1f}", f"GpuMinPlusPct{suf}": pct(mp / sg), f"GpuMaxMinPct{suf}": pct(mm / sg),
                       f"GpuSolveRate{suf}": f"{so / 1e3:.1f}", f"GpuStrsmRate{suf}": f"{sc / 1e3:.1f}", f"GpuSolvePct{suf}": pct(so / sc),
                       f"GpuInverseRate{suf}": f"{to / 1e3:.1f}", f"GpuStrtriRate{suf}": f"{tc / 1e3:.1f}", f"GpuInversePct{suf}": pct(to / tc)})
    kb = k[k["gpu.kind"] == "b200"]
    with open(os.path.join(OUT, "data", "gpu_kernels.dat"), "w") as f:      # panel (b): one B200, T multiply-adds/s
        f.write("n minplus maxmin sgemm solve strsm inverse strtri\n")
        for n in sorted(kb.n.unique()):
            r = [kr("b200", *a_, n) / 1e3 for a_ in (("gemm", "ours", "MinPlus"), ("gemm", "ours", "MaxMin"), ("gemm", "cublas", "PlusTimes"),
                 ("trsmL", "ours", "MinPlus"), ("trsmL", "cublas", "PlusTimes"), ("trtri", "ours", "MinPlus"), ("trtri", "cusolver", "PlusTimes"))]
            f.write(f"{n} " + " ".join(f"{v:.4g}" if np.isfinite(v) else "nan" for v in r) + "\n")
    macros["GpuKernelYmax"] = f"{5 * math.ceil(kb.gmuladd.max() / 1e3 * 1.08 / 5):g}"
    macros["GpuKernelXmin"] = str(int(kb.n.min())); macros["GpuKernelXmax"] = str(int(kb.n.max()))

# ================= other semirings (one B200; Table tab-gpu-semirings, macros \GpuSr...) =================
# gpu_semirings.csv (bench gpu_semirings.jl): the whole call (apsp_gpu!, output preallocated), t_cold.median, per graph,
# semiring, and element type; each row's check: sampled columns of the result against the CPU closure in the same semiring
# on a Float64 copy of the weights (Boolean: breadth-first search). gpu_semirings_kernels.csv: the semiring GEMM, n x n.
# Rows at COMMIT if there are any, else at GPU_SR_COMMIT (default lib-f: lib-e plus the three
# commits that end lib-g, patch-identical; the same GPU kernels as lib-g), in the variants that are the library's own
# choice ("default"; "pair1" = HEURISTIC_PAIR on, its default; "pair0" is an A/B setting). Time: over the tropical closure
# in Float32 on the same graph, same job and variant. GEMM: at the largest n measured there; the element types other than
# Float32 (not measured by lib-f's run; the semiring commits change only Float32 kernels) from another commit, logged.
SR_ROWS = [("MinPlus", "Float32", r"tropical $(\min, +)$", 0), ("MaxPlus", "Float32", r"max-plus $(\max, +)$", 0),
           ("MinProd", "Float32", r"min-times $(\min, \times)$", 0), ("MaxProd", "Float32", r"max-times $(\max, \times)$", 0),
           ("MaxMin", "Float32", r"bottleneck $(\max, \min)$", 0), ("MinMax", "Float32", r"bottleneck $(\min, \max)$", 0),
           ("PlusProd", "Float32", r"probability $(+, \times)$", 0), ("OrAnd", "Int32", r"Boolean $(\lor, \land)$", 0),
           ("MaxMin", "Int32", r"bottleneck $(\max, \min)$", 1), ("MinPlus", "Float64", r"tropical $(\min, +)$", 1),
           ("MaxMin", "Float64", r"bottleneck $(\max, \min)$", 1), ("PlusProd", "Float64", r"probability $(+, \times)$", 1)]
SR_SUF = {"Float32": "", "Int32": "Int", "Float64": "Dbl"}                   # macro suffix per element type
SR_VARIANTS = {"default", "pair1", "nan"}                                      # (nan: rows of a script without variants)
SR_EXACT = {"MaxMin", "MinMax", "OrAnd"}                                       # exact for bottleneck and Boolean
sr = read(GDATA, "gpu_semirings.csv")
if sr is not None:
    sr["variant"] = sr.variant.astype(str)
    sr = sr[sr.code.astype(str).str.startswith("ours") & sr.variant.isin(SR_VARIANTS)]
    sr_commit = str(COMMIT) if (sr.commit.astype(str) == str(COMMIT)).any() else os.environ.get("GPU_SR_COMMIT", "lib-f")
    if sr_commit != str(COMMIT): warn(f"semirings: no rows at {COMMIT}; rows at {sr_commit}")
    sr = ok(latest(sr[sr.commit.astype(str) == sr_commit], ["graph", "semiring", "eltype", "job", "variant"]))
    sr_ref = sr[(sr.semiring == "MinPlus") & (sr.eltype == "Float32")].set_index(["job", "variant", "graph"])["t_cold.median"]
    srk = read(GDATA, "gpu_semirings_kernels.csv")
    if srk is not None:
        srk["variant"] = srk.variant.astype(str)
        srk = srk[(srk.op == "gemm") & (srk.impl == "ours") & srk.variant.isin(SR_VARIANTS)]
        sr_kat = srk[srk.commit.astype(str) == sr_commit]
        if srk.empty: srk = None; warn("semirings: no GEMM rows")
        else: sr_kn = int((sr_kat if len(sr_kat) else srk).n.max())
    sr_tab, sr_from = {}, []
    for sem, T, name, grp in SR_ROWS:
        r = sr[(sr.semiring == sem) & (sr.eltype == T)]
        x = pd.Series([t / sr_ref.get((j, v, g), np.nan) for j, v, g, t in zip(r.job, r.variant, r.graph, r["t_cold.median"])],
                      index=r.graph.values, dtype=float).dropna()
        x = x[~x.index.duplicated(keep="last")]                                # (rows sorted by date: the latest job)
        if x.empty: warn(f"semirings: no checked rows for {sem} {T} at {sr_commit}"); continue
        rate = np.nan
        if srk is not None:
            kr_ = srk[(srk.semiring == sem) & (srk["T"] == T) & (srk.n == sr_kn)].sort_values("date")
            at = kr_[kr_.commit.astype(str) == sr_commit]
            if len(at): rate = float(at.gmuladd.iloc[-1])
            elif T != "Float32" and len(kr_): rate = float(kr_.gmuladd.iloc[-1]); sr_from.append(f"{sem} {T} ({kr_.commit.iloc[-1]})")
        exact = bool((r["check.mode"] == "exact").all())
        sr_tab[(sem, T)] = dict(name=name, grp=grp, x=x, rate=rate, exact=exact, maxrel=float(r["check.cpu_closure_maxrel"].astype(float).max()),
                                ref=sorted(set(r["check.reference"].astype(str))), cols=sorted(set(r["check.columns"].astype(int))))
        if (sem in SR_EXACT) != exact: warn(f"semirings: {sem} {T}: check mode {sorted(set(r['check.mode']))}")
        if not all(("BFS" in c) if sem == "OrAnd" else c.startswith("PATHPACK.CPU closure") for c in sr_tab[(sem, T)]["ref"]):
            warn(f"semirings: {sem} {T}: checked against {sr_tab[(sem, T)]['ref']} (expected: CPU closure; Boolean: breadth-first search)")
    if sr_from: warn(f"semirings: GEMM rates not measured at {sr_commit}, from: {', '.join(sr_from)}")
    sr_graphs = sorted(set().union(*[set(v["x"].index) for v in sr_tab.values()]))
    for (sem, T), v in sr_tab.items():
        if len(v["x"]) < len(sr_graphs): warn(f"semirings: {sem} {T} on {len(v['x'])} of {len(sr_graphs)} graphs")
    if set(sr_graphs) != set(ROME16) | {"598a", "luxembourg_osm"}:
        warn(f"semirings: graphs are not ROME's 16 plus 598a and luxembourg_osm (the reported set): {sr_graphs}")
    f2 = lambda v: f"{v:.2f}"
    r2 = lambda vs: f"{f2(min(vs))}--{f2(max(vs))}" if f2(min(vs)) != f2(max(vs)) else f2(min(vs))
    for (sem, T), v in sr_tab.items():
        p = f"GpuSr{sem}{SR_SUF[T]}"
        macros.update({p + "Geo": f2(geo(v["x"])), p + "Min": f2(v["x"].min()), p + "Max": f2(v["x"].max())})
        if np.isfinite(v["rate"]): macros[p + "Rate"] = f"{v['rate'] / 1e3:.1f}"
    gm = lambda keys: [geo(sr_tab[k]["x"]) for k in keys if k in sr_tab]
    macros["GpuSrTimesRange"] = r2(gm([("MaxPlus", "Float32"), ("MinProd", "Float32"), ("MaxProd", "Float32")]))
    macros["GpuSrBottleneckRange"] = r2(gm([("MaxMin", "Float32"), ("MinMax", "Float32")]))
    dbl = gm([k for k in sr_tab if k[1] == "Float64"])
    if dbl: macros["GpuSrDblRange"] = f"{fx(min(dbl))}--{fx(max(dbl))}"
    inexact = [v["maxrel"] for v in sr_tab.values() if not v["exact"]]
    if inexact:                                                                 # the largest relative error, rounded up
        e = math.floor(math.log10(max(inexact))); d = math.ceil(max(inexact) / 10 ** e - 1e-9)
        macros["GpuSrMaxRelErr"] = (f"{d}\\cdot" if d < 10 else "") + f"10^{{{e + (d >= 10)}}}"
        if d == 1: macros["GpuSrMaxRelErr"] = f"10^{{{e}}}"
    macros.update(GpuSrGraphs=str(len(sr_graphs)), GpuSrCommit=sr_commit,
                  GpuSrCheckCols="/".join(str(c) for c in sorted(set(c for v in sr_tab.values() for c in v["cols"]))))
    if srk is not None: macros["GpuSrKernN"] = thou(sr_kn)
    for sem, mk in (("MinPlus", "GpuMinPlusRate"), ("MaxMin", "GpuMaxMinRate")):   # the kernel figure's rates, as macros
        v = sr_tab.get((sem, "Float32"), {}).get("rate", np.nan)
        if mk in macros and np.isfinite(v) and f"{v / 1e3:.1f}" != macros[mk]:
            warn(f"semirings: {sem} GEMM {v / 1e3:.1f} T/s in the table, {macros[mk]} in \\{mk} (gpu_kernels.csv)")
    lines = ["% generated by scripts/gpu_data.py", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{relative time} & GEMM \\", r"\cmidrule(lr){3-4}",
             r"Semiring & Type & mean & range & (T/s) \\", r"\midrule"]
    prev = None
    for sem, T, name, grp in SR_ROWS:
        if (sem, T) not in sr_tab: continue
        v = sr_tab[(sem, T)]
        if prev is not None and grp != prev: lines.append(r"\midrule")
        prev = grp
        isref = (sem, T) == ("MinPlus", "Float32")
        rt = f"{v['rate'] / 1e3:.1f}" if np.isfinite(v["rate"]) else "--"
        lines.append(f"{name} & \\texttt{{{T}}} & {'1' if isref else f2(geo(v['x']))} & {'' if isref else r2(list(v['x']))} & {rt} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tables", "tab-gpu-semirings.tex"), "w").write("\n".join(lines) + "\n")
    print(f"gpu_data: semirings at {sr_commit}, {len(sr_graphs)} graphs; references {sorted(set(x for v in sr_tab.values() for x in v['ref']))}")

# ================= several GPUs: strong scaling against Fast-APSP =================
# the several-GPU figure, one panel per graph (Fast-APSP's graphs by n, then the road networks), time against 1..8 B200s:
#   fastapsp  Fast-APSP, its canonical scope (t.median), every count it ran
#   ours      our whole call (t.median), ordering GPU_MORDER
#   reuse     new weights on the same graph (t_reuse): numeric refactorization + copies + closure, symbolic analysis reused
#   closure   the closure alone (t_closure), on one factorization made outside the timer
#   ideal     ideal scaling of the whole call from its first count
#   floor     the "setup" floor, the part of the whole call that does not scale: whole call minus closure (t.median -
#             t_closure) at the largest count. Chosen over the multi-GPU timeline (timeline.setup / factor / replica_*):
#             it comes from the same medians as the plotted curves (the timeline is one extra instrumented call, noisy on
#             the roads), it exists for every graph, and at the largest count whole = floor + closure by construction.
#   counts at which our output does not fit the GPUs (status "too large for g GPUs") are shaded in the panel
m = read(GDATA, "gpu_multigpu.csv")
MORDER = os.environ.get("GPU_MORDER", "amf")                                  # our multi-GPU rows: AMF where present
if m is not None:
    m_any = m.copy()                                                            # every commit: the memory limits below
    m = pick_commit(m, lambda d: d.code == "ours")
    keyord = m.key.astype(str).str.extract(r"_(amf|auto|bestfill)(?:_nt\d+)?$")[0]        # (older rows: no ordering column)
    m["ord"] = (m["ordering"] if "ordering" in m else pd.Series(np.nan, index=m.index)).fillna(keyord).fillna("bestfill").astype(str).str.lower()
    m = latest(m, ["graph", "gpu.kind", "code", "ngpus", "ord"])
    mall = m[m["gpu.kind"] == "b200"]
    mg = ok(mall)
    used = {}
    def series(g, code):
        s = mg[(mg.graph == g) & (mg.code == code)]
        if code == "ours":
            have = list(s.ord.unique())
            o = next((x for x in [MORDER, "auto", "amf", "bestfill"] if x in have), have[0] if have else MORDER)
            used[g] = o; s = s[s.ord == o]
        return s.set_index("ngpus").sort_index()
    MG = [g for g in FAP + ROADS if (((mg.graph == g) & (mg.code == "ours")).any() and
                                      mg[(mg.graph == g) & (mg.code == "ours")]["t.median"].notna().any())]
    if len(MG) < len(FAP + ROADS): warn(f"multigpu: no rows for {[g for g in FAP + ROADS if g not in MG]}")
    eff, effclo, effre, fap8, fap1, fapeff, floorshare, one8 = [], [], [], {}, {}, [], [], {}
    road8, roadbest, roadre8, fapmissing = {}, {}, {}, {}
    P, tabrows = [], {}
    for i, g in enumerate(MG):
        o, fp = series(g, "ours"), series(g, "fastapsp")
        o = o[o["t.median"].notna()]
        g0, top = int(o.index.min()), int(o.index.max()); t0 = o.loc[g0, "t.median"]
        floor = float(o.loc[top, "t.median"] - o.loc[top, "t_closure"])
        tl = mall[(mall.graph == g) & (mall.code == "ours") & (mall.ord == used[g])]
        oom = sorted(int(c) for c in tl[tl.status.astype(str).str.startswith("too large")].ngpus)
        if not oom:                                                                # the output's memory need does not depend on
            ta = m_any[(m_any.graph == g) & (m_any.code == "ours") & (m_any["gpu.kind"] == "b200")]   # the commit: earlier runs
            oom = sorted(set(int(c) for c in ta[ta.status.astype(str).str.startswith("too large")].ngpus if int(c) < g0))
            if oom: warn(f"multigpu: {g}: counts too large for the GPUs' memory from earlier runs: {oom}")
        with open(os.path.join(OUT, "data", f"gpu_multigpu_{i}.dat"), "w") as f:
            f.write("g ours reuse closure fastapsp ideal floor\n")
            for gg in range(1, 9):
                v = lambda s, c: f"{s.loc[gg, c]:.4g}" if gg in s.index and c in s and np.isfinite(s.loc[gg, c]) else "nan"
                f.write(f"{gg} {v(o, 't.median')} {v(o, 't_reuse')} {v(o, 't_closure')} {v(fp, 't.median')} "
                        f"{(t0 * g0 / gg) if gg >= g0 else float('nan'):.4g} {floor if gg >= g0 else float('nan'):.4g}\n")
        if 1 in o.index and top > 1:
            eff.append(o.loc[1, "t.median"] / (top * o.loc[top, "t.median"]))
            effclo.append(o.loc[1, "t_closure"] / (top * o.loc[top, "t_closure"]))
            effre.append(o.loc[1, "t_reuse"] / (top * o.loc[top, "t_reuse"]))
        floorshare.append(floor / o.loc[top, "t.median"])
        if g in FAP:
            if 8 in o.index and 8 in fp.index: fap8[g] = fp.loc[8, "t.median"] / o.loc[8, "t.median"]
            if 1 in o.index and 1 in fp.index: fap1[g] = fp.loc[1, "t.median"] / o.loc[1, "t.median"]
            if 1 in o.index and 8 in fp.index: one8[g] = fp.loc[8, "t.median"] / o.loc[1, "t.median"]
            if 1 in fp.index and 8 in fp.index: fapeff.append(fp.loc[1, "t.median"] / (8 * fp.loc[8, "t.median"]))
        else:
            if 8 in o.index and 8 in fp.index: road8[g] = fp.loc[8, "t.median"] / o.loc[8, "t.median"]
            roadbest[g] = o["t.median"].min(); roadre8[g] = o.loc[top, "t_reuse"]
        miss = [c for c in range(1, 9) if c not in fp.index]
        if miss: fapmissing[g] = miss
        L = "ABCDEFGH"[i]
        macros[f"GpuMultiTitle{L}"] = tex(g)
        macros[f"GpuMultiStart{L}"] = str(g0)
        # the panel's numbers: our speedup over Fast-APSP at our first and last counts (drawn as arrows between the two
        # curves); the closure's parallel efficiency from the first count to the last (a macro, not drawn)
        fpx = {c: float(fp.loc[c, "t.median"] / o.loc[c, "t.median"]) for c in (g0, top) if c in fp.index and np.isfinite(fp.loc[c, "t.median"])}
        ceff = float(o.loc[g0, "t_closure"] * g0 / (top * o.loc[top, "t_closure"])) if top > g0 else float("nan")
        for c, nm in ((g0, "First"), (top, "Last")):
            if c in fpx: macros[f"GpuMultiFap{nm}{L}"] = fx(fpx[c])
        if np.isfinite(ceff): macros[f"GpuMultiCloEff{L}"] = pct(ceff)
        # the table's row (Table tab-gpu-multigpu): the same series, counts and ratios as the panel and the macros above
        tabrows[g] = dict(n=int(o.loc[g0, "n"]), g0=g0, top=top, t0=float(t0), ttop=float(o.loc[top, "t.median"]),
                          clo=float(o.loc[g0, "t_closure"] / o.loc[top, "t_closure"]) if top > g0 else float("nan"),
                          fptop=float(fp.loc[top, "t.median"]) if top in fpx else float("nan"), x=fpx.get(top, float("nan")))
        # the panel (four per row), read by figures/fig-multigpu.tex; y labels on the first column, x labels on the last row
        nrows = math.ceil(len(MG) / 4)
        vals_ = pd.concat([o[c] for c in ("t.median", "t_reuse", "t_closure")] + [fp["t.median"]]).astype(float)
        lo_, hi_ = math.log10(vals_[vals_ > 0].min()), math.log10(vals_[vals_ > 0].max())
        sp_ = max(hi_ - lo_, 1)                                                   # y range: the curves, a little margin
        extra = f", ymin={10 ** (lo_ - 0.06 * sp_):.4g}, ymax={10 ** (hi_ + 0.04 * sp_):.4g}"
        extra += ", ylabel={time}" if i % 4 == 0 else ""
        if i >= 4 * (nrows - 1): extra += ", xlabel={GPUs}"
        if i == 0: extra += ", legend to name=multigpulegend"
        f_ = f"data/gpu_multigpu_{i}.dat"
        P.append(f"  \\nextgroupplot[title={{{tex(g)}}}{extra}]")
        if oom:                                                                    # shade the counts that do not fit
            x1 = math.sqrt(max(oom) * (max(oom) + 1))
            P.append(f"    \\fill[oomfill] ({{axis cs:1,1}}|-{{rel axis cs:0,0}}) rectangle ({{axis cs:{x1:.3f},1}}|-{{rel axis cs:0,1}});")
        for st, col in (("fapline", "fastapsp"), ("closureline", "closure"), ("reuseline", "reuse"), ("oursline", "ours")):
            P.append(f"    \\addplot[{st}, forget plot] table[x=g, y={col}] {{{f_}}};")
        if i == 0:                                                                 # the legend, in reading order
            for st, e in (("fapline", "Fast-APSP"), ("oursline", "ours, whole call"),
                          ("reuseline", "ours, new weights (symbolic analysis reused)"), ("closureline", "ours, star alone"),
                          ("area legend, oomlegend", "our output exceeds the GPUs' memory")):
                P.append(f"    \\addlegendimage{{{st}}}\\addlegendentry{{{e}}}")
            P.append("    \\addlegendimage{legend image code/.code={\\draw[ratio arrow] (0.13cm,-0.11cm) -- (0.13cm,0.13cm);}}"
                     "\\addlegendentry{our speedup over Fast-APSP}")
        # labels: right of the first arrow when it stands at the left edge, else left of it (over the shaded counts);
        # left of the last arrow; the first is drawn only when the counts are at least a factor 2 apart
        for c, nm, xs_, anc in ((g0, "First", 1.07, "west" if g0 == 1 else "east"), (top, "Last", 1 / 1.07, "east")):
            if c not in fpx or (nm == "First" and top < 2 * g0): continue           # first arrow: only if apart from the last
            lo_, hi_ = float(o.loc[c, "t.median"]) * 1.3, float(fp.loc[c, "t.median"]) / 1.3
            P.append(f"    \\draw[ratio arrow] (axis cs:{c * xs_:.4g},{lo_:.4g}) -- (axis cs:{c * xs_:.4g},{hi_:.4g})"
                     f" node[midway, anchor={anc}, multi label] {{\\GpuMultiFap{nm}{L}\\x}};")
    open(os.path.join(OUT, "data", "gpu_multigpu_panels.tex"), "w").write("% generated by scripts/gpu_data.py\n" + "\n".join(P) + "\n")
    print(f"gpu_data: multigpu panels {MG}; Fast-APSP counts missing: {fapmissing}")
    macros["GpuMultiRows"] = str(math.ceil(len(MG) / 4))
    macros["GpuMultiPanels"] = str(len(MG))
    macros["GpuMultiMax"] = str(int(mg[mg.code == "ours"].ngpus.max()))
    ords = set(used[g] for g in MG)
    macros["GpuMultiOrdering"] = (ORDNAMES.get(next(iter(ords)), next(iter(ords))) if len(ords) == 1 else
        ORDNAMES.get(MORDER, MORDER) + " (" + ", ".join(f"{tex(g)}: {ORDNAMES.get(used[g], used[g])}" for g in MG if used[g] != MORDER) + ")")
    if any(used[g] != MORDER for g in MG): warn("multigpu: other orderings used for " + ", ".join(g for g in MG if used[g] != MORDER))
    pc = lambda v: f"{100 * min(v):.0f}--{100 * max(v):.0f}"
    if fap8:                                                                    # Fast-APSP's graphs, 8 GPUs each
        x = pd.Series(fap8)
        macros.update(GpuFapEightGeo=fx(geo(x)), GpuFapEightRange=rng(x), GpuFapEightMin=fx(x.min()), GpuFapEightMax=fx(x.max()), GpuFapEightOf=str(len(x)))
    if fap1:
        x = pd.Series(fap1); macros.update(GpuFapOneGeo=fx(geo(x)), GpuFapOneRange=rng(x))
    if one8:                                                                    # our 1 GPU against its 8
        x = pd.Series(one8); macros.update(GpuFapOneVsEightGeo=fx(geo(x)), GpuFapOneVsEightRange=rng(x))
    if road8:                                                                   # road networks, 8 GPUs each
        x = pd.Series(road8); macros.update(GpuFapEightRoadsGeo=fx(geo(x)), GpuFapEightRoadsRange=rng(x))
    for g, t in roadbest.items():
        macros[f"GpuRoadBest{cname(g)}"] = f"{t:.2f}"; macros[f"GpuRoadReuse{cname(g)}"] = f"{roadre8[g]:.3f}"
    if roadre8: macros["GpuRoadReuseRange"] = f"{min(roadre8.values()):.2f}--{max(roadre8.values()):.2f}"
    if eff: macros.update(GpuEffEight=pc(eff), GpuEffEightMedian=f"{100 * np.median(eff):.0f}")
    if effclo: macros.update(GpuEffEightClosure=pc(effclo), GpuEffEightClosureMedian=f"{100 * np.median(effclo):.0f}")
    if effre: macros.update(GpuEffEightReuse=pc(effre), GpuEffEightReuseMedian=f"{100 * np.median(effre):.0f}")
    if fapeff: macros.update(GpuFapEffEight=pc(fapeff))
    if floorshare: macros["GpuFloorShareEight"] = pc(floorshare)            # the setup floor's share of the call at 8 GPUs
    # table (in place of the several-GPU figure): per graph, Fast-APSP's graphs then the road networks, each group by n; our whole
    # call on the fewest GPUs that hold the output (1 on Fast-APSP's graphs) and on the most, the closure alone's speedup
    # between the two counts, and Fast-APSP on the most GPUs (its canonical scope) with our speedup over it
    M = int(macros["GpuMultiMax"])
    for g, r in tabrows.items():                                                # (road networks: the time on M GPUs, beside
        if g in ROADS and r["top"] == M: macros[f"GpuRoadEight{cname(g)}"] = f"{r['ttop']:.2f}"   # the best, GpuRoadBest)
    fapclo = [tabrows[g]["clo"] for g in FAP if g in tabrows and tabrows[g]["g0"] == 1 and tabrows[g]["top"] == M]
    if fapclo: macros["GpuFapCloSpeedEightGeo"] = fx(geo(pd.Series(fapclo)))
    sec4 = lambda t: "--" if not np.isfinite(t) else (f"{t:.3f}" if t < 1 else (f"{t:.2f}" if t < 10 else (f"{t:.1f}" if t < 100 else f"{t:.0f}")))
    fxn = lambda v: fx(v) if np.isfinite(v) else "--"
    def trow(g):
        r = tabrows[g]; at = lambda v: v if r["top"] == M else float("nan")
        return (f"{tex(g)} & {thou(r['n'])} & {r['g0']} & {sec4(r['t0'])} & {sec4(at(r['ttop']))} & {sec4(at(r['fptop']))} & "
                f"{fxn(at(r['clo']))} & {fxn(at(r['x']))} \\\\")
    lines = ["% generated by scripts/gpu_data.py", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{fewest GPUs} & \multicolumn{2}{c}{" + str(M) + r" GPUs} & \multicolumn{2}{c@{}}{speedup (\x)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
             r"Graph & $n$ & GPUs & ours & ours & Fast-APSP & star & ours \\", r"\midrule"]               # times in seconds
    fg = sorted([g for g in FAP if g in tabrows], key=lambda g: tabrows[g]["n"])
    rg = sorted([g for g in ROADS if g in tabrows], key=lambda g: tabrows[g]["n"])
    lines += [trow(g) for g in fg]
    if fg:
        lines += [r"\midrule", f"geo.\\ mean & & & & & & {macros.get('GpuFapCloSpeedEightGeo', '--')} & {macros.get('GpuFapEightGeo', '--')} \\\\"]
    if rg: lines += [r"\midrule"] + [trow(g) for g in rg]
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tables", "tab-gpu-multigpu.tex"), "w").write("\n".join(lines) + "\n")
    big = mg[(mg.code == "ours") & mg["t.median"].notna()].sort_values("n")
    if len(big):
        r = big.iloc[-1]; bg = big[big.graph == r.graph]
        tb = bg.sort_values("t.median").iloc[0]
        macros.update(GpuMaxNMulti=thou(r.n), GpuMaxNMultiGraph=tex(r.graph), GpuMaxNMultiMinGpus=str(int(bg.ngpus.min())),
                      GpuMaxNMultiTB=f"{4 * float(r.n) ** 2 / 1e12:.1f}", GpuMaxNMultiTime=f"{tb['t.median']:.2f}",
                      GpuMaxNMultiTimeGpus=str(int(tb.ngpus)))

# ================= setup: hardware, software, memory limit =================
b = a[a.code.str.startswith("ours") & (a["gpu.kind"] == "b200")]
pretty = lambda s: re.sub(r"\s+", " ", re.sub(r"\(R\)|\(TM\)|\d+-Core Processor", "", str(s))).strip().title().replace("Amd Epyc", "AMD EPYC")
first = lambda d, c: d[c].dropna().mode()[0] if c in d and d[c].notna().any() else "--"
macros["GpuMemGB"] = str(GPU_MEM_SPEC_GB)
macros["GpuPerNode"] = str(int(pd.to_numeric(m["gpu.count"], errors="coerce").max())) if m is not None else "--"
macros["GpuHostCPU"] = pretty(first(b, "gpu.host_cpu"))
macros["GpuHostCPURtx"] = pretty(first(a[a["gpu.kind"] == "rtx"], "gpu.host_cpu"))
macros["GpuJulia"] = str(first(b, "provenance.julia")); macros["GpuCudaJl"] = str(first(b, "gpu.cuda_jl"))
macros["GpuCudaRuntime"] = ".".join(str(first(b, "gpu.runtime")).split(".")[:2])
macros["GpuJuliaThreads"] = str(int(first(b, "gpu.julia_threads")))
memb = float(first(b, "gpu.memory_gb")) * 1e9                                  # bytes CUDA reports
macros["GpuMaxNBound"] = thou(1000 * math.floor(math.sqrt(memb / 4) / 1000))    # 4 n^2 bytes of Float32 output fill the GPU
one = ours["b200"].reset_index().sort_values("n")                               # our headline rows (this commit)
macros.update(GpuMaxNMeasured=thou(one.n.iloc[-1]), GpuMaxNMeasuredGraph=tex(one.graph.iloc[-1]),
              GpuMaxNMeasuredTime=f"{one['t_cold.median'].iloc[-1]:.2f}")
macros["GpuDataCommit"] = COMMIT or "--"
# the headline ordering: in captions ("Ours, with \GpuOrdering, and ROME") and after "On GPUs, the ordering's time is
# significant next to the closure, so" (\GpuOrderingText)
macros["GpuOrdering"] = "a heuristic ordering" if HEADORD == "auto" else ORDNAMES.get(HEADORD, HEADORD)
macros["GpuOrderingText"] = ("we use AMF alone" if HEADORD == "amf" else
    r"we use a heuristic meta-ordering based on AMF~\cite{rothberg1998} and nested dissection~\cite{george1973}" if HEADORD == "auto" else
    r"we use AutoOrder, a rule evaluated in $O(n)$ per component from its degrees: AMF by default; AMF with the vertices of highest degree eliminated last on graphs with hubs (maximum degree above $2\sqrt{n}$ and above three times the mean); and a breadth-first nested dissection on lattice-like graphs (most vertices of one degree, at most~8)")

with open(os.path.join(OUT, "data", "gpu_results.tex"), "w") as f:
    f.write(f"% generated by scripts/gpu_data.py from {GDATA} (our rows at library commit {COMMIT})\n")
    for kk, v in macros.items(): f.write(f"\\def\\{kk}{{{v}}}\n")
print(f"gpu_data: {len(macros)} macros, commit {COMMIT}")
for kk, v in macros.items():
    if not (kk.startswith("GpuRomeX") or kk.startswith("GpuPhaseX")): print(f"  {kk} = {v}")
