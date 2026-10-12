#!/usr/bin/env python3
"""The plan and the report of the quick benchmark (quick.sh): our solver against state-of-the-art codes on a few graphs
on your machine, compared with the reference measurements in data/.

  quick_report.py plan cpu|gpu MEM_GB       the default graphs whose n x n output fits in MEM_GB, smallest first
  quick_report.py report cpu|gpu CSV_DIR    the run's results (collect.py's CSVs in CSV_DIR) against data/

What a run on other hardware can reproduce:
  - exactly: n, m and nnz(L+U). The symbolic analysis is deterministic: the reference rows of this library version
    (lib-g, data/gpu/gpu_apsp.csv) give the same numbers on every machine, for the CPU's factor and the GPU's;
  - checks: every result is checked (64 columns against Dijkstra's algorithm; on the GPU, also against the CPU's Float64
    star);
  - speedups measured on one machine: over n runs of Dijkstra's algorithm on one core (CPU) and over ROME on the same
    GPU (GPU). They depend on the hardware, but much less than the times do;
  - times: they depend on the hardware. The reference runs' are shown for scale (one B200; one EPYC 7702 NUMA region).
"""
import csv, math, os, sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ROME16 = ("USpowerGrid delaunay_n13 G67 c-42 CurlCurl_0 fe_sphere nd6k lpl1 email-Enron t60k wing delaunay_n16 c-69 "
          "oilpan finan512 fe_tooth").split()                     # ROME's test graphs (the headline 3.1x)
CPU_QUICK = "USpowerGrid delaunay_n13 G67 c-42 CurlCurl_0 delaunay_n14 fe_sphere".split()   # n runs of Dijkstra < 25 s each

def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def ok(r):
    return r is not None and str(r.get("status", "")).startswith("ok") and str(r.get("check.ok", "true")).lower() != "false"

def gmean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None

def reference_gpu():
    """The reference one-B200 rows of this version: ours (ordering auto) and ROME's fastest passing build per graph."""
    ours, rome = {}, {}
    for r in rows(os.path.join(DATA, "gpu", "gpu_apsp.csv")):
        if r["commit"] != "lib-g" or r["gpu.kind"] != "b200": continue
        if r["code"] == "ours-auto": ours[r["graph"]] = r
        elif r["code"].startswith("rome-") and ok(r):
            if r["graph"] not in rome or num(r["t_match.median"]) < num(rome[r["graph"]]["t_match.median"]): rome[r["graph"]] = r
    return ours, rome

def exact(r, ref):
    """yes: n, m and nnz(L+U) as in the reference runs; ~: nnz(L+U) within 1% (see METIS_NOTE); else NO and the differences."""
    if r is None or ref is None: return "-"
    diff = [c for c in ("n", "m", "nnzLU") if r.get(c) != ref.get(c)]
    if not diff: return "yes"
    if diff == ["nnzLU"] and abs(num(r["nnzLU"]) / num(ref["nnzLU"]) - 1) <= 0.01: return "~"
    return "NO (" + ", ".join(f"{c} {r.get(c)} vs {ref.get(c)}" for c in diff) + ")"

# On a graph with several components the library orders the components concurrently, and METIS (one of the orderings
# BestFill(AMF(), METIS()) tries for the reported factor) is not deterministic when called from several threads at once.
METIS_NOTE = ("~ = nnz(L+U) within 1% of the reference runs': on graphs with several components (lpl1, net4-1, thermomech_dM) the "
              "components are\n  ordered concurrently, and METIS, one of the two orderings BestFill tries for the reported "
              "factor, is not deterministic\n  then. The timed runs (ordering auto) do not use METIS.")

def f(x, w=9, p=4):
    return f"{x:{w}.{p}f}" if x is not None else f"{'-':>{w}s}"

def x_(x, w=8):
    return f"{x:{w - 1}.2f}x" if x is not None else f"{'-':>{w}s}"

def plan(mode, mem_gb):
    ours, _ = reference_gpu()
    n = {g: int(r["n"]) for g, r in ours.items()}
    graphs = ROME16 if mode == "gpu" else CPU_QUICK
    need = (lambda k: 4.0 * k * k * 1.3) if mode == "gpu" else (lambda k: 8.0 * k * k * 1.15)
    fit = [g for g in sorted(graphs, key=lambda g: n[g]) if need(n[g]) <= mem_gb * 1e9]
    left = [g for g in graphs if g not in fit]
    if left: print(f"(not run: {' '.join(left)}; their output needs more than {mem_gb:.0f} GB)", file=sys.stderr)
    print(" ".join(fit))

def group(left, *spans):        # a header line over column groups: (title, width) after `left` blank columns
    return " " * left + "   ".join(f" {t} ".center(w, "-") for t, w in spans)

def report_gpu(d):
    ours_p, rome_p = reference_gpu()
    run = rows(os.path.join(d, "gpu_apsp.csv"))
    if not run: sys.exit(f"no results in {d}/gpu_apsp.csv")
    ours = {r["graph"]: r for r in run if r["code"] == "ours-auto"}
    if not ours: sys.exit(f"no rows of ours-auto in {d}/gpu_apsp.csv")
    rome = {}
    for r in run:
        if r["code"].startswith("rome-"): rome.setdefault(r["graph"], []).append(r)
    gpu = next((r.get("gpu.name") for r in ours.values() if r.get("gpu.name")), "?")
    mem = next((num(r.get("gpu.memory_gb")) for r in ours.values() if r.get("gpu.memory_gb")), None)
    print(f"This machine: {gpu}" + (f" ({mem:.0f} GB)" if mem else "") + f", library {next(iter(ours.values()))['commit']}.")
    print("Reference runs: one NVIDIA B200, library lib-g (data/gpu/gpu_apsp.csv).")
    print("Times in seconds (median of the cold calls); speedup = ROME / ours; exact = n, m and nnz(L+U) as in the reference runs.\n")
    print(group(34, ("this machine", 44), ("reference", 28)))
    print(f"{'graph':15s} {'n':>8s}  {'exact':6s}  {'ours':>9s} {'check':5s}  {'ROME':>9s} {'build':8s} {'speedup':>8s}   "
          f"{'ours':>9s} {'ROME':>9s} {'speedup':>8s}")
    sp_here, sp_ref, nex, nok, nrome_ok, graphs = [], [], 0, 0, 0, sorted(ours, key=lambda g: int(ours[g]["n"]))
    for g in graphs:
        o = ours[g]; to = num(o.get("t_total")) if ok(o) else None
        passed = [r for r in rome.get(g, []) if ok(r) and num(r.get("t_match.median"))]
        best = min(passed, key=lambda r: num(r["t_match.median"])) if passed else None
        tr = num(best["t_match.median"]) if best else None
        sp = tr / to if to and tr else None
        po, pr = ours_p.get(g), rome_p.get(g)
        tpo = num(po["t_total"]) if po else None; tpr = num(pr["t_match.median"]) if pr else None
        psp = tpr / tpo if tpo and tpr else None
        e = exact(o, po); nex += e == "yes"; nok += ok(o); nrome_ok += best is not None
        if sp and psp: sp_here.append(sp); sp_ref.append(psp)
        status = "ok" if ok(o) else ("FAIL" if o.get("status", "").startswith(("failed", "check")) else o.get("status", "?")[:20])
        rbuild = best["code"][5:] if best else ("failed" if rome.get(g) else "-")
        print(f"{g:15s} {int(o['n']):8,d}  {e if len(e) < 7 else 'NO':6s}  {f(to)} {status:5s}  {f(tr)} {rbuild:8s} {x_(sp)}   "
              f"{f(tpo)} {f(tpr)} {x_(psp)}")
        if len(e) >= 7: print(f"{'':17s}{e}")
    print()
    napprox = sum(exact(ours[g], ours_p.get(g)) == "~" for g in graphs)
    print(f"Exact: {nex}/{len(graphs)} graphs" + (f", {napprox} within 1% (~)" if napprox else "") +
          f". Our results checked and passing: {nok}/{len(graphs)}.")
    if napprox: print(METIS_NOTE)
    if rome:
        print(f"ROME: a passing build on {nrome_ok}/{len(graphs)} graphs.")
        if sp_here:
            sub = "ROME's 16 graphs" if set(graphs) >= set(ROME16) else f"the {len(sp_here)} graphs measured here"
            print(f"Speedup over ROME, geometric mean over {sub}: {gmean(sp_here):.2f}x here, {gmean(sp_ref):.2f}x in the "
                  f"reference runs. Faster here on {sum(x > 1 for x in sp_here)}/{len(sp_here)} graphs.")
    else:
        print("ROME was not run (quick.sh gpu --rome runs it on the same GPU).")

def report_cpu(d):
    ours_p, _ = reference_gpu()
    run = rows(os.path.join(d, "apsp.csv"))
    if not run: sys.exit(f"no results in {d}/apsp.csv")
    ref = [r for r in rows(os.path.join(DATA, "apsp.csv")) if r["commit"] == "lib-a"]
    t_ours = lambda r: sum(num(r[k]) or 0 for k in ("t_sym", "t_num", "t_closure"))   # as scripts/fig_apsp_combined.py
    by = lambda rs, nt: {r["graph"]: r for r in rs if r["provenance.threads"] == str(nt)}
    T = max(int(r["provenance.threads"]) for r in run)
    h1, hT, p1, p16 = by(run, 1), by(run, T), by(ref, 1), by(ref, 16)
    cpu = next((r.get("machine") for r in run if r.get("machine")), "?")
    print(f"This machine: {cpu}, ours on 1 and {T} threads, library {run[0]['commit']}.")
    print("Reference runs: AMD EPYC 7702, 1 core and one NUMA region of 16 cores, library lib-a (data/apsp.csv).")
    print("Times in seconds; speedup = n runs of Dijkstra's algorithm on one core / ours (symbolic + numeric + star);")
    print("exact = n, m and nnz(L+U) as in the reference rows of this version.\n")
    print(group(34, ("this machine", 51), ("reference", 19)))
    print(f"{'graph':15s} {'n':>8s}  {'exact':6s}  {'Dijkstra':>9s} {'ours 1':>9s} {'ours ' + str(T):>9s} {'x, 1':>8s} "
          f"{'x, ' + str(T):>8s}   {'x, 1':>9s} {'x, 16':>9s}")
    nex, graphs = 0, sorted(set(h1) | set(hT), key=lambda g: int((h1.get(g) or hT.get(g))["n"]))
    for g in graphs:
        a, b = h1.get(g), hT.get(g); r = a or b
        dij = num(a["t_dijkstra_all"]) if a else None
        o1 = t_ours(a) if a else None; oT = t_ours(b) if b else None
        pd = num(p1[g]["t_dijkstra_all"]) if g in p1 else None
        q1 = pd / t_ours(p1[g]) if pd and g in p1 else None; q16 = pd / t_ours(p16[g]) if pd and g in p16 else None
        e = exact(r, ours_p.get(g)); nex += e == "yes"
        print(f"{g:15s} {int(r['n']):8,d}  {e if len(e) < 7 else 'NO':6s}  {f(dij, 9, 3)} {f(o1)} {f(oT)} "
              f"{x_(dij / o1 if dij and o1 else None)} {x_(dij / oT if dij and oT else None)}   {x_(q1, 9)} {x_(q16, 9)}")
        if len(e) >= 7: print(f"{'':17s}{e}")
    print()
    napprox = sum(exact(h1.get(g) or hT.get(g), ours_p.get(g)) == "~" for g in graphs)
    print(f"Exact: {nex}/{len(graphs)} graphs" + (f", {napprox} within 1% (~)" if napprox else "") +
          ". Every result was checked against Dijkstra's algorithm on 64 columns (a failed check stops the run).")
    if napprox: print(METIS_NOTE)

if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("plan", "report") or sys.argv[2] not in ("cpu", "gpu"): sys.exit(__doc__)
    if sys.argv[1] == "plan": plan(sys.argv[2], float(sys.argv[3]))
    else: (report_gpu if sys.argv[2] == "gpu" else report_cpu)(sys.argv[3])
