#!/usr/bin/env python3
"""Compare reruns (results/*.csv, written by hpc/run.sh and hpc/run_gpu.sh) with the reference measurements (data/).

For every experiment in results/ that the reference runs also measured (same code, graph, GPU and GPU count, kernel and
size, ...):
  - times: the ratio rerun / reference of each timed column (medians over the matching rows on both sides);
    SLOWER if the rerun is more than --tol slower, faster if it is more than --tol faster (not a failure);
  - exact quantities (n, m, nnz(L+U), fill): must be identical (the symbolic analysis is deterministic), except
    nnz(L+U) within 1% on graphs with several components, a note: their components are ordered concurrently, and
    METIS, which BestFill(AMF(), METIS()) tries, is not deterministic on several threads at once;
  - checks: every rerun row must pass its own correctness check (check.ok, status ok).
Differences are failures when the reference runs used the same library version (lib-g) or the same kernels (the
semirings: lib-f's GPU kernels are lib-g's, and their closures are compared as the cost relative to min-plus Float32 in
the same job). Against another version (the CPU rows: lib-a, lib-b) they are reported as notes: the code here relabels
the vertices within each component before ordering, so AMF and METIS break ties differently and the fill, and with it
the times, differ by a few percent.
The reference rows are those of the library version this repository holds (lib-g) where they exist; the semirings were
measured at lib-f, whose GPU kernels are those of lib-g (variant pair1 there = the default here); the CPU experiments at
lib-a/lib-b (the CPU code here is lib-g's: same algorithms, later performance work, so it may be faster).
Usage: regress.py [--tol 0.15] [--results results] [--data data] [--all]      (exit status 1 if anything failed)
"""
import argparse, csv, glob, os, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=0.15, help="relative time tolerance (B200 node-to-node noise is ~15%%)")
ap.add_argument("--results", default="results"); ap.add_argument("--data", default="data")
ap.add_argument("--all", action="store_true", help="print every comparison, not only the worst")
A = ap.parse_args()

def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def ours_metric(r):                                  # the time reported for each code (scripts/gpu_data.py)
    c = r.get("code", "")
    if c.endswith("-phases"): return "phases.total"   # the per-step rows: the whole call, the median of 7 synchronized calls
    return "t_match.median" if c.startswith("rome") else "t_cold.median" if c.startswith("ours") else "t.median"

SR_REF = lambda r: (r.get("tag") in ("sr", "sr2") and r.get("variant") == "pair1")
FAMILIES = [
    # name, file, reference rows, key columns, timed columns (a function of the row, or a list), exact columns
    ("GPU all-pairs (one GPU)", "gpu/gpu_apsp.csv", lambda r: r["commit"] == "lib-g",
     ["code", "graph", "gpu.kind", "eltype"], ours_metric, ["n", "m", "nnzLU"], True),
    ("GPU all-pairs (several GPUs)", "gpu/gpu_multigpu.csv", lambda r: r["commit"] == "lib-g",
     ["code", "graph", "gpu.count", "gpu.kind"], ["t.median", "t_closure", "t_reuse"], ["n", "m"], True),
    ("GPU kernels", "gpu/gpu_kernels.csv", lambda r: r["commit"] == "lib-g",
     ["op", "impl", "semiring", "T", "n", "gpu.kind"], ["time"], [], True),
    ("GPU semirings (closures)", "gpu/gpu_semirings.csv", SR_REF,
     ["code", "semiring", "eltype", "graph"], ["rel_minplus"], ["n", "m"], True),
    ("GPU semirings (kernels)", "gpu/gpu_semirings_kernels.csv", SR_REF,
     ["op", "impl", "semiring", "T", "n"], ["time"], [], True),
    ("CPU all-pairs", "apsp.csv", lambda r: True,
     ["graph", "provenance.threads"], ["t_sym", "t_num", "t_inverse", "t_solve", "t_closure", "t_dijkstra_all"],
     ["n", "m", "nnzLU"], False),
    ("CPU queries", "queries.csv", lambda r: True, ["graph"],
     ["t_sym", "t_num.1", "t_num.16", "single.source_nt1.median", "single.source_nt16.median",
      "single.target_nt1.median", "single.target_nt16.median"], ["n", "m", "nnzLU"], False),
    ("CPU bottleneck", "bottleneck.csv", lambda r: True, ["graph", "provenance.threads"],
     ["t_sym", "t_num", "t_closure"], ["n", "m"], False),
]

def rel_minplus(rs):                     # a semiring's closure time over min-plus Float32's, same graph and job
    base = {}
    for r in rs:
        if r.get("semiring") == "MinPlus" and r.get("eltype") == "Float32" and num(r.get("t_cold.median")):
            base[(r["graph"], r.get("commit"), r.get("tag"), r.get("variant"), r.get("job"))] = num(r["t_cold.median"])
    for r in rs:
        b = base.get((r["graph"], r.get("commit"), r.get("tag"), r.get("variant"), r.get("job")))
        t = num(r.get("t_cold.median"))
        r["rel_minplus"] = t / b if b and t and not (r.get("semiring") == "MinPlus" and r.get("eltype") == "Float32") else ""
    return rs

def key(r, cols):
    return tuple(r.get(c, "") for c in cols)

def passed(r):
    if r.get("status", "ok") not in ("ok", ""): return False
    return str(r.get("check.ok", "true")).lower() not in ("false", "0")

fails, out = 0, []
print(f"tolerance {A.tol:.0%}: SLOWER = rerun more than {A.tol:.0%} slower than the reference (a failure); faster = more than "
      f"{A.tol:.0%} faster (reported, not a failure)\n")
for name, f, ref_rows, kcols, tcols, exact, strict in FAMILIES:
    new = rows(os.path.join(A.results, os.path.basename(f)))
    if not new: continue
    old = [r for r in rows(os.path.join(A.data, f)) if ref_rows(r)]
    if tcols == ["rel_minplus"]: new, old = rel_minplus(new), rel_minplus(old)
    kc = [c for c in kcols if c in (new[0].keys() & (old[0].keys() if old else set()))]
    bykey = {}
    for r in old: bykey.setdefault(key(r, kc), []).append(r)
    newby = {}
    for r in new: newby.setdefault(key(r, kc), []).append(r)
    ratios, bad, notes, vnotes, mnotes, unmatched = [], [], [], [], [], 0
    for k, rs in sorted(newby.items()):
        ok = [r for r in rs if passed(r)]
        ours = not rs[0].get("code") or rs[0]["code"].startswith("ours")
        if len(ok) < len(rs):
            if ours or any(passed(r) for r in bykey.get(k, [])):       # ours, or a baseline that completed in the reference runs
                bad.append(f"{k}: {len(rs) - len(ok)} of {len(rs)} rows failed their check or did not finish")
            else: notes.append(f"{k}: did not complete, as in the reference runs")
        if not ok: continue
        if k not in bykey: unmatched += 1; continue
        os_ = [r for r in bykey[k] if passed(r)]
        if not os_: notes.append(f"{k}: completed now, not in the reference runs"); continue
        rs = ok
        vo, vn = sorted({r.get("commit", "") for r in os_}), sorted({r.get("commit", "") for r in rs})
        hard = strict or vo == vn                 # failures count; else differences are version notes
        for c in exact:
            a = {r.get(c, "") for r in os_ if r.get(c, "")}; b = {r.get(c, "") for r in rs if r.get(c, "")}
            if a and b and a != b:
                msg = f"{k}: {c} reference {sorted(a)} ({'/'.join(vo)}) rerun {sorted(b)} ({'/'.join(vn)})"
                close = c == "nnzLU" and all(abs(num(x) / num(y) - 1) <= 0.01 for x in a for y in b)
                if hard and close and max(num(r.get("ncc")) or 1 for r in rs + os_) > 1:   # METIS_NOTE below
                    mnotes.append(msg)
                else:
                    (bad if hard else vnotes).append(msg)
        cols = [tcols(rs[0])] if callable(tcols) else tcols
        for c in cols:
            a = [num(r.get(c)) for r in os_]; b = [num(r.get(c)) for r in rs]
            a = [x for x in a if x is not None and x > 0]; b = [x for x in b if x is not None and x > 0]
            if not a or not b: continue
            x = statistics.median(b) / statistics.median(a)
            ratios.append(x); out.append((name, k, c, statistics.median(a), statistics.median(b), x))
            if x > 1 + A.tol:
                (bad if hard else vnotes).append(f"{k} {c}: {'SLOWER' if hard else 'slower'} {x:.2f}x (reference {statistics.median(a):.4g}, rerun {statistics.median(b):.4g})")
    g = statistics.geometric_mean(ratios) if ratios else float("nan")
    print(f"== {name}: {len(newby)} experiments rerun, {len(newby) - unmatched} also in the reference runs, {len(ratios)} timings compared; "
          f"rerun / reference: geometric mean {g:.3f}, range {min(ratios, default=float('nan')):.2f}-{max(ratios, default=float('nan')):.2f}; "
          f"faster than the reference by more than {A.tol:.0%}: {sum(x < 1 / (1 + A.tol) for x in ratios)}")
    for b in bad: print("   FAIL", b)
    if mnotes:
        print(f"   {len(mnotes)} differences in nnz(L+U) under 1%, on graphs with several components (notes: such graphs have "
              "their components ordered concurrently, and METIS, which BestFill tries, is not deterministic then):")
        for b in mnotes: print("     ", b)
    if vnotes:
        print(f"   {len(vnotes)} differences against another library version (notes, not failures; see the top of this script):")
        for b in vnotes[:8]: print("     ", b)
        if len(vnotes) > 8: print(f"      ... and {len(vnotes) - 8} more")
    for b in notes[:5]: print("   note", b)
    if len(notes) > 5: print(f"   note ... and {len(notes) - 5} more baselines that did not complete, as in the reference runs")
    fails += len(bad)

if out:
    os.makedirs(A.results, exist_ok=True)
    with open(os.path.join(A.results, "regress.csv"), "w", newline="") as h:
        w = csv.writer(h); w.writerow(["family", "key", "column", "reference", "rerun", "ratio"])
        for n_, k, c, a, b, x in out: w.writerow([n_, "|".join(k), c, f"{a:.6g}", f"{b:.6g}", f"{x:.4f}"])
    show = out if A.all else sorted(out, key=lambda t: -abs(t[5] - 1))[:15]
    print(f"\n{'all' if A.all else 'the 15 largest deviations of'} {len(out)} comparisons (also in {A.results}/regress.csv):")
    for n_, k, c, a, b, x in show:
        print(f"  {x:6.2f}x  {n_}: {' '.join(k)}  {c}  reference {a:.4g}  rerun {b:.4g}")
print(f"\n{'PASS' if fails == 0 else f'FAIL ({fails} problems)'}")
sys.exit(1 if fails else 0)
