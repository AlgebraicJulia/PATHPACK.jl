# rome_report.py [CSV] [VERSION]: ours-auto vs ROME's best build (same job) on one B200, per graph and per set (the
# speedups over ROME). Defaults: results/gpu_apsp.csv (your reruns; data/gpu/gpu_apsp.csv: the reference runs), lib-g.
import csv, math, sys
CSV = sys.argv[1] if len(sys.argv) > 1 else "results/gpu_apsp.csv"; VER = sys.argv[2] if len(sys.argv) > 2 else "lib-g"
rows = [r for r in csv.DictReader(open(CSV)) if r["commit"] == VER and r["gpu.kind"] == "b200"]
R16 = "USpowerGrid delaunay_n13 G67 c-42 CurlCurl_0 fe_sphere nd6k lpl1 email-Enron t60k wing delaunay_n16 c-69 oilpan finan512 fe_tooth".split()
A26 = "USpowerGrid delaunay_n13 G67 c-42 CurlCurl_0 fe_sphere nd6k delaunay_n14 OPF_6000 EB_8192_256 EB_16384_64 rgg2d_14 rgg3d_14 hypercube_14 lpl1 email-Enron t60k wing delaunay_n16 c-69 oilpan finan512 fe_tooth onera_dual net4-1 luxembourg_osm".split()
LG = "598a luxembourg_osm web-sk-2005 usroads-48 fe_ocean 144 Dubcova3 wave c-73 cont-300 thermomech_dM".split()
f = lambda x: float(x) if x not in ("", None) else None
ours, cpuf, rome = {}, {}, {}
for r in rows:
    g, c = r["graph"], r["code"]
    if c == "ours-auto": ours[g] = r
    elif c == "ours-auto-cpufactor": cpuf[g] = r
    elif c.startswith("rome-"): rome[g] = r
def line(g):
    o = ours.get(g); ro = rome.get(g); cp = cpuf.get(g)
    to = f(o["t_total"]) if o else None
    tr = f(ro["t_match.median"]) if ro and ro["status"].startswith("ok") else None
    sp = tr / to if to and tr else None
    rng = lambda r, k: f"{f(r[k+'.min']):.4f}-{f(r[k+'.max']):.4f}" if r and r.get(k+'.min') else ""
    return g, o, ro, cp, to, tr, sp, rng
print(f"{'graph':15s} {'n':>7s} {'ours':>8s} {'(min-max)':>17s} {'ROME':>8s} {'build':8s} {'(min-max)':>17s} {'x':>6s} {'cpufac':>8s} {'choice':6s} status")
for g in dict.fromkeys(A26 + LG):
    g, o, ro, cp, to, tr, sp, rng = line(g)
    print(f"{g:15s} {o['n'] if o else '':>7s} {to or float('nan'):8.4f} {rng(o, 't_cold'):>17s} {tr or float('nan'):8.4f} {(ro['code'][5:] if ro else '-'):8s} {rng(ro, 't_match'):>17s} {sp or float('nan'):6.2f} {f(cp['t_total']) if cp else float('nan'):8.4f} {o['auto_choice'] if o else '':6s} {o['status'] if o else 'MISSING'}/{ro['status'] if ro else '-'} {'' if not o else 'chk ' + o.get('check.ok','')}")
for name, S in (("ROME's 16", R16), ("all 26", A26), ("large set", LG)):
    sps = [line(g)[6] for g in S]; sps = [x for x in sps if x]
    gm = math.exp(sum(map(math.log, sps)) / len(sps)) if sps else float('nan')
    print(f"{name:10s}: geomean {gm:.2f}x over {len(sps)} graphs, wins {sum(x > 1 for x in sps)}/{len(sps)}, min {min(sps):.2f} max {max(sps):.2f}")
