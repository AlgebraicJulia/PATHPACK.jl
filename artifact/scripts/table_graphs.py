"""The table of all test graphs (LaTeX): n and d-bar = m/n from the data, fill = nnz(L+U)/nnz(A) with AMF
(GPU runs: code ours-amf at commit lib-d, else its latest; DIMACS: data/queries.csv), no n/rho (ROME lists it for its 16 graphs only)."""
import csv, sys
from collections import defaultdict
APSP = ["USpowerGrid", "delaunay_n13", "G67", "c-42", "CurlCurl_0", "fe_sphere", "nd6k", "delaunay_n14", "OPF_6000",
        "EB_8192_256", "EB_16384_64", "rgg2d_14", "rgg3d_14", "hypercube_14", "lpl1", "email-Enron", "t60k", "wing",
        "delaunay_n16", "c-69", "oilpan", "finan512", "fe_tooth", "onera_dual", "net4-1", "luxembourg_osm"]  # scripts/gpu_data.py
FAP = ["598a", "luxembourg_osm", "web-sk-2005", "usroads-48", "fe_ocean"]
# ROME's Table 2 (n/rho), read from its PDF: G0..G15
ROME = {"USpowerGrid": 549.0, "delaunay_n13": 88.1, "G67": 50.0, "c-42": 143.4, "CurlCurl_0": 50.2, "fe_sphere": 71.6,
        "nd6k": 5.7, "lpl1": 377.4, "email-Enron": 134.4, "t60k": 759.6, "wing": 105.1, "delaunay_n16": 293.7,
        "c-69": 372.7, "oilpan": 110.9, "finan512": 1495.0, "fe_tooth": 89.9}
info = {}
best = defaultdict(dict)
for r in csv.DictReader(open("data/gpu/gpu_apsp.csv")):
    if r.get("fill") and r["ordering"] == "AMF()" and r["code"] == "ours-amf":
        best[r["graph"]][r["commit"]] = (int(r["n"]), int(r["m"]), float(r["fill"]), r["date"])
auto = {}  # graphs run only with the automatic ordering (the larger graphs): its latest run, and the ordering it chose
for r in csv.DictReader(open("data/gpu/gpu_apsp.csv")):
    if r.get("fill") and r["code"] == "ours-auto" and r["graph"] not in best:
        if r["graph"] not in auto or r["date"] > auto[r["graph"]][3]:
            auto[r["graph"]] = (int(r["n"]), int(r["m"]), float(r["fill"]), r["date"], r["auto_choice"])
NOTAMF = {g for g, v in auto.items() if v[4] != "AMF"}
for g, v in auto.items():
    best[g] = {"auto": v[:4]}
for g, by in best.items():
    info[g] = by["lib-d"][:3] if "lib-d" in by else max(by.values(), key=lambda v: v[3])[:3]
for r in csv.DictReader(open("data/queries.csv")):
    if r["graph"].startswith("USA-t-"):
        info[r["graph"]] = (int(r["n"]), int(r["m"]), float(r["fill"]))
LARGER = sorted([g for g in info if not g.startswith("USA-t-") and g not in APSP and g not in FAP], key=lambda g: info[g][0])
DIMACS = sorted([g for g in info if g.startswith("USA-t-")], key=lambda g: info[g][0])
for g in APSP + FAP:
    assert g in info, g
thou = lambda x: f"{x:,}".replace(",", "{,}")
def num(x):
    return f"{x:.1f}" if x < 10 else thou(round(x))
def row(set_, g):
    n, m, fill = info[g]
    name = g.replace("_", "\\_") + ("$^\\dagger$" if g == "luxembourg_osm" else "")
    mark = "$^\\ddagger$" if g in NOTAMF else ""
    return f"{set_:<13}& {name} & {thou(n)} & {m / n:.1f} & {num(fill)}{mark} \\\\"
groups = [("SuperFW/ROME", sorted(APSP, key=lambda g: info[g][0])),
          ("Fast-APSP", sorted([g for g in FAP if g != "luxembourg_osm"], key=lambda g: info[g][0])),
          ("Larger", LARGER), ("DIMACS", DIMACS)]
L = ["\\begin{table}[t]", "\\caption{Test graphs.}", "\\label{tab:graphs}",
     "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
     "Set & Graph & $n$ & $\\bar d$ & fill \\\\"]
for k, (s, gs) in enumerate(groups):
    L.append("\\midrule")
    for i, g in enumerate(gs):
        L.append(row(s if i == 0 else "", g))
L += ["\\bottomrule", "\\end{tabular}",
      "\\par\\smallskip{\\footnotesize $\\bar d$: average degree; fill: $\\nnz(L+U)/\\nnz(A)$ with AMF. $^\\dagger$Also one of Fast-APSP's graphs. $^\\ddagger$With the ordering chosen as in Section~\\ref{sec:setup}, not AMF.\\par}",
      "\\end{table}"]
sys.stdout.write("\n".join(L) + "\n")
print(f"% {sum(len(g) for _, g in groups)} rows; larger: {LARGER}", file=sys.stderr)
