#!/usr/bin/env python3
"""Parse the C++ baseline logs (GAP, PHAST, CCH; written by hpc/baselines.sh) into data/external.csv.

One row per measurement: tool, graph, threads, placement, variant, metric, value (seconds, or queries/s).
Rows with the same (tool, graph, threads, placement, variant, metric) are replaced by newer ones.
Usage: parse_baselines.py --out data LOG... [--set key=value ...]
"""
import argparse, csv, json, os, re

KEY = ["tool", "graph", "threads", "placement", "variant", "metric"]
COLS = KEY + ["value", "source", "commit", "machine", "date", "job"]

def provenance(path, defaults):
    d = os.path.dirname(os.path.abspath(path))
    while True:
        p = os.path.join(d, "PROVENANCE.json")
        if os.path.isfile(p):
            m = dict(defaults); m.update(json.load(open(p))); return m
        if os.path.dirname(d) == d: return dict(defaults)
        d = os.path.dirname(d)

PHAST = re.compile(r"threads=(\d+) \| CH build ([\d.]+) s.*rank order per query: k=1 ([\d.]+), k=16 ([\d.]+), k=64 ([\d.]+) ms"
                   r" \| level order per query: k=1 ([\d.]+), k=16 ([\d.]+), k=64 ([\d.]+) ms \| one tree on \d+ threads \(level-parallel\): ([\d.]+) ms")
CCH = re.compile(r"order_ms=([\d.]+) cch_build_ms=([\d.]+) customize_ms=([\d.]+) \| query_us=([\d.]+)")
TRIAD = re.compile(r"^Triad:\s+([\d.]+)")
BOOSTW = re.compile(r"boost widest-path Dijkstra: n=(\d+) arcs=(\d+) sources=(\d+) (?:total_s=([\d.eE+-]+) )?mean ([\d.]+) us per source")
GAPAPSP = re.compile(r"gap_apsp: n=(\d+) threads=(\d+) delta=\S+ total_s=([\d.eE+-]+)")
DIMACS_CODES = {"NY", "BAY", "COL", "FLA", "NW", "NE", "CAL", "E", "W", "CTR", "USA"}
def gname(token):                       # DIMACS region codes in the logs mean the travel-time networks
    return "USA-t-" + token if token in DIMACS_CODES else token

def parse(path):
    rows = []; head = None; acc = {}                 # acc: GAP runs with several timing lines per header (averaged)
    for line in open(path):
        line = line.strip()
        if line.startswith("## "):
            t = [x.strip() for x in line[3:].split("|")]; head = t; continue
        if head is None: continue
        tool = head[0].split()[0]
        if tool == "gap":
            region = "region" in head[1] if len(head) > 3 else False
            threads = 16 if region else 1
            g = gname(head[-2]); delta = head[-1].replace("delta ", "delta=")
            m = re.match(r"Average Time:\s+([\d.eE+-]+)", line)
            if m: rows.append(dict(tool="gap", graph=g, threads=threads, placement="region" if region else "core", variant=delta, metric="latency_s", value=m.group(1)))
            m = re.match(r"region: (\d+) queries/s", line)
            if m: rows.append(dict(tool="gap", graph=g, threads=16, placement="region-processes", variant=delta, metric="qps", value=m.group(1)))
        elif tool == "gap_apsp" and GAPAPSP.search(line):   # full matrix: "gap_apsp: n=... threads=P delta=D total_s=T"
            m = GAPAPSP.search(line); k = int(m.group(2))
            rows.append(dict(tool="gap_apsp", graph=gname(head[2]), threads=k, placement="core" if k == 1 else "threads",
                             variant=";".join(h.replace("delta ", "delta=") for h in head[3:]), metric="total_s", value=m.group(3)))
        elif tool in ("gap_apsp", "gap_query"):       # "## gap_apsp | 1 thread | G | delta D", "## gap_apsp | P processes | ...", "## gap_query | T threads | ..."
            m = re.match(r"Average Time:\s+([\d.eE+-]+)", line)
            if not m: continue
            k = int(head[1].split()[0]); place = "processes" if "processes" in head[1] else ("core" if k == 1 else "region" if k <= 16 else "regions")
            acc.setdefault((tool, gname(head[2]), k, place, ";".join(h.replace("delta ", "delta=") for h in head[3:])), []).append(float(m.group(1)))
        elif tool == "phast":
            m = PHAST.search(line)
            if not m: continue
            g = gname(head[-1]); v = m.groups(); th = int(v[0])
            names = ["ch_build_s", "rank_k1_s", "rank_k16_s", "rank_k64_s", "level_k1_s", "level_k16_s", "level_k64_s", "tree_s"]
            vals = [float(v[1])] + [float(x) / 1e3 for x in v[2:]]
            for n_, x in zip(names, vals):
                rows.append(dict(tool="phast", graph=g, threads=th, placement="core" if th == 1 else "region", variant="", metric=n_, value=repr(x)))
        elif tool == "boost_widest":
            m = BOOSTW.search(line)
            if m:
                rows.append(dict(tool="boost_widest", graph=gname(head[-1]), threads=1, placement="core", variant="", metric="latency_s", value=repr(float(m.group(5)) / 1e6)))
                if m.group(4): rows.append(dict(tool="boost_widest", graph=gname(head[-1]), threads=1, placement="core", variant="full matrix", metric="total_s", value=m.group(4)))
            elif line.startswith("boost check:"): rows.append(dict(tool="boost_widest", graph=gname(head[-1]), threads=1, placement="core", variant="", metric="check_pass", value="1" if "PASS" in line else "0"))
        elif tool == "stream":
            m = TRIAD.match(line)
            if not m: continue
            th = 16 if "16" in head[1] else 1
            rows.append(dict(tool="stream", graph="-", threads=th, placement="core" if th == 1 else "region", variant="", metric="triad_GBps", value=repr(float(m.group(1)) / 1e3)))
        elif tool == "cch":
            m = CCH.search(line)
            if not m: continue
            g = gname(head[-1]); o, b, c, q = map(float, m.groups())
            for n_, x in [("order_s", o / 1e3), ("build_s", b / 1e3), ("customize_s", c / 1e3), ("query_s", q / 1e6)]:
                rows.append(dict(tool="cch", graph=g, threads=1, placement="core", variant="", metric=n_, value=repr(x)))
    for (tool, g, k, place, variant), v in acc.items():
        rows.append(dict(tool=tool, graph=g, threads=k, placement=place, variant=variant, metric="latency_s", value=repr(sum(v) / len(v))))
        rows.append(dict(tool=tool, graph=g, threads=k, placement=place, variant=variant, metric="samples", value=str(len(v))))
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("logs", nargs="+")
    ap.add_argument("--set", nargs="*", default=[]); a = ap.parse_args()
    defaults = dict(kv.split("=", 1) for kv in a.set)
    path = os.path.join(a.out, "external.csv"); table = {}
    if os.path.isfile(path):
        for r in csv.DictReader(open(path)): table[tuple(r[k] for k in KEY)] = r
    n = 0
    for log in a.logs:
        meta = provenance(log, defaults)
        for r in parse(log):
            r = {k: str(v) for k, v in r.items()}
            r.update(source=log, **{k: meta.get(k, "") for k in ("commit", "machine", "date", "job")})
            table[tuple(r[k] for k in KEY)] = r; n += 1
    os.makedirs(a.out, exist_ok=True)
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=COLS); w.writeheader(); w.writerows(sorted(table.values(), key=lambda r: tuple(r[k] for k in KEY)))
    print(f"{path}: {len(table)} rows ({n} parsed now)")

if __name__ == "__main__":
    main()
