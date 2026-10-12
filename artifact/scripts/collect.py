#!/usr/bin/env python3
"""Collect raw benchmark results (JSON) into the benchmark CSV files.

Each raw JSON file becomes one CSV row (a JSON list becomes one row per element), with nested fields
flattened into dotted column names (sweep.64.row). Rows carry their provenance -- library commit,
machine, date, job -- read from the nearest PROVENANCE.json above the file, or given on the command line.

Merging: rows are keyed by file name and library commit; a newer file with the same name and commit replaces
the old row, and every other row is kept (rows of another commit stay). So collecting the results of one target updates exactly the rows it produced.

Usage:
  collect.py --out data --results DIR[:DIR...] [--scale DIR[:DIR...]] [--set key=value ...] [--fresh]
Later directories take precedence over earlier ones for files of the same name.
"""
import argparse, csv, glob, json, os, re, sys

# experiment id -> CSV file; experiments in SCALE are read only from --scale directories
FAMILY = {"apsp": "apsp", "queries": "queries", "bottleneck": "bottleneck", "kernels": "kernels",     # the benchmark suite
          "gpuapsp": "gpu_apsp", "gpukernels": "gpu_kernels", "gpumulti": "gpu_multigpu",               # GPU benchmarks (hpc/run_gpu.sh)
          # earlier experiments (data/legacy):
          "exp1": "kernels", "exp13": "factor", "exp16": "queries", "exp4": "apsp", "exp9": "dense",
          "exp5e": "diffcon", "exp14a": "widest", "exp14b": "reach", "exp14c": "reach_single",
          "exp5b": "mmatrix", "exp5c": "reach_single_old", "exp2": "orderings",
          "exp18": "scaling", "exp7c": "scaling_factor", "exp12": "scaling_rows",
          "regions": "regions", "exp19": "bottleneck", "exp3": "legacy_queries", "exp3b": "legacy_queries_b", "exp11": "legacy_queries_elim"}
SCALE = {"exp18", "exp7c", "exp12"}
META = ["file", "source", "commit", "machine", "date", "job"]

SUBDIRS = {"exp2_rows": "exp2"}                                     # result subfolders -> experiment id

def family(rel):
    parts = rel.split("/")
    if len(parts) == 2 and re.fullmatch(r"region_\d+", parts[0]): return "regions"   # eight regions at once (target: regions)
    if len(parts) == 2: return SUBDIRS.get(parts[0])
    exp = parts[0].split("_")[0]
    return exp if exp in FAMILY else None

def cell(v):
    if v is None: return "null"
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, float): return repr(v)
    if isinstance(v, (list, dict)): return "json:" + json.dumps(v, separators=(",", ":"))
    return str(v)

def flat(x, prefix, out):
    if isinstance(x, dict) and x:
        for k, v in x.items(): flat(v, f"{prefix}{k}.", out)
    else:
        out[prefix[:-1]] = cell(x)

def provenance(path, defaults):
    d = os.path.dirname(os.path.abspath(path))
    while True:
        p = os.path.join(d, "PROVENANCE.json")
        if os.path.isfile(p):
            meta = dict(defaults); meta.update(json.load(open(p))); return meta
        if os.path.dirname(d) == d: return dict(defaults)
        d = os.path.dirname(d)

def rows_of(path, rel, defaults):
    rec = json.load(open(path)); meta = provenance(path, defaults)
    base = {"file": rel, "source": path, **{k: meta.get(k, "") for k in META[2:]}}
    items = rec if isinstance(rec, list) else [rec]
    out = []
    for i, r in enumerate(items):
        row = dict(base)
        if isinstance(rec, list): row["_row"] = str(i)
        flat(r, "", row); out.append(row)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--results", default=""); ap.add_argument("--scale", default="")
    ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    defaults = dict(kv.split("=", 1) for kv in a.set)
    new = {}                                                       # family -> file -> rows (later dirs override)
    for kind, dirs in (("results", a.results), ("scale", a.scale)):
        for d in [x for x in dirs.split(":") if x]:
            files = sorted(glob.glob(os.path.join(d, "*.json")) + [f for s in SUBDIRS for f in glob.glob(os.path.join(d, s, "*.json"))]
                           + glob.glob(os.path.join(d, "region_*", "*.json")))
            for f in files:
                rel = os.path.relpath(f, d); fam = family(rel)
                if fam is None or (fam in SCALE) != (kind == "scale"): continue
                new.setdefault(FAMILY[fam], {})[rel] = rows_of(f, rel, defaults)
    os.makedirs(a.out, exist_ok=True)
    for name, files in sorted(new.items()):
        path = os.path.join(a.out, name + ".csv"); old = {}
        if os.path.isfile(path) and not a.fresh:
            for r in csv.DictReader(open(path)): old.setdefault((r["file"], r.get("commit", "")), []).append(r)
        # (keyed by file name and library commit: a run at another commit adds its rows next to the earlier ones)
        old.update({(f, rs[0].get("commit", "") if rs else ""): rs for f, rs in files.items()})
        rows = [r for f in sorted(old) for r in old[f]]
        cols = META + (["_row"] if any("_row" in r for r in rows) else []) + sorted({k for r in rows for k in r} - set(META) - {"_row"})
        with open(path, "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
        print(f"{path}: {len(rows)} rows ({len(files)} files collected now)")

if __name__ == "__main__":
    main()
