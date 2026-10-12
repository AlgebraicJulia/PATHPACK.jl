#!/usr/bin/env python3
"""gpu_semirings.csv (closures) and gpu_semirings_kernels.csv (kernel rates) from the raw JSON of the semiring runs,
flattened as collect.py does for gpu_apsp.csv (META columns, then dotted names). Usage: build_csv.py RAWROOT OUTDIR"""
import csv, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))   # collect.py
from collect import rows_of, META
root, out = sys.argv[1], sys.argv[2]
for fam, name in (("gpuapsp", "gpu_semirings"), ("gpukernels", "gpu_semirings_kernels")):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*", "task_*", "out", fam + "_*.json"))):
        rel = os.path.relpath(f, root)
        for r in rows_of(f, rel, {}):
            r.pop("source", None); rows.append(r)
    meta = [m for m in META if m != "source"]
    lead = ["code", "semiring", "eltype", "graph", "n", "m", "tag", "variant", "t_cold.median", "check.ok", "status", "bithash"] if fam == "gpuapsp" else \
           ["op", "impl", "semiring", "T", "n", "tag", "variant", "gmuladd", "time"]
    cols = meta + [c for c in lead if any(c in r for r in rows)]
    cols += sorted({k for r in rows for k in r} - set(cols))
    path = os.path.join(out, name + ".csv")
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
    print(path, len(rows), "rows", len(cols), "columns")
