#!/usr/bin/env python3
"""Compare two macro files (\\def or \\newcommand): same names, same values. Usage: compare_macros.py OLD NEW"""
import re, sys
def load(p):
    s = open(p).read()
    d = dict(re.findall(r"\\def\\(\w+)\{(.*)\}\s*$", s, re.M))
    d.update(re.findall(r"\\newcommand\*?\{\\(\w+)\}(?:\[\d\])?\{(.*)\}\s*$", s, re.M))
    return d
old, new = load(sys.argv[1]), load(sys.argv[2])
diff = [(k, old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
print(f"{sys.argv[2]}: {len(new)} macros, {len(old)} in the reference; {len(diff)} differ")
for k, a, b in diff[:60]: print(f"  {k}: reference {a!r}  now {b!r}")
sys.exit(1 if diff else 0)
