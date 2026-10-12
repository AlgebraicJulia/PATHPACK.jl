import json, glob, math, sys, collections
root = sys.argv[1]
R = {}
for f in glob.glob(root + '/raw/*/task_*/out/gpuapsp_*.json'):
    r = json.load(open(f))
    if 't_total' not in r: print('NO TIME', f, r.get('status')); continue
    R[(r['tag'], r.get('variant', 'default'), r['semiring'] + ':' + r['eltype'], r['graph'])] = r
gm = lambda x: math.exp(sum(map(math.log, x)) / len(x))
def table(runs, title):
    print('==', title)
    cfgs = sorted({k[2] for k in R if k[:2] in runs}, key=lambda c: (c.split(':')[1] != 'Float32', c))
    for c in cfgs:
        rat, ok, nck, fails, mx = [], 0, 0, [], 0.0
        for k, r in R.items():
            if k[:2] not in runs or k[2] != c: continue
            b = R[(k[0], k[1], 'MinPlus:Float32', k[3])]
            rat.append((r['t_total'] / b['t_total'], k[3]))
            ck = r.get('check', {})
            if 'ok' in ck:
                nck += 1; ok += ck['ok']; mx = max(mx, ck.get('cpu_closure_maxrel', 0.0))
                if not ck['ok']: fails.append(k[3])
        x = [a for a, _ in rat]; lo = min(rat); hi = max(rat)
        print(f"  {c:18s} n={len(x):2d} gm {gm(x):.2f}  range {lo[0]:.2f} ({lo[1]}) - {hi[0]:.2f} ({hi[1]})  check {ok}/{nck} maxrel {mx:.1e} {fails if fails else ''}")
table({('base', 'default'), ('base2', 'default')}, 'lib-e: t / t(MinPlus Float32, same job)')
table({('sr', 'pair1'), ('sr2', 'pair1')}, 'lib-f (pair1)')
table({('sr', 'pair0')}, 'lib-f, HEURISTIC_PAIR off (pair0)')
# absolute: tropical times and fix effect
print('== MinPlus Float32 cold median (s): base / sr pair0 / sr pair1')
for g in sorted({k[3] for k in R}, key=lambda g: R[('base', 'default', 'MinPlus:Float32', g)]['n']):
    a = [R[(t, v, 'MinPlus:Float32', g)]['t_total'] for t, v in (('base', 'default'), ('sr', 'pair0'), ('sr', 'pair1'))]
    print(f"  {g:15s} n={R[('base','default','MinPlus:Float32',g)]['n']:7d} " + ' '.join(f'{x:8.4f}' for x in a))
