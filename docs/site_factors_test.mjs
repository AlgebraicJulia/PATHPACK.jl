// Check the browser's PATHPACK query (src/components/lib/pathpack.ts) against PATHPACK.jl's rmul!.
//
//   julia --project=docs docs/site_factors.jl REFDIR    # factor files, and reference rows in REFDIR
//   node docs/site_factors_test.mjs REFDIR
//
// For each demo: the graph site_factors.jl factorized is the one the component builds (same arcs, same Float32
// weights); every reference row of rmul! (PATHPACK.jl, in Julia) equals the TypeScript row bit for bit; and the rows
// agree with the components' Dijkstra (src/components/lib/graph.ts) up to Float32 rounding.
import { readFileSync } from 'node:fs'
import { gunzipSync } from 'node:zlib'
import { join } from 'node:path'
import { fromBin, distancesFrom } from './src/components/lib/pathpack.ts'
import { csr, dijkstra } from './src/components/lib/graph.ts'

const DATA = join(import.meta.dirname, 'src', 'public', 'data')
const REF = process.argv[2]
if (!REF) { console.error('usage: node docs/site_factors_test.mjs REFDIR'); process.exit(2) }

// the format of src/components/lib/bin.ts (loadBin), from a file
const CTORS = { float32: Float32Array, float64: Float64Array, uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, int8: Int8Array, int16: Int16Array, int32: Int32Array }
function readBin(path) {
  const z = gunzipSync(readFileSync(path)), buf = z.buffer.slice(z.byteOffset, z.byteOffset + z.byteLength)
  const hl = new Uint32Array(buf, 0, 1)[0]
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)))
  const base = 4 + hl, a = {}
  for (const [k, [t, off, n]] of Object.entries(head.arrays)) a[k] = new CTORS[t](buf.slice(base + off, base + off + n * CTORS[t].BYTES_PER_ELEMENT))
  return { meta: head.meta, a }
}

// the graphs as the components build them (code copied from ParisLive.vue and MeshPaint.vue)
function parisGraph() {
  const { meta, a } = readBin(join(DATA, 'paris.bin.gz'))
  const n = meta.n, m = meta.m
  const w = new Float32Array(m); for (let e = 0; e < m; e++) w[e] = a.wm[e] / 1000
  return { n, I: a.ei, J: a.ej, W: w }
}
function bunnyGraph() {
  const { meta, a } = readBin(join(DATA, 'bunny.bin.gz'))
  const n = meta.n, nf = meta.f, tri = a.tri
  const P = new Float32Array(3 * n); for (let i = 0; i < 3 * n; i++) P[i] = meta.lo[i % 3] + a.pos[i] / meta.scale
  const seen = new Set(), I = [], J = [], W = []
  for (let f = 0; f < nf; f++) {
    const [i, j, k] = [tri[3 * f], tri[3 * f + 1], tri[3 * f + 2]]
    for (const [p, q] of [[i, j], [j, k], [k, i]]) {
      const key = Math.min(p, q) * 65536 + Math.max(p, q); if (seen.has(key)) continue; seen.add(key)
      I.push(p); J.push(q); W.push(Math.hypot(P[3 * p] - P[3 * q], P[3 * p + 1] - P[3 * q + 1], P[3 * p + 2] - P[3 * q + 2]))
    }
  }
  return { n, I, J, W }
}

let failed = false
for (const [name, graph] of [['paris', parisGraph], ['bunny', bunnyGraph]]) {
  const F = fromBin(readBin(join(DATA, `${name}-lu.bin.gz`)))
  const ref = readBin(join(REF, `${name}-ref.bin.gz`)), n = ref.meta.n, k = ref.meta.k
  const G = graph(), g = csr(G.n, G.I, G.J, G.W)

  // 1. the factorized graph is the browser's graph: the same arcs, and the Float32 weights csr stores
  let same = G.n === n && G.I.length === ref.a.I.length
  for (let e = 0; same && e < G.I.length; e++) same = G.I[e] === ref.a.I[e] && G.J[e] === ref.a.J[e] && Math.fround(G.W[e]) === ref.a.W[e]
  console.log(`${name}: graph of ${n} vertices and ${G.I.length} edges ${same ? 'identical to' : 'DIFFERENT from'} the one factorized`)
  if (!same) failed = true

  // 2. rows of A* from the TypeScript port against rmul! in Julia, bit for bit
  const row = new Float32Array(n), bits = new Uint32Array(row.buffer)
  const refBits = new Uint32Array(ref.a.rows.buffer)
  let bad = 0, badEntries = 0
  for (let r = 0; r < k; r++) {
    distancesFrom(F, ref.a.src[r], row)
    let diff = 0
    for (let j = 0; j < n; j++) if (bits[j] !== refBits[r * n + j]) diff++
    if (diff) { bad++; badEntries += diff }
  }
  console.log(`${name}: ${k} sources (first ${ref.a.src[0]}, last ${ref.a.src[1]}, ...): ${k - bad} rows identical bit for bit to rmul!, ${bad} differ (${badEntries} entries)`)
  if (bad) failed = true

  // 3. against the components' Dijkstra (Float64 sums of the Float32 weights), rounded to Float32
  const d = new Float64Array(n)
  let exact = 0, total = 0, maxAbs = 0, maxRel = 0, infMismatch = 0
  for (let r = 0; r < k; r++) {
    const s = ref.a.src[r]
    dijkstra(g, [s], 'sum', d)
    for (let j = 0; j < n; j++) {
      const p = ref.a.rows[r * n + j], q = Math.fround(d[j]); total++
      if (p === Infinity || q === Infinity) { if (p !== q) infMismatch++; else exact++; continue }
      if (p === q) exact++
      const ad = Math.abs(p - q); maxAbs = Math.max(maxAbs, ad); if (q > 0) maxRel = Math.max(maxRel, ad / q)
    }
  }
  console.log(`${name}: against Dijkstra: ${exact} of ${total} distances equal in Float32, max abs diff ${maxAbs.toExponential(2)}, max rel diff ${maxRel.toExponential(2)}, reachability mismatches ${infMismatch}`)
  if (infMismatch || maxRel > 1e-5) failed = true

  // 4. time one row
  for (let r = 0; r < 20; r++) distancesFrom(F, ref.a.src[r % k], row)
  let t = performance.now(), reps = 300
  for (let r = 0; r < reps; r++) distancesFrom(F, ref.a.src[r % k], row)
  const tp = (performance.now() - t) / reps
  t = performance.now()
  for (let r = 0; r < reps; r++) dijkstra(g, [ref.a.src[r % k]], 'sum', d)
  const td = (performance.now() - t) / reps
  console.log(`${name}: one row in Node: PATHPACK query ${tp.toFixed(3)} ms, Dijkstra ${td.toFixed(3)} ms`)
}
if (failed) { console.log('FAILED'); process.exit(1) }
console.log('all checks passed')
