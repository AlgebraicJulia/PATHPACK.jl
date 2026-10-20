// The model behind the How it works explainer: a small road-like graph, three elimination orders, the symbolic
// elimination of each (fill-in, elimination tree), the min-plus LU factorization, and the two triangular solves that
// give one row of A*. Everything shown on the page is computed here, for the graph shown.

export interface Graph {
  n: number
  x: Float64Array; y: Float64Array        // positions in [0, 1]
  edges: [number, number, number][]       // (u, v, length), u < v
  nbr: number[][]                         // neighbors
}

function rng(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// A jittered k-by-k street grid: a few streets missing, a few diagonal avenues (one per block, so it stays planar).
export function makeGraph(k = 12, seed = 7): Graph {
  const r = rng(seed), n = k * k, x = new Float64Array(n), y = new Float64Array(n)
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
    const v = i * k + j
    x[v] = (j + 0.5 + (r() - 0.5) * 0.55) / k
    y[v] = (i + 0.5 + (r() - 0.5) * 0.55) / k
  }
  const cand: [number, number][] = []
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
    const v = i * k + j
    if (j + 1 < k) cand.push([v, v + 1])
    if (i + 1 < k) cand.push([v, v + k])
  }
  // drop some streets, but keep the graph connected
  const adj: Set<number>[] = Array.from({ length: n }, () => new Set())
  for (const [u, v] of cand) { adj[u].add(v); adj[v].add(u) }
  const connected = () => {
    const seen = new Uint8Array(n), st = [0]; seen[0] = 1; let c = 1
    while (st.length) { const u = st.pop()!; for (const w of adj[u]) if (!seen[w]) { seen[w] = 1; c++; st.push(w) } }
    return c === n
  }
  for (const [u, v] of cand) {
    if (r() < 0.13) { adj[u].delete(v); adj[v].delete(u); if (!connected()) { adj[u].add(v); adj[v].add(u) } }
  }
  for (let i = 0; i + 1 < k; i++) for (let j = 0; j + 1 < k; j++) {
    if (r() < 0.12) {
      const v = i * k + j
      if (r() < 0.5) { adj[v].add(v + k + 1); adj[v + k + 1].add(v) } else { adj[v + 1].add(v + k); adj[v + k].add(v + 1) }
    }
  }
  const edges: [number, number, number][] = []
  for (let u = 0; u < n; u++) for (const v of adj[u]) if (u < v) {
    edges.push([u, v, Math.round(Math.hypot(x[u] - x[v], y[u] - y[v]) * k * 100) / 100])
  }
  const nbr = Array.from({ length: n }, (_, u) => [...adj[u]].sort((a, b) => a - b))
  return { n, x, y, edges, nbr }
}

// ---- orders. An order is a permutation: order[p] = the vertex in position p; pos[v] = its position. ----
export interface Order { name: string; order: Int32Array; pos: Int32Array }
function mk(name: string, order: ArrayLike<number>): Order {
  const o = Int32Array.from(order), p = new Int32Array(o.length)
  o.forEach((v, i) => { p[v] = i })
  return { name, order: o, pos: p }
}
// the order a data file might give: arbitrary
export function inputOrder(g: Graph, seed = 3): Order {
  const r = rng(seed), o = Array.from({ length: g.n }, (_, i) => i)
  for (let i = o.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [o[i], o[j]] = [o[j], o[i]] }
  return mk('input', o)
}
// row by row, top to bottom: a banded matrix
export function sweepOrder(g: Graph): Order {
  const k = Math.round(Math.sqrt(g.n))
  const o = Array.from({ length: g.n }, (_, i) => i).sort((a, b) => Math.floor(g.y[a] * k) - Math.floor(g.y[b] * k) || g.x[a] - g.x[b])
  return mk('sweep', o)
}

// Nested dissection by coordinates. Each tree node holds the vertices it numbers: a leaf numbers a small part; an
// inner node numbers its separator, after both halves.
export interface NDNode { verts: number[]; kids: NDNode[]; depth: number; first: number; size: number; sep: boolean; id: number }
export function nestedDissection(g: Graph, leaf = 10): { ord: Order; root: NDNode; nodes: NDNode[]; nodeOf: Int32Array } {
  const order: number[] = [], nodes: NDNode[] = []
  const nodeOf = new Int32Array(g.n)
  const inSet = new Int32Array(g.n).fill(-1), inS = new Int32Array(g.n).fill(-1)
  let stamp = 0
  const rec = (S: number[], depth: number): NDNode => {
    if (S.length <= leaf) {
      const node: NDNode = { verts: S, kids: [], depth, first: order.length, size: S.length, sep: false, id: nodes.length }
      nodes.push(node); for (const v of S) { nodeOf[v] = node.id; order.push(v) }
      return node
    }
    // split along the longer side, near the median, where the separator is smallest
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
    for (const v of S) { x0 = Math.min(x0, g.x[v]); x1 = Math.max(x1, g.x[v]); y0 = Math.min(y0, g.y[v]); y1 = Math.max(y1, g.y[v]) }
    const sS = ++stamp; for (const v of S) inS[v] = sS
    const c = x1 - x0 >= y1 - y0 ? g.x : g.y
    const sorted = [...S].sort((a, b) => c[a] - c[b])
    let best: { A: number[]; B: number[]; sep: number[] } | null = null, bestScore = Infinity
    const lo = Math.floor(sorted.length * 0.38), hi = Math.ceil(sorted.length * 0.62)
    for (let cut = lo; cut <= hi; cut++) {
      const s = ++stamp
      for (let i = 0; i < cut; i++) inSet[sorted[i]] = s            // side A
      const bA: number[] = [], bB: number[] = []
      for (let i = 0; i < sorted.length; i++) {
        const v = sorted[i], mine = i < cut
        let touches = false
        for (const w of g.nbr[v]) {
          if (inS[w] !== sS) continue
          if ((inSet[w] === s) !== mine) { touches = true; break }
        }
        if (touches) (mine ? bA : bB).push(v)
      }
      const sep = bA.length <= bB.length ? bA : bB
      const score = sep.length + Math.abs(cut - sorted.length / 2) * 0.08
      if (score < bestScore) {
        bestScore = score
        const sepSet = new Set(sep)
        best = { A: sorted.slice(0, cut).filter(v => !sepSet.has(v)), B: sorted.slice(cut).filter(v => !sepSet.has(v)), sep }
      }
    }
    const kids = [rec(best!.A, depth + 1), rec(best!.B, depth + 1)]
    const node: NDNode = { verts: best!.sep, kids, depth, first: order.length, size: best!.sep.length, sep: true, id: nodes.length }
    nodes.push(node); for (const v of best!.sep) { nodeOf[v] = node.id; order.push(v) }
    return node
  }
  const root = rec(Array.from({ length: g.n }, (_, i) => i), 0)
  return { ord: mk('nd', order), root, nodes, nodeOf }
}

// ---- symbolic elimination: eliminating a vertex joins its remaining neighbors into a clique ----
export interface Symbolic {
  ord: Order
  hi: number[][]                 // hi[v]: v's neighbors in the filled graph that come after v (column v of L)
  fill: [number, number, number][] // (u, w, step): fill edge u-w created when the vertex at position step is eliminated
  fillAt: Int32Array             // fill edges created up to (not including) step p: fill.slice(0, fillAt[p])
  parent: Int32Array             // elimination tree
  height: number
  work: number                   // sum over pivots of (remaining degree)^2: the update operations
  nnzA: number; nnzLU: number    // off-diagonal nonzeros of A, and of L + U
}
export function symbolic(g: Graph, ord: Order): Symbolic {
  const n = g.n, adj = g.nbr.map(a => new Set(a)), done = new Uint8Array(n)
  const hi: number[][] = new Array(n), fill: [number, number, number][] = [], fillAt = new Int32Array(n + 1)
  const parent = new Int32Array(n).fill(-1)
  let work = 0
  for (let p = 0; p < n; p++) {
    fillAt[p] = fill.length
    const v = ord.order[p]
    const h = [...adj[v]].filter(w => !done[w]).sort((a, b) => ord.pos[a] - ord.pos[b])
    hi[v] = h; work += h.length * h.length
    if (h.length) parent[v] = h[0]
    for (let i = 0; i < h.length; i++) for (let j = i + 1; j < h.length; j++) {
      const a = h[i], b = h[j]
      if (!adj[a].has(b)) { adj[a].add(b); adj[b].add(a); fill.push([a, b, p]) }
    }
    done[v] = 1
  }
  fillAt[n] = fill.length
  const depth = new Int32Array(n); let height = 0
  for (let p = n - 1; p >= 0; p--) { const v = ord.order[p]; depth[v] = parent[v] < 0 ? 1 : depth[parent[v]] + 1; height = Math.max(height, depth[v]) }
  const nnzA = 2 * g.edges.length
  return { ord, hi, fill, fillAt, parent, height, work, nnzA, nnzLU: nnzA + 2 * fill.length }
}

// ---- numeric: min-plus LU. After it, M[u][v] is the length of a shortest u-v path through vertices eliminated
// before both u and v: the entries of L and U. ----
export function factorize(g: Graph, s: Symbolic): Float64Array {
  const n = g.n, M = new Float64Array(n * n).fill(Infinity)
  for (let i = 0; i < n; i++) M[i * n + i] = 0
  for (const [u, v, w] of g.edges) { M[u * n + v] = w; M[v * n + u] = w }
  for (let p = 0; p < n; p++) {
    const k = s.ord.order[p], h = s.hi[k]
    for (const i of h) { const a = M[i * n + k]; for (const j of h) if (i !== j) { const b = a + M[k * n + j]; if (b < M[i * n + j]) M[i * n + j] = b } }
  }
  return M
}

// One row of A*: the distances from vertex src. A forward solve with U (it only reaches src's ancestors in the
// elimination tree), then a backward solve with L.
export function solveRow(g: Graph, s: Symbolic, M: Float64Array, src: number, out?: Float64Array): { d: Float64Array; up: number[] } {
  const n = g.n, d = out || new Float64Array(n), up: number[] = []
  d.fill(Infinity); d[src] = 0
  for (let p = s.ord.pos[src]; p < n; p++) {
    const v = s.ord.order[p]; if (d[v] === Infinity) continue
    up.push(v)
    for (const u of s.hi[v]) { const b = d[v] + M[v * n + u]; if (b < d[u]) d[u] = b }
  }
  for (let p = n - 1; p >= 0; p--) {
    const t = s.ord.order[p]
    for (const w of s.hi[t]) { const b = d[w] + M[w * n + t]; if (b < d[t]) d[t] = b }
  }
  return { d, up }
}

// ---- everything the explainer shows ----
export type Key = 'input' | 'sweep' | 'nd'
export interface Model {
  g: Graph; k: number
  ords: Record<Key, Order>; sym: Record<Key, Symbolic>
  nd: ReturnType<typeof nestedDissection>
  M: Float64Array                // the min-plus factors in nested-dissection order
  D: Float64Array                // A*, row by row from the factors: D[s * n + t] = distance from s to t
  dmax: number
}
export function build(k = 12, seed = 11): Model {
  const g = makeGraph(k, seed), nd = nestedDissection(g)
  const ords = { input: inputOrder(g), sweep: sweepOrder(g), nd: nd.ord }
  const sym = { input: symbolic(g, ords.input), sweep: symbolic(g, ords.sweep), nd: symbolic(g, ords.nd) }
  const M = factorize(g, sym.nd), n = g.n, D = new Float64Array(n * n), row = new Float64Array(n)
  let dmax = 0
  for (let s = 0; s < n; s++) {
    solveRow(g, sym.nd, M, s, row); D.set(row, s * n)
    for (let t = 0; t < n; t++) if (row[t] < Infinity && row[t] > dmax) dmax = row[t]
  }
  return { g, k, ords, sym, nd, M, D, dmax }
}
