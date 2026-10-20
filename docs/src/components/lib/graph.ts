// A graph in compressed adjacency form, and Dijkstra's algorithm over two semirings:
// 'sum' (min-plus: shortest paths) and 'max' (min-max: bottleneck paths, the mildest worst step).
export interface CSR { n: number; ptr: Int32Array; adj: Int32Array; w: Float32Array }
export function csr(n: number, I: ArrayLike<number>, J: ArrayLike<number>, W: ArrayLike<number>): CSR {
  const deg = new Int32Array(n + 1)
  for (let k = 0; k < I.length; k++) { deg[I[k] + 1]++; deg[J[k] + 1]++ }
  for (let i = 0; i < n; i++) deg[i + 1] += deg[i]
  const ptr = deg.slice(), fill = deg.slice(0, n), adj = new Int32Array(2 * I.length), w = new Float32Array(2 * I.length)
  for (let k = 0; k < I.length; k++) {
    const a = I[k], b = J[k]
    adj[fill[a]] = b; w[fill[a]++] = W[k]; adj[fill[b]] = a; w[fill[b]++] = W[k]
  }
  return { n, ptr, adj, w }
}
// binary heap of (key, vertex), keys in a Float64Array
export function dijkstra(g: CSR, sources: number[], mode: 'sum' | 'max' = 'sum', out?: Float64Array): Float64Array {
  const n = g.n, d = out || new Float64Array(n)
  d.fill(Infinity)
  const hk = new Float64Array(n + g.adj.length), hv = new Int32Array(n + g.adj.length); let hn = 0
  const push = (k: number, v: number) => { let i = hn++; while (i > 0) { const p = (i - 1) >> 1; if (hk[p] <= k) break; hk[i] = hk[p]; hv[i] = hv[p]; i = p } hk[i] = k; hv[i] = v }
  const pop = () => { const k = hk[0], v = hv[0], lk = hk[--hn], lv = hv[hn]; let i = 0
    while (true) { let c = 2 * i + 1; if (c >= hn) break; if (c + 1 < hn && hk[c + 1] < hk[c]) c++; if (hk[c] >= lk) break; hk[i] = hk[c]; hv[i] = hv[c]; i = c }
    hk[i] = lk; hv[i] = lv; return v }
  const zero = mode === 'sum' ? 0 : -Infinity
  for (const s of sources) { d[s] = zero; push(zero, s) }
  const done = new Uint8Array(n)
  while (hn > 0) {
    const u = pop(); if (done[u]) continue; done[u] = 1; const du = d[u]
    for (let k = g.ptr[u]; k < g.ptr[u + 1]; k++) {
      const v = g.adj[k], nd = mode === 'sum' ? du + g.w[k] : Math.max(du, g.w[k])
      if (nd < d[v]) { d[v] = nd; push(nd, v) }
    }
  }
  return d
}
// nearest vertex to (x, y), with a uniform grid built once
export function nearest(xy: Float32Array, cell = 0): (x: number, y: number) => number {
  const n = xy.length / 2; let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
  for (let i = 0; i < n; i++) { x0 = Math.min(x0, xy[2 * i]); x1 = Math.max(x1, xy[2 * i]); y0 = Math.min(y0, xy[2 * i + 1]); y1 = Math.max(y1, xy[2 * i + 1]) }
  const s = cell || Math.sqrt(((x1 - x0) * (y1 - y0)) / n) * 2, gx = Math.ceil((x1 - x0) / s) + 1, gy = Math.ceil((y1 - y0) / s) + 1
  const buckets: number[][] = Array.from({ length: gx * gy }, () => [])
  for (let i = 0; i < n; i++) buckets[Math.floor((xy[2 * i + 1] - y0) / s) * gx + Math.floor((xy[2 * i] - x0) / s)].push(i)
  return (x, y) => {
    const cx = Math.floor((x - x0) / s), cy = Math.floor((y - y0) / s); let best = -1, bd = Infinity
    for (let r = 0; r < Math.max(gx, gy); r++) {
      for (let j = cy - r; j <= cy + r; j++) for (let i = cx - r; i <= cx + r; i++) {
        if (i < 0 || j < 0 || i >= gx || j >= gy || (Math.abs(i - cx) !== r && Math.abs(j - cy) !== r)) continue
        for (const v of buckets[j * gx + i]) { const dd = (xy[2 * v] - x) ** 2 + (xy[2 * v + 1] - y) ** 2; if (dd < bd) { bd = dd; best = v } }
      }
      if (best >= 0 && r > 0) break
    }
    return best
  }
}
