// Drawing for the How it works explainer: the graph (left) and its matrix (right), on two canvases. All state lives in
// `ui` (owned by the Vue component); this file only reads it, animates between states, and draws.
import type { Model, Key } from './model'
import { solveRow } from './model'
import { ramp, isDark } from '../lib/palette'

export interface UI {
  step: number; ord: Key; elim: number; workers: number; src: number
  hoverV: number; hoverR: number; hoverC: number   // hovered vertex; hovered matrix row and column (vertices)
}
type RGB = [number, number, number]
export interface QueryInfo { up: number; touchU: number; touchL: number }

const ease = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t)

export function createScene(gc: HTMLCanvasElement, mc: HTMLCanvasElement, md: Model, ui: UI, reduce: boolean) {
  const { g, nd, D, dmax } = md, n = g.n
  const gx = gc.getContext('2d')!, mx = mc.getContext('2d')!
  // colors, from the site's CSS variables
  let C: Record<string, RGB> = {}, R = ramp('light'), dark = false
  const hex = (s: string, fb: RGB): RGB => {
    s = s.trim()
    if (s.startsWith('rgb')) { const m = s.match(/[\d.]+/g); return m && m.length >= 3 ? [+m[0], +m[1], +m[2]] : fb }
    if (s[0] !== '#') return fb
    if (s.length === 4) s = '#' + s[1] + s[1] + s[2] + s[2] + s[3] + s[3]
    return [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)]
  }
  const readColors = () => {
    dark = isDark(); R = ramp(dark ? 'dark' : 'light')
    const cs = getComputedStyle(gc)
    const v = (name: string, fb: RGB) => hex(cs.getPropertyValue(name), fb)
    C = { blue: v('--pp-blue', [47, 84, 200]), red: v('--pp-red', [217, 69, 43]), ink: v('--pp-ink', [26, 31, 46]),
      muted: v('--pp-muted', [93, 101, 120]), line: v('--pp-line', [226, 230, 240]), bg: v('--pp-hiw-bg', [255, 255, 255]),
      w1: v('--pp-hiw-w1', [47, 84, 200]), w2: v('--pp-hiw-w2', [217, 69, 43]), w3: v('--pp-hiw-w3', [19, 145, 127]), w4: v('--pp-hiw-w4', [201, 138, 16]),
      s0: v('--pp-sep0', [163, 34, 26]), s1: v('--pp-sep1', [207, 59, 36]), s2: v('--pp-sep2', [232, 103, 74]), s3: v('--pp-sep3', [242, 154, 124]) }
  }
  const rgba = (c: RGB, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`
  const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
  const rampC = (d: number): RGB => { const i = Math.max(0, Math.min(255, Math.round((d / dmax) * 255))) * 4; return [R[i], R[i + 1], R[i + 2]] }
  const workerC = (w: number): RGB => [C.w1, C.w2, C.w3, C.w4][w]
  // a separator's tone by its level in the dissection (the top separator darkest)
  const sepDepths = [...new Set(nd.nodes.filter(x => x.sep).map(x => x.depth))].sort((a, b) => a - b)
  const sepC = (depth: number): RGB => [C.s0, C.s1, C.s2, C.s3][Math.min(3, Math.max(0, sepDepths.indexOf(depth)))]
  readColors()

  // ---- sizes ----
  let dpr = 1
  const fit = (c: HTMLCanvasElement) => {
    const r = c.getBoundingClientRect(); dpr = Math.min(2.5, window.devicePixelRatio || 1)
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr))
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h }
  }
  const G = () => { const s = gc.width, u = s / 420, pad = 18 * u; return { s, u, pad, span: s - 2 * pad } }
  const X = (v: number, L = G()) => L.pad + g.x[v] * L.span
  const Y = (v: number, L = G()) => L.pad + g.y[v] * L.span
  const Mg = () => { const s = mc.width, u = s / 420, ox = 14 * u, S = s - 28 * u; return { s, u, ox, oy: 4 * u, S, cs: S / n } }

  // ---- the order on screen, animated between orders ----
  const orderOf = (): Key => (ui.step === 0 ? 'input' : ui.step === 2 ? ui.ord : 'nd')
  let shown: Key = orderOf()
  const posFrom = Float32Array.from(md.ords[shown].pos), pos = Float32Array.from(posFrom)
  let posT0 = -1e9, posDur = 1
  const setOrder = (key: Key) => {
    if (key === shown) return
    posFrom.set(pos); shown = key; posT0 = performance.now(); posDur = reduce ? 1 : 1300
  }
  const updatePos = (now: number) => {
    const to = md.ords[shown].pos, t = (now - posT0) / posDur
    if (t >= 1.4) { for (let v = 0; v < n; v++) pos[v] = to[v]; return false }
    for (let v = 0; v < n; v++) {
      const delay = 0.4 * (to[v] / n), e = ease(clamp01((t - delay) / 0.6))
      pos[v] = posFrom[v] + (to[v] - posFrom[v]) * e
    }
    return true
  }

  // ---- step animations ----
  let stepT0 = performance.now(), elimT = performance.now()
  const restart = () => { stepT0 = performance.now(); dirty = true }
  // query state
  let qd = new Float64Array(n), qup: number[] = [], qT0 = 0
  const query = (src: number): QueryInfo => {
    const r = solveRow(g, md.sym.nd, md.M, src, qd); qup = r.up; qT0 = performance.now(); dirty = true
    let touchU = 0; for (const v of qup) touchU += md.sym.nd.hi[v].length
    return { up: qup.length, touchU, touchL: (md.sym.nd.nnzLU) / 2 }
  }
  const T1 = () => (reduce ? 0 : Math.min(1100, 260 + 34 * qup.length)), T2 = reduce ? 0 : 1700
  const qPhase = (now: number) => {
    const t = now - qT0, t1 = T1()
    return { climbed: t1 === 0 ? qup.length : Math.min(qup.length, Math.floor((t / t1) * qup.length + 1e-9) + (t >= t1 ? 1 : 0)),
      down: T2 === 0 ? n : t <= t1 ? 0 : Math.min(n, Math.floor(((t - t1) / T2) * n)), busy: t < t1 + T2 + 100 }
  }
  const finalAt = (v: number, down: number) => n - 1 - md.ords.nd.pos[v] < down   // the backward solve finalizes from the last position down
  const SOLVE = reduce ? 1 : 3400
  const solveRows = (now: number) => clamp01((now - stepT0) / SOLVE)

  // ---- the graph ----
  const ndParent = new Int32Array(nd.nodes.length).fill(-1)
  for (const x of nd.nodes) for (const c of x.kids) ndParent[c.id] = x.id
  const leafShade = new Uint8Array(nd.nodes.length); { let i = 0; for (const x of nd.nodes) if (!x.sep) leafShade[x.id] = i++ % 2 }
  // convex hull of each part (monotone chain)
  const hulls = nd.nodes.map(x => {
    const P = [...x.verts].sort((a, b) => g.x[a] - g.x[b] || g.y[a] - g.y[b])
    if (P.length < 3) return P
    const cr = (o: number, a: number, b: number) => (g.x[a] - g.x[o]) * (g.y[b] - g.y[o]) - (g.y[a] - g.y[o]) * (g.x[b] - g.x[o])
    const lo: number[] = [], hi: number[] = []
    for (const p of P) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p) }
    for (let i = P.length - 1; i >= 0; i--) { const p = P[i]; while (hi.length >= 2 && cr(hi[hi.length - 2], hi[hi.length - 1], p) <= 0) hi.pop(); hi.push(p) }
    return lo.slice(0, -1).concat(hi.slice(0, -1))
  })
  const filledAdj = (key: Key, upto: number) => {
    const S = md.sym[key], A = g.nbr.map(a => new Set(a))
    for (let f = 0; f < S.fillAt[Math.min(upto, n)]; f++) { const [a, b] = S.fill[f]; A[a].add(b); A[b].add(a) }
    return A
  }
  const line = (c: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) => { c.moveTo(x0, y0); c.lineTo(x1, y1) }
  const dot = (c: CanvasRenderingContext2D, x: number, y: number, r: number) => { c.moveTo(x + r, y); c.arc(x, y, r, 0, 2 * Math.PI) }

  const drawGraph = (now: number) => {
    const L = G(), c = gx, u = L.u, rr = 3.5 * u, step = ui.step
    c.clearRect(0, 0, gc.width, gc.height); c.lineCap = 'round'
    const edges = (alpha: (a: number, b: number) => number, col: RGB, w: number) => {
      // batch edges by alpha bucket
      const buckets = new Map<number, number[]>()
      for (const [a, b] of g.edges) { const al = Math.round(alpha(a, b) * 20) / 20; if (al <= 0) continue; let B = buckets.get(al); if (!B) buckets.set(al, (B = [])); B.push(a, b) }
      c.lineWidth = w
      for (const [al, B] of buckets) { c.strokeStyle = rgba(col, al); c.beginPath(); for (let i = 0; i < B.length; i += 2) line(c, X(B[i], L), Y(B[i], L), X(B[i + 1], L), Y(B[i + 1], L)); c.stroke() }
    }
    const hv = ui.hoverV, hr = ui.hoverR, hc = ui.hoverC
    if (step === 0 || step === 1) {
      const rv = step === 1 ? (now - stepT0) / (reduce ? 1 : 420) : 99         // reveal by depth
      const shownAt = (depth: number) => clamp01(rv - depth)
      if (step === 1) {
        // the parts, as districts: each part's convex hull, rounded
        const cell = L.span / md.k
        c.lineJoin = 'round'; c.lineWidth = 0.62 * cell
        for (const x of nd.nodes) {
          if (x.sep) continue
          const p = ndParent[x.id], a = p < 0 ? 1 : shownAt(nd.nodes[p].depth + 1)
          if (a <= 0) continue
          const col = rgba(mix(C.bg, C.blue, (leafShade[x.id] ? 0.16 : 0.09) * a))
          c.fillStyle = col; c.strokeStyle = col; c.beginPath()
          const h = hulls[x.id]
          h.forEach((v, i) => (i ? c.lineTo(X(v, L), Y(v, L)) : c.moveTo(X(v, L), Y(v, L))))
          c.closePath(); c.fill(); c.stroke()
        }
      }
      edges(() => (dark ? 0.42 : 0.34), C.ink, 1.5 * u)
      if (step === 1) {
        // separators: the edges inside each, thick
        c.lineWidth = 7 * u
        for (const x of nd.nodes) {
          if (!x.sep) continue
          const a = shownAt(x.depth); if (a <= 0) continue
          const S = new Set(x.verts); c.strokeStyle = rgba(sepC(x.depth), 0.36 * a); c.beginPath()
          for (const v of x.verts) for (const w of g.nbr[v]) if (v < w && S.has(w)) line(c, X(v, L), Y(v, L), X(w, L), Y(w, L))
          c.stroke()
        }
      }
      // vertices
      c.fillStyle = rgba(C.blue, 1); c.beginPath()
      for (let v = 0; v < n; v++) { const x = nd.nodes[nd.nodeOf[v]]; if (step === 1 && x.sep && shownAt(x.depth) > 0) continue; dot(c, X(v, L), Y(v, L), rr) }
      c.fill()
      if (step === 1) for (const x of nd.nodes) {
        if (!x.sep) continue
        const a = shownAt(x.depth); if (a <= 0) continue
        c.fillStyle = rgba(mix(C.blue, sepC(x.depth), a), 1); c.beginPath(); for (const v of x.verts) dot(c, X(v, L), Y(v, L), rr * (1 + 0.35 * a)); c.fill()
      }
      hoverMarks(L, g.nbr.map(a => a))
    } else if (step === 2) {
      const S = md.sym[ui.ord], p = ui.elim, P = md.ords[ui.ord].pos
      const gone = (v: number) => P[v] < p
      edges((a, b) => (gone(a) || gone(b) ? 0.1 : dark ? 0.42 : 0.34), C.ink, 1.5 * u)
      // fill edges made so far
      const fa = S.fillAt[p]
      c.lineWidth = 1.2 * u
      c.strokeStyle = rgba(C.red, dark ? 0.16 : 0.12); c.beginPath()
      for (let f = 0; f < fa; f++) { const [a, b] = S.fill[f]; if (gone(a) || gone(b)) line(c, X(a, L), Y(a, L), X(b, L), Y(b, L)) }
      c.stroke()
      c.strokeStyle = rgba(C.red, ui.ord === 'input' ? 0.42 : 0.6); c.beginPath()
      for (let f = 0; f < fa; f++) { const [a, b, s] = S.fill[f]; if (!gone(a) && !gone(b) && s !== p - 1) line(c, X(a, L), Y(a, L), X(b, L), Y(b, L)) }
      c.stroke()
      // the last pivot and the clique it made
      if (p > 0) {
        const v = md.ords[ui.ord].order[p - 1], h = S.hi[v], pop = clamp01((now - elimT) / 260)
        c.lineWidth = 2.4 * u; c.strokeStyle = rgba(C.red, 0.95); c.beginPath()
        for (let f = S.fillAt[p - 1]; f < S.fillAt[p]; f++) { const [a, b] = S.fill[f]; line(c, X(a, L), Y(a, L), X(b, L), Y(b, L)) }
        c.stroke()
        c.lineWidth = 1.6 * u; c.setLineDash([3 * u, 3 * u]); c.strokeStyle = rgba(C.ink, 0.7); c.beginPath()
        for (const w of h) line(c, X(v, L), Y(v, L), X(w, L), Y(w, L))
        c.stroke(); c.setLineDash([])
        c.strokeStyle = rgba(C.red, 1); c.lineWidth = 2 * u; c.beginPath(); dot(c, X(v, L), Y(v, L), rr * (1.6 + 0.8 * (1 - pop))); c.stroke()
      }
      // vertices: the remaining ones solid, the eliminated ones hollow
      c.fillStyle = rgba(C.blue, 1); c.beginPath(); for (let v = 0; v < n; v++) if (!gone(v)) dot(c, X(v, L), Y(v, L), rr); c.fill()
      c.strokeStyle = rgba(C.muted, 0.7); c.lineWidth = 1.1 * u; c.beginPath(); for (let v = 0; v < n; v++) if (gone(v)) dot(c, X(v, L), Y(v, L), rr * 0.7); c.stroke()
      if (hv >= 0 || hr >= 0) hoverMarks(L, filledAdj(ui.ord, p).map(s => [...s]))
    } else if (step === 3) {
      const W = ui.workers, per = n / W, done = solveRows(now) * per, P = md.ords.nd.pos
      edges(() => (dark ? 0.3 : 0.24), C.ink, 1.4 * u)
      for (let w = 0; w < W; w++) {
        const col = workerC(w)
        c.fillStyle = rgba(col, 1); c.strokeStyle = rgba(col, 0.9); c.lineWidth = 1.3 * u
        const solid = new Path2D(), ring = new Path2D()
        for (let v = 0; v < n; v++) {
          if (Math.floor(P[v] / per) !== w) continue
          const k = P[v] - w * per
          if (k < done) { solid.moveTo(X(v, L) + rr, Y(v, L)); solid.arc(X(v, L), Y(v, L), rr, 0, 2 * Math.PI) }
          else { ring.moveTo(X(v, L) + rr * 0.8, Y(v, L)); ring.arc(X(v, L), Y(v, L), rr * 0.8, 0, 2 * Math.PI) }
        }
        c.fill(solid); c.stroke(ring)
        // the row this worker is on
        const k = Math.floor(done)
        if (k < per && solveRows(now) < 1) {
          const v = md.ords.nd.order[Math.floor(w * per) + k], ph = (now / 500) % 1
          c.strokeStyle = rgba(col, 0.9 * (1 - ph)); c.lineWidth = 2 * u; c.beginPath(); dot(c, X(v, L), Y(v, L), rr * (1.4 + 2.2 * ph)); c.stroke()
          c.fillStyle = rgba(col, 1); c.beginPath(); dot(c, X(v, L), Y(v, L), rr * 1.3); c.fill()
        }
      }
      if (hr >= 0 || hv >= 0) { const v = hr >= 0 ? hr : hv; c.strokeStyle = rgba(C.ink, 0.95); c.lineWidth = 2 * u; c.beginPath(); dot(c, X(v, L), Y(v, L), rr * 2); c.stroke() }
    } else {
      const ph = qPhase(now), src = ui.src
      edges(() => (dark ? 0.16 : 0.13), C.ink, 1.4 * u)
      // edges whose ends are both final, colored by distance
      c.lineWidth = 2.6 * u
      for (const [a, b] of g.edges) {
        if (!finalAt(a, ph.down) || !finalAt(b, ph.down)) continue
        const ca = rampC(qd[a]), cb = rampC(qd[b]), xa = X(a, L), ya = Y(a, L), xb = X(b, L), yb = Y(b, L)
        const gr = c.createLinearGradient(xa, ya, xb, yb); gr.addColorStop(0, rgba(ca)); gr.addColorStop(1, rgba(cb))
        c.strokeStyle = gr; c.beginPath(); line(c, xa, ya, xb, yb); c.stroke()
      }
      // the climb: the source's ancestors in the elimination tree
      const cl = ph.climbed
      if (cl > 1) {
        c.setLineDash([4 * u, 3.5 * u]); c.lineWidth = 1.8 * u; c.strokeStyle = rgba(C.ink, ph.down >= n ? 0.45 : 0.85); c.beginPath()
        for (let i = 0; i + 1 < cl; i++) line(c, X(qup[i], L), Y(qup[i], L), X(qup[i + 1], L), Y(qup[i + 1], L))
        c.stroke(); c.setLineDash([])
      }
      for (let v = 0; v < n; v++) {
        const f = finalAt(v, ph.down)
        c.fillStyle = f ? rgba(rampC(qd[v])) : rgba(C.muted, 0.5)
        c.beginPath(); dot(c, X(v, L), Y(v, L), f ? rr : rr * 0.7); c.fill()
      }
      c.strokeStyle = rgba(C.ink, 0.85); c.lineWidth = 1.5 * u; c.beginPath()
      for (let i = 1; i < cl; i++) dot(c, X(qup[i], L), Y(qup[i], L), rr * 1.75)
      c.stroke()
      // the source
      c.fillStyle = rgba(C.bg); c.strokeStyle = rgba(C.red); c.lineWidth = 3.2 * u
      c.beginPath(); dot(c, X(src, L), Y(src, L), rr * 1.9); c.fill(); c.stroke()
      if (hv >= 0 && hv !== src) { c.strokeStyle = rgba(C.ink); c.lineWidth = 1.8 * u; c.beginPath(); dot(c, X(hv, L), Y(hv, L), rr * 2); c.stroke() }
    }
  }
  // hovered vertex: its edges and neighbors; hovered entry: its row and column vertices
  const hoverMarks = (L: ReturnType<typeof G>, adj: number[][]) => {
    const c = gx, u = L.u, rr = 3.5 * u, hv = ui.hoverV >= 0 ? ui.hoverV : -1
    if (hv >= 0) {
      c.strokeStyle = rgba(C.red, 0.95); c.lineWidth = 2.4 * u; c.beginPath()
      for (const w of adj[hv]) line(c, X(hv, L), Y(hv, L), X(w, L), Y(w, L))
      c.stroke()
      c.fillStyle = rgba(C.red); c.beginPath(); for (const w of adj[hv]) dot(c, X(w, L), Y(w, L), rr * 1.15); c.fill()
      c.fillStyle = rgba(C.bg); c.strokeStyle = rgba(C.red); c.lineWidth = 3 * u; c.beginPath(); dot(c, X(hv, L), Y(hv, L), rr * 1.9); c.fill(); c.stroke()
    }
    if (ui.hoverR >= 0) {
      const a = ui.hoverR, b = ui.hoverC, linked = a !== b && adj[a].includes(b)
      if (a !== b) {
        c.strokeStyle = rgba(linked ? C.red : C.muted, 0.95); c.lineWidth = (linked ? 3 : 1.6) * u
        if (!linked) c.setLineDash([4 * u, 4 * u])
        c.beginPath(); line(c, X(a, L), Y(a, L), X(b, L), Y(b, L)); c.stroke(); c.setLineDash([])
      }
      c.fillStyle = rgba(C.bg); c.strokeStyle = rgba(C.red); c.lineWidth = 3 * u; c.beginPath(); dot(c, X(a, L), Y(a, L), rr * 1.9); c.fill(); c.stroke()
      if (a !== b) { c.strokeStyle = rgba(C.ink); c.beginPath(); dot(c, X(b, L), Y(b, L), rr * 1.9); c.fill(); c.stroke() }
    }
  }

  // ---- the matrix ----
  const star = document.createElement('canvas'); star.width = n; star.height = n
  const sctx = star.getContext('2d')!, img = sctx.createImageData(n, n)
  const painted = new Uint8Array(n); let starKey = ''
  const drawMatrix = (now: number) => {
    const L = Mg(), c = mx, { u, ox, oy, S, cs } = L, step = ui.step
    c.clearRect(0, 0, mc.width, mc.height)
    c.fillStyle = rgba(C.bg); c.fillRect(ox, oy, S, S)
    const sz = Math.max(cs * 0.94, 2.3 * dpr), ins = (cs - sz) / 2
    const cell = (r: number, col: number, grow = 1) => { const e = (sz * (grow - 1)) / 2; c.rect(ox + col * cs + ins - e, oy + r * cs + ins - e, sz * grow, sz * grow) }
    const band = (r: number, col: RGB, a: number, vertical: boolean) => { c.fillStyle = rgba(col, a); vertical ? c.fillRect(ox + r * cs, oy, cs, S) : c.fillRect(ox, oy + r * cs, S, cs) }
    const nzA = (col: RGB, a: number) => { c.fillStyle = rgba(col, a); c.beginPath(); for (const [p, q] of g.edges) { cell(pos[p], pos[q]); cell(pos[q], pos[p]) } c.fill() }
    const diag = (a: number) => { c.strokeStyle = rgba(C.muted, a); c.lineWidth = Math.max(1, 0.8 * u); c.setLineDash([2 * u, 3 * u]); c.beginPath(); line(c, ox, oy, ox + S, oy + S); c.stroke(); c.setLineDash([]) }
    const hoverBands = () => {
      if (ui.hoverV >= 0) { band(pos[ui.hoverV], C.red, 0.1, false); band(pos[ui.hoverV], C.red, 0.1, true) }
      if (ui.hoverR >= 0) { band(pos[ui.hoverR], C.blue, 0.1, false); band(pos[ui.hoverC], C.blue, 0.1, true) }
    }
    const blocks = (alpha: number) => {
      if (alpha <= 0) return
      const lo = (x: typeof nd.nodes[0]): number => (x.kids.length ? lo(x.kids[0]) : x.first)
      c.lineWidth = Math.max(1, 0.9 * u)
      for (const x of nd.nodes) {
        if (!x.sep) { c.strokeStyle = rgba(C.blue, 0.5 * alpha); c.strokeRect(ox + x.first * cs, oy + x.first * cs, x.size * cs, x.size * cs); continue }
        const a = lo(x), b = x.first + x.size
        const sc = sepC(x.depth)
        c.fillStyle = rgba(sc, 0.08 * alpha)
        c.fillRect(ox + a * cs, oy + x.first * cs, (b - a) * cs, x.size * cs); c.fillRect(ox + x.first * cs, oy + a * cs, x.size * cs, (x.first - a) * cs)
        c.strokeStyle = rgba(sc, 0.7 * alpha); c.strokeRect(ox + x.first * cs, oy + x.first * cs, x.size * cs, x.size * cs)
        c.strokeStyle = rgba(C.muted, 0.35 * alpha); c.strokeRect(ox + a * cs, oy + a * cs, (b - a) * cs, (b - a) * cs)
      }
    }
    if (step <= 1) {
      diag(0.5)
      hoverBands()
      if (step === 1) blocks(clamp01((now - posT0 - posDur * 1.1) / 500))
      nzA(C.ink, dark ? 0.9 : 0.85)
      if (ui.hoverV >= 0) { const v = ui.hoverV; c.fillStyle = rgba(C.red); c.beginPath(); for (const w of g.nbr[v]) { cell(pos[v], pos[w], 1.25); cell(pos[w], pos[v], 1.25) } c.fill() }
      if (ui.hoverR >= 0) { c.strokeStyle = rgba(C.red); c.lineWidth = 1.6 * u; c.strokeRect(ox + pos[ui.hoverC] * cs - u, oy + pos[ui.hoverR] * cs - u, cs + 2 * u, cs + 2 * u) }
    } else if (step === 2) {
      const S2 = md.sym[ui.ord], p = ui.elim
      diag(0.4)
      if (ui.ord === 'nd' && shown === 'nd') blocks(0.55 * clamp01((now - posT0 - posDur) / 400))
      // the part still to eliminate
      if (p > 0 && p < n && shown === ui.ord) {
        c.fillStyle = rgba(C.ink, dark ? 0.05 : 0.03); c.fillRect(ox, oy, p * cs, S); c.fillRect(ox + p * cs, oy, S - p * cs, p * cs)
        c.strokeStyle = rgba(C.red, 0.6); c.lineWidth = Math.max(1, u); c.strokeRect(ox + (p - 1) * cs, oy + (p - 1) * cs, (n - p + 1) * cs, (n - p + 1) * cs)
      }
      hoverBands()
      nzA(C.ink, dark ? 0.9 : 0.85)
      const fa = S2.fillAt[p], pop = clamp01((now - elimT) / 260)
      c.fillStyle = rgba(C.red, 1); c.beginPath()
      for (let f = 0; f < fa; f++) { const [a, b, s] = S2.fill[f]; const gr = s === p - 1 ? 1 + 0.9 * (1 - pop) : 1; cell(pos[a], pos[b], gr); cell(pos[b], pos[a], gr) }
      c.fill()
      if (ui.hoverR >= 0) { c.strokeStyle = rgba(C.blue); c.lineWidth = 1.6 * u; c.strokeRect(ox + pos[ui.hoverC] * cs - u, oy + pos[ui.hoverR] * cs - u, cs + 2 * u, cs + 2 * u) }
    } else if (step === 3) {
      // A*, row by row, each row tinted by the worker that computes it
      const W = ui.workers, per = n / W, done = solveRows(now) * per, O = md.ords.nd.order, key = W + (dark ? 'd' : 'l')
      if (key !== starKey) { starKey = key; painted.fill(0); img.data.fill(0) }
      const bg = C.bg
      let changed = false
      for (let r = 0; r < n; r++) {
        const isDone = r - Math.floor(r / per) * per < done
        if (isDone === !!painted[r]) continue
        painted[r] = isDone ? 1 : 0; changed = true
        if (!isDone) { for (let q = 0; q < n; q++) img.data[(r * n + q) * 4 + 3] = 0; continue }
        const col = workerC(Math.floor(r / per)), s = O[r]
        for (let q = 0; q < n; q++) {
          const t = D[s * n + O[q]] / dmax, cc = mix(col, bg, 0.1 + 0.78 * Math.pow(t, 0.8)), i = (r * n + q) * 4
          img.data[i] = cc[0]; img.data[i + 1] = cc[1]; img.data[i + 2] = cc[2]; img.data[i + 3] = 255
        }
      }
      if (changed) sctx.putImageData(img, 0, 0)
      c.imageSmoothingEnabled = false; c.drawImage(star, ox, oy, S, S)
      // workers' bands in the margin, and the rows being computed
      for (let w = 0; w < W; w++) {
        const col = workerC(w)
        c.fillStyle = rgba(col, 0.25); c.fillRect(ox - 9 * u, oy + w * per * cs, 5 * u, per * cs - u)
        c.fillStyle = rgba(col, 1); c.fillRect(ox - 9 * u, oy + w * per * cs, 5 * u, Math.min(per, done) * cs - (done >= per ? u : 0))
        const k = Math.floor(done)
        if (k < per && solveRows(now) < 1) { c.fillStyle = rgba(col, 0.9); c.fillRect(ox, oy + (w * per + k) * cs, S, cs) }
      }
      if (ui.hoverR >= 0) {
        c.strokeStyle = rgba(C.ink, 0.9); c.lineWidth = Math.max(1, u)
        c.strokeRect(ox, oy + pos[ui.hoverR] * cs, S, cs); c.strokeRect(ox + pos[ui.hoverC] * cs - u, oy + pos[ui.hoverR] * cs - u, cs + 2 * u, cs + 2 * u)
      }
    } else {
      // the factors, and the entries the two solves read
      const S2 = md.sym.nd, ph = qPhase(now), P = md.ords.nd.pos
      diag(0.4)
      blocks(0.4)
      nzA(C.muted, dark ? 0.45 : 0.35)
      c.fillStyle = rgba(C.muted, dark ? 0.45 : 0.35); c.beginPath(); for (const [a, b] of S2.fill) { cell(P[a], P[b]); cell(P[b], P[a]) } c.fill()
      // forward solve: rows of U at the source's ancestors
      c.fillStyle = rgba(C.blue, 1); c.beginPath()
      for (let i = 0; i < ph.climbed; i++) { const v = qup[i]; cell(P[v], P[v], 1.15); for (const w of S2.hi[v]) cell(P[v], P[w], 1.15) }
      c.fill()
      // backward solve: columns of L, from the last one down
      c.fillStyle = rgba(C.red, 0.85); c.beginPath()
      for (let k = 0; k < ph.down; k++) { const t = md.ords.nd.order[n - 1 - k]; for (const w of S2.hi[t]) cell(P[w], P[t], 1.15) }
      c.fill()
      if (ph.down > 0 && ph.down < n) { const r = n - ph.down; c.fillStyle = rgba(C.red, 0.12); c.fillRect(ox + r * cs, oy, cs, S) }
      // the result: one row of A*
      const sy = oy + S + 6 * u
      c.fillStyle = rgba(C.line); c.fillRect(ox, sy, S, 10 * u)
      for (let q = 0; q < n; q++) {
        const v = md.ords.nd.order[q]
        if (!finalAt(v, ph.down)) continue
        c.fillStyle = rgba(rampC(qd[v])); c.fillRect(ox + q * cs, sy, cs + 0.5, 10 * u)
      }
      c.strokeStyle = rgba(C.red); c.lineWidth = 1.6 * u; c.strokeRect(ox + P[ui.src] * cs - u, sy - u, cs + 2 * u, 12 * u)
      if (ui.hoverV >= 0) { c.strokeStyle = rgba(C.ink); c.strokeRect(ox + P[ui.hoverV] * cs - u, sy - u, cs + 2 * u, 12 * u) }
    }
    c.strokeStyle = rgba(C.muted, 0.45); c.lineWidth = Math.max(1, 0.8 * u); c.strokeRect(ox, oy, S, S)
  }

  // ---- the loop ----
  let dirty = true, raf = 0, alive = true, visible = true
  const busy = (now: number) =>
    now - posT0 < posDur * 1.6 + 600 || (ui.step === 1 && now - stepT0 < 3000) || (ui.step === 2 && now - elimT < 300) ||
    (ui.step === 3 && solveRows(now) < 1.05) || (ui.step === 4 && qPhase(now).busy)
  let onFrame: ((now: number) => void) | null = null
  const frame = (now: number) => {
    if (!alive) return
    raf = requestAnimationFrame(frame)
    onFrame?.(now)
    if (!visible) return
    if (isDark() !== dark) { readColors(); starKey = ''; dirty = true }
    const want = orderOf(); if (want !== shown) setOrder(want)
    if (!(dirty || busy(now))) return
    dirty = false
    updatePos(now)
    drawGraph(now); drawMatrix(now)
  }
  const ro = new ResizeObserver(() => { fit(gc); fit(mc); dirty = true })
  ro.observe(gc); ro.observe(mc); fit(gc); fit(mc)
  const io = new IntersectionObserver((es) => { for (const e of es) visible = e.isIntersecting; dirty = true })
  io.observe(gc.parentElement!.parentElement!)
  raf = requestAnimationFrame(frame)

  // pointer helpers: the vertex near a point of the graph canvas, and the (row, column) vertices under a point of the matrix
  const vertexAt = (px: number, py: number, radius = 22) => {
    const L = G(), x = px * dpr, y = py * dpr; let best = -1, bd = (radius * dpr) ** 2
    for (let v = 0; v < n; v++) { const d = (X(v, L) - x) ** 2 + (Y(v, L) - y) ** 2; if (d < bd) { bd = d; best = v } }
    return best
  }
  const cellAt = (px: number, py: number): [number, number] => {
    const L = Mg(), r = Math.floor((py * dpr - L.oy) / L.cs), q = Math.floor((px * dpr - L.ox) / L.cs)
    if (r < 0 || q < 0 || r >= n || q >= n) return [-1, -1]
    const O = md.ords[shown].order; return [O[r], O[q]]
  }
  return {
    vertexAt, cellAt, query, restart,
    elimChanged: () => { elimT = performance.now(); dirty = true },
    touch: () => { dirty = true },
    shownKey: () => shown,
    set onFrame(f: ((now: number) => void) | null) { onFrame = f },
    destroy: () => { alive = false; cancelAnimationFrame(raf); ro.disconnect(); io.disconnect() },
  }
}
