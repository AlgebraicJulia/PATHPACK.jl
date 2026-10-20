<script setup lang="ts">
// Paris's streets (OpenStreetMap, 11,840 junctions). "From your cursor": per mouse move, the browser queries PATHPACK.jl's
// factorization of the street graph (docs/site_factors.jl, lib/pathpack.ts) for one row of the distance matrix, and
// every street is colored by its walking distance from the junction under the cursor (one Dijkstra in the browser if
// the factorization cannot be loaded). "Every street at once": each street's closeness (mean distance to all other
// junctions), from PATHPACK.jl's all 140 million distances.
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'

const canvas = ref<HTMLCanvasElement>(), wrap = ref<HTMLDivElement>()
const mode = ref<'live' | 'all'>('live'), ms = ref(''), pp = ref(true), street = ref(''), km = ref(''), pinned = ref(false)
const tip = ref({ x: 0, y: 0, show: false }), dot = ref({ x: 0, y: 0, show: false })
let api: { setMode: (m: 'live' | 'all') => void } | null = null
let stop = () => {}
function setMode(m: 'live' | 'all') { mode.value = m; api?.setMode(m) }

onMounted(async () => {
  await (await import('./lib/bin')).nearView(canvas.value!)
  const [{ loadBin }, { csr, dijkstra, nearest }, { FieldGL }, { isDark }, { fromBin, distancesFrom }] = await Promise.all([
    import('./lib/bin'), import('./lib/graph'), import('./lib/fieldgl'), import('./lib/palette'), import('./lib/pathpack')])
  // the streets, and PATHPACK.jl's factorization of their graph (null if it cannot be loaded: then Dijkstra)
  const [{ meta, a }, lu] = await Promise.all([loadBin(withBase('/data/paris.bin.gz')),
    loadBin(withBase('/data/paris-lu.bin.gz')).then(fromBin).catch(() => null)])
  const n = meta.n, m = meta.m, ei = a.ei, ej = a.ej, off = a.off as Uint32Array
  const dq = (q: number, c: number) => meta.lo[c] + q / meta.scale
  const xy = new Float32Array(2 * n); for (let i = 0; i < 2 * n; i++) xy[i] = dq(a.xy[i], i & 1)
  const w = new Float32Array(m); for (let e = 0; e < m; e++) w[e] = a.wm[e] / 1000
  const clo = new Float32Array(n); for (let i = 0; i < n; i++) clo[i] = meta.clo_lo + a.clo[i] / 10000
  // street geometry: the first point of each street, then steps along it
  const pts = new Float32Array(2 * off[m])
  for (let e = 0, sI = 0; e < m; e++) {
    let x = a.abs[2 * e], y = a.abs[2 * e + 1]
    for (let p = off[e]; p < off[e + 1]; p++) {
      if (p > off[e]) { x += a.step[2 * sI]; y += a.step[2 * sI + 1]; sI++ }
      pts[2 * p] = dq(x, 0); pts[2 * p + 1] = dq(y, 1)
    }
  }
  const g = csr(n, ei, ej, w)
  let dirty = true
  // polyline pieces: edge, and the distance along the street at each end of the piece
  let ns = 0; for (let e = 0; e < m; e++) ns += off[e + 1] - off[e] - 1
  const seg = new Float32Array(4 * ns), wid = new Float32Array(ns).fill(1.35)
  const se = new Int32Array(ns), c0 = new Float32Array(ns), c1 = new Float32Array(ns)
  let s = 0
  for (let e = 0; e < m; e++) {
    const p0 = off[e], p1 = off[e + 1]; let tot = 0
    for (let p = p0; p < p1 - 1; p++) tot += Math.hypot(pts[2 * p + 2] - pts[2 * p], pts[2 * p + 3] - pts[2 * p + 1])
    let c = 0
    for (let p = p0; p < p1 - 1; p++) {
      const L = Math.hypot(pts[2 * p + 2] - pts[2 * p], pts[2 * p + 3] - pts[2 * p + 1])
      seg.set([pts[2 * p], pts[2 * p + 1], pts[2 * p + 2], pts[2 * p + 3]], 4 * s)
      se[s] = e; c0[s] = tot > 0 ? (c / tot) * w[e] : 0; c += L; c1[s] = tot > 0 ? (c / tot) * w[e] : 0; s++
    }
  }
  const fg = new FieldGL(canvas.value!); fg.pad = 0.02
  fg.setGeometry(seg, wid, [{ start: 0, count: ns, width: 1.35, alpha: 1 }])
  fg.onresize = () => { dirty = true }
  const near = nearest(xy), d = new Float64Array(n), vals = new Float64Array(2 * ns)
  const cmin = Math.min(...clo), cmax = Math.max(...clo)
  let src = 8816, vmax = 10, t0 = 0, raf = 0, alive = true, want = -1
  const fill = () => {
    if (mode.value === 'live') {
      for (let k = 0; k < ns; k++) { const e = se[k], L = w[e], da = d[ei[e]], db = d[ej[e]]
        vals[2 * k] = Math.min(da + c0[k], db + L - c0[k]); vals[2 * k + 1] = Math.min(da + c1[k], db + L - c1[k]) }
      vmax = 10
    } else {
      for (let k = 0; k < ns; k++) { const e = se[k], L = w[e], ca = clo[ei[e]] - cmin, cb = clo[ej[e]] - cmin
        vals[2 * k] = ca + (cb - ca) * (c0[k] / (L || 1)); vals[2 * k + 1] = ca + (cb - ca) * (c1[k] / (L || 1)) }
      vmax = cmax - cmin
    }
    fg.setValues(vals); dirty = true
  }
  const name = (v: number) => meta.names[a.name[v]] || 'a street'
  const run = (v: number) => {
    src = v; const t = performance.now(); if (lu) distancesFrom(lu, v, d); else dijkstra(g, [v], 'sum', d); ms.value = (performance.now() - t).toFixed(1)
    street.value = name(v); const [cx, cy] = fg.toCss(xy[2 * v], xy[2 * v + 1]); dot.value = { x: cx, y: cy, show: mode.value === 'live' }
    fill()
  }
  api = { setMode: (mm) => { mode.value = mm; dot.value.show = mm === 'live'; fill(); t0 = performance.now() } }
  const frame = (now: number) => {
    if (!alive) return
    if (want >= 0 && want !== src) { run(want) }
    want = -1
    const dark = isDark(); if ((fg.theme === 'dark') !== dark) { fg.setTheme(dark ? 'dark' : 'light'); dirty = true }
    const p = Math.min(1, (now - t0) / 900), e = 1 - Math.pow(1 - p, 3)
    if (dirty || p < 1) { fg.draw({ reach: p < 1 ? e * vmax * 1.02 : 1e29, vmax, front: p < 1 ? vmax * 0.05 : 0, glow: 0.4,
      base: dark ? [0.9, 0.92, 0.96, 0.12] : [0.1, 0.12, 0.18, 0.1] }); dirty = false }
    raf = requestAnimationFrame(frame)
  }
  const at = (ev: PointerEvent) => { const r = canvas.value!.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top] }
  const onMove = (ev: PointerEvent) => {
    const [px, py] = at(ev), [x, y] = fg.toWorld(px, py), v = near(x, y)
    tip.value = { x: px, y: py, show: true }
    if (mode.value === 'all') { street.value = name(v); km.value = clo[v].toFixed(1) + ' km on average'; return }
    if (!pinned.value) want = v
    km.value = d[v] < Infinity ? d[v].toFixed(2) + ' km away' : ''
    if (pinned.value) street.value = name(v)
  }
  const onLeave = () => { tip.value.show = false }
  const onClick = (ev: PointerEvent) => { if (mode.value !== 'live') return; pinned.value = !pinned.value; if (!pinned.value) onMove(ev) }
  const el = canvas.value!
  el.addEventListener('pointermove', onMove); el.addEventListener('pointerleave', onLeave); el.addEventListener('click', onClick as any)
  pp.value = !!lu
  run(src); t0 = performance.now(); raf = requestAnimationFrame(frame)
  stop = () => { alive = false; cancelAnimationFrame(raf); el.removeEventListener('pointermove', onMove); el.removeEventListener('pointerleave', onLeave); fg.destroy() }
})
onBeforeUnmount(() => stop())
</script>

<template>
  <div class="pp-demo">
    <div class="pp-demo-bar">
      <div class="pp-seg" role="group" aria-label="What to show">
        <button type="button" :class="{ on: mode === 'live' }" :aria-pressed="mode === 'live'" @click="setMode('live')">From your cursor</button>
        <button type="button" :class="{ on: mode === 'all' }" :aria-pressed="mode === 'all'" @click="setMode('all')">Every street at once</button>
      </div>
      <span class="pp-demo-stat" v-if="mode === 'live' && pp">PATHPACK.jl's factorization, queried in your browser: <b>{{ ms }} ms</b></span>
      <span class="pp-demo-stat" v-else-if="mode === 'live'">one Dijkstra in your browser: <b>{{ ms }} ms</b></span>
      <span class="pp-demo-stat" v-else>closeness: each junction's mean distance to all 11,840</span>
    </div>
    <div class="pp-demo-stage" ref="wrap">
      <canvas ref="canvas" class="pp-demo-canvas" aria-label="Paris streets colored by walking distance"></canvas>
      <div class="pp-dot" v-show="dot.show" :style="{ left: dot.x + 'px', top: dot.y + 'px' }"></div>
      <div class="pp-tip" v-show="tip.show" :style="{ left: tip.x + 'px', top: tip.y + 'px' }">
        <b>{{ street }}</b><span>{{ km }}</span>
      </div>
      <div class="pp-legend">
        <span>{{ mode === 'live' ? '0 km' : '3.7 km' }}</span><i></i><span>{{ mode === 'live' ? '10 km' : '8.5 km' }}</span>
      </div>
    </div>
    <p class="pp-demo-note">
      Move over the map{{ mode === 'live' ? '; click to pin the source' : '' }}. Your browser computes one row of the
      distance matrix per move<template v-if="pp">, with PATHPACK.jl's triangular solves on its factorization of the
      street graph, made once ahead of time</template>. PATHPACK.jl computes every row, all 140 million distances, in 0.28 s on 16 CPU cores:
      <b>70×</b> faster than igraph and <b>3.4×</b> faster than graph-tool on the same cores.
      <span class="pp-credit">Streets © OpenStreetMap contributors.</span>
    </p>
  </div>
</template>
