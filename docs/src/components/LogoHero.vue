<script setup lang="ts">
// The home page's opening: the [PATH PACK]* logo as a street network of 35,466 points. A distance wave spreads from
// the asterisk; a click anywhere starts a new one from that point (one Dijkstra in the browser). The strokes are
// cheap to travel along, so the wave races through each letter and crawls between them.
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'

const canvas = ref<HTMLCanvasElement>()
const ms = ref<string>('')
const ready = ref(false)
const copied = ref(false)
let stop = () => {}

function copy() {
  const t = 'using Pkg; Pkg.add(url = "https://github.com/AlgebraicJulia/PATHPACK.jl")'
  navigator.clipboard?.writeText(t).then(() => { copied.value = true; setTimeout(() => (copied.value = false), 1600) }).catch(() => {})
}

onMounted(async () => {
  const [{ loadBin, bit }, { csr, dijkstra, nearest }, { FieldGL }, { isDark }] = await Promise.all([
    import('./lib/bin'), import('./lib/graph'), import('./lib/fieldgl'), import('./lib/palette')])
  const { meta, a } = await loadBin(withBase('/data/logo.bin.gz'))
  const yieldNow = () => new Promise((r) => setTimeout(r))   // split the setup into short tasks, so the page stays responsive
  const n = meta.n, m = meta.m, xy = new Float32Array(2 * n), E = new Uint32Array(2 * m), inside = new Uint8Array(m), W = new Float32Array(m)
  for (let i = 0; i < 2 * n; i++) xy[i] = meta.lo[i & 1] + a.q[i] / meta.scale
  for (let k = 0, u = 0; k < m; k++) {   // edges: steps of the first vertex, jumps to the second
    u += a.first[k]; const v = u + a.jump[k]; E[2 * k] = u; E[2 * k + 1] = v; inside[k] = bit(a.inbits, k)
    W[k] = Math.hypot(xy[2 * u] - xy[2 * v], xy[2 * u + 1] - xy[2 * v + 1]) * (inside[k] ? meta.inner_weight : 1)
  }
  await yieldNow()
  const g = csr(n, E.filter((_, k) => k % 2 === 0), E.filter((_, k) => k % 2 === 1), W)
  await yieldNow()
  // segments: the joining roads first (thin, faint), then the strokes
  const order: number[] = []; for (let k = 0; k < m; k++) if (!inside[k]) order.push(k); const nout = order.length
  for (let k = 0; k < m; k++) if (inside[k]) order.push(k)
  const seg = new Float32Array(4 * m), wid = new Float32Array(m)
  order.forEach((k, i) => { const u = E[2 * k], v = E[2 * k + 1]; seg.set([xy[2 * u], xy[2 * u + 1], xy[2 * v], xy[2 * v + 1]], 4 * i); wid[i] = inside[k] ? 1.25 : 0.9 })
  await yieldNow()
  if (!canvas.value) return
  const fg = new FieldGL(canvas.value!); fg.pad = 0.02
  fg.setGeometry(seg, wid, [{ start: 0, count: nout, width: 0.9, alpha: 0.42 }, { start: nout, count: m - nout, width: 1.25, alpha: 1 }])
  const near = nearest(xy), vals = new Float64Array(2 * m)
  const onLetter = new Uint8Array(n); for (let k = 0; k < m; k++) if (inside[k]) { onLetter[E[2 * k]] = 1; onLetter[E[2 * k + 1]] = 1 }
  const letters: number[] = []; onLetter.forEach((f, i) => { if (f) letters.push(i) })
  let d = new Float64Array(n), vmax = 1, t0 = 0, last = 0, raf = 0, alive = true, auto = 0, dirty = true, visible = true
  fg.onresize = () => { dirty = true }
  // draw only while a wave moves (or the size or theme changed), and nothing while the logo is off screen
  const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) dirty = true }); io.observe(canvas.value!)
  // no animation for reduced motion, or without a GPU, where every frame would block the page
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches || fg.software
  const wave = (s: number) => {
    const t = performance.now(); dijkstra(g, [s], 'sum', d); ms.value = (performance.now() - t).toFixed(1)
    vmax = 0; for (let i = 0; i < n; i++) if (d[i] < Infinity && d[i] > vmax) vmax = d[i]
    order.forEach((k, i) => { vals[2 * i] = d[E[2 * k]]; vals[2 * i + 1] = d[E[2 * k + 1]] })
    fg.setValues(vals); t0 = performance.now(); last = t0
  }
  const frame = (now: number) => {
    if (!alive) return
    if (!visible) { raf = requestAnimationFrame(frame); return }
    const dark = isDark(); if ((fg.theme === 'dark') !== dark) { fg.setTheme(dark ? 'dark' : 'light'); dirty = true }
    const p = Math.min(1, (now - t0) / (reduce ? 1 : 1700)), e = 1 - Math.pow(1 - p, 3)
    if (p < 1 || dirty) fg.draw({ reach: e * vmax * 1.02, vmax, front: p < 1 ? vmax * 0.07 : 0, glow: dark ? 0.5 : 0.35,
      base: dark ? [0.9, 0.92, 0.96, 0.13] : [0.1, 0.12, 0.18, 0.1] })
    dirty = false
    // a new wave from a random point of a letter every few seconds, until the visitor clicks
    if (!reduce && auto >= 0 && now - last > 5200) wave(letters[Math.floor(Math.random() * letters.length)])
    raf = requestAnimationFrame(frame)
  }
  const onClick = (ev: MouseEvent) => {
    const r = canvas.value!.getBoundingClientRect(), [x, y] = fg.toWorld(ev.clientX - r.left, ev.clientY - r.top)
    auto = -1; wave(near(x, y))
  }
  canvas.value!.addEventListener('click', onClick)
  wave(meta.src); raf = requestAnimationFrame(frame); ready.value = true
  stop = () => { alive = false; cancelAnimationFrame(raf); io.disconnect(); canvas.value?.removeEventListener('click', onClick); fg.destroy() }
})
onBeforeUnmount(() => stop())
</script>

<template>
  <section class="pp-hero">
    <div class="pp-hero-text">
      <p class="pp-eyebrow">Julia · CPUs · NVIDIA GPUs</p>
      <h1 class="pp-hero-title">Every path,<br>between every pair.</h1>
      <p class="pp-hero-sub">
        PATHPACK.jl computes shortest paths, widest paths, reachability and their relatives between
        <em>all pairs</em> of vertices of a sparse graph, as one sparse factorization. One call, any semiring,
        on multithreaded CPUs and on one GPU or many.
      </p>
      <div class="pp-hero-actions">
        <a class="pp-btn pp-btn-brand" :href="withBase('/generated/road_networks')">Get started</a>
        <a class="pp-btn" href="https://github.com/AlgebraicJulia/PATHPACK.jl">GitHub</a>
      </div>
      <button class="pp-install" type="button" @click="copy" :aria-label="'Copy the install command'">
        <span class="pp-install-prompt">julia&gt;</span>
        <code>]add https://github.com/AlgebraicJulia/PATHPACK.jl</code>
        <span class="pp-install-copy">{{ copied ? 'copied' : 'copy' }}</span>
      </button>
    </div>
    <figure class="pp-hero-fig">
      <div class="pp-hero-stage">
        <img class="pp-hero-ph pp-light" :class="{ gone: ready }" :src="withBase('/logo.svg')" alt="" aria-hidden="true">
        <img class="pp-hero-ph pp-dark" :class="{ gone: ready }" :src="withBase('/logo-dark.svg')" alt="" aria-hidden="true">
        <canvas ref="canvas" class="pp-hero-canvas" aria-label="The PATHPACK logo as a network; click to send a distance wave"></canvas>
      </div>
      <figcaption>
        The logo is a graph: 35,466 points. <strong>Click anywhere</strong> to send a distance wave from there:
        one Dijkstra in your browser<span v-if="ms">, {{ ms }} ms</span>. PATHPACK.jl computes all 1.26 billion
        pairs in 46 ms on one NVIDIA B200.
      </figcaption>
    </figure>
  </section>
</template>
