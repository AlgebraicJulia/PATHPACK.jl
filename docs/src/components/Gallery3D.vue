<script setup lang="ts">
// Point clouds with their geodesic distances (distance along the surface or through atomic contacts) from a source,
// drawn as depth-buffered splats with moving distance bands. Drag to turn; double-click to switch the source.
// Each object renders only while it is on screen.
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'

const OBJ = [
  { key: 'spike', m: 21539, name: 'Coronavirus spike', what: '21,539 atoms, contacts within 4.5 Å', time: '48 ms', note: 'PDB 6VYB', a: 0.6, tilt: 0.05, step: 1, splat: 3, zoom: 0.95 },
  { key: 'brain', m: 110000, name: 'Folded human cortex', what: '163,842 vertices', time: '0.90 s', note: 'fsaverage (FreeSurfer, via neuromaps)', a: 0.3, tilt: 0.15, step: 1, splat: 2, zoom: 0.93 },
  { key: 'dragon', m: 90000, name: 'Stanford dragon', what: '436,846 vertices', time: '0.70 s on 8 GPUs', note: 'Stanford 3D Scanning Repository', a: -1.1, tilt: 0.2, step: 1, splat: 2, zoom: 0.95 },
  { key: 'capsid', m: 85620, name: 'Virus capsid, 60 copies', what: '85,620 atoms', time: '296 ms', note: 'PDB 2BUK', a: 0.6, tilt: 0.4, step: 1, splat: 2, zoom: 0.92 },
  { key: 'shuttle', m: 110000, name: 'Space Shuttle', what: '71,279 vertices', time: '114 ms', note: 'Smithsonian Open Access (CC0)', a: -1.1, tilt: 0.25, step: 1, splat: 2, zoom: 1.0 },
  { key: 'palm', m: 100000, name: 'Palm tree', what: '25,115 vertices', time: '31 ms', note: 'Quaternius (CC0)', a: 0.2, tilt: 0.2, step: 1, splat: 2, zoom: 0.95 },
]
const cards = ref<HTMLCanvasElement[]>([])
let stop = () => {}

onMounted(async () => {
  const { ramp, isDark } = await import('./lib/palette')
  const views: any[] = OBJ.map(() => null), visible = OBJ.map(() => false)
  const load = async (i: number) => {
    const o = OBJ[i], img = new Image(); img.src = withBase(`/data/pts/${o.key}_pts.png`); await img.decode()
    const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight
    const g = c.getContext('2d')!; g.drawImage(img, 0, 0); const d = g.getImageData(0, 0, c.width, c.height).data
    const m = o.m, x = new Float32Array(m), y = new Float32Array(m), z = new Float32Array(m), d0 = new Uint8Array(m), d1 = new Uint8Array(m)
    for (let k = 0; k < m; k++) { const p = k * 12
      x[k] = (d[p] * 256 + d[p + 1]) / 65535 - 0.5; y[k] = (d[p + 2] * 256 + d[p + 4]) / 65535 - 0.5; z[k] = (d[p + 5] * 256 + d[p + 6]) / 65535 - 0.5
      d0[k] = d[p + 8]; d1[k] = d[p + 9] }
    const cv = cards.value[i], S = cv.width, ctx = cv.getContext('2d')!
    views[i] = { x, y, z, D: [d0, d1], m, S, ctx, img: ctx.createImageData(S, S), zb: new Float32Array(S * S), yaw: o.a, tilt: o.tilt, src: 0, t0: performance.now(), drag: false, last: performance.now() }
  }
  const draw = (i: number, now: number, dark: boolean, L: Uint8Array, still = false) => {
    const v = views[i], o = OBJ[i]; if (!v) return
    const { x, y, z, S, img, zb } = v, D = v.D[v.src], out = img.data
    if (!v.drag && !still) v.yaw += (now - v.last) * 0.00018; v.last = now
    const reach = still ? 255 : Math.min(1, (now - v.t0) / 2600) * 255, bands = still ? 0 : ((now - v.t0) / 1000) * 30
    out.fill(0); zb.fill(-9)
    const ca = Math.cos(v.yaw), sa = Math.sin(v.yaw), ct = Math.cos(v.tilt), st = Math.sin(v.tilt), k = S * 0.92 * o.zoom, h = S / 2, sp = o.splat * (S / 400)
    const base = dark ? [92, 100, 124] : [205, 210, 222], spi = Math.max(1, Math.round(sp))
    for (let i2 = 0; i2 < v.m; i2 += o.step) {
      const X = ca * x[i2] + sa * z[i2], Z0 = -sa * x[i2] + ca * z[i2], Y = ct * y[i2] - st * Z0, Z = st * y[i2] + ct * Z0
      const px = (h + X * k) | 0, py = (h - Y * k) | 0; if (px < 0 || py < 0 || px > S - spi || py > S - spi) continue
      const dv = D[i2]; let r = base[0], g = base[1], b = base[2]
      if (dv <= reach) { r = L[dv * 4]; g = L[dv * 4 + 1]; b = L[dv * 4 + 2]
        if ((dv + 2560 - bands) % 16 < 2.5) { if (dark) { r = r * 0.35 + 166; g = g * 0.35 + 166; b = b * 0.35 + 166 } else { r *= 0.5; g *= 0.5; b *= 0.5 } } }
      const sh = 0.55 + 0.45 * Math.min(1, Math.max(0, Z + 0.5))
      for (let dy = 0; dy < spi; dy++) for (let dx = 0; dx < spi; dx++) { const j = (py + dy) * S + px + dx
        if (Z > zb[j]) { zb[j] = Z; const q = j * 4; out[q] = r * sh; out[q + 1] = g * sh; out[q + 2] = b * sh; out[q + 3] = 255 } }
    }
    v.ctx.putImageData(img, 0, 0)
  }
  let raf = 0, alive = true
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    const i = cards.value.indexOf(e.target as HTMLCanvasElement); visible[i] = e.isIntersecting
    if (e.isIntersecting && !views[i]) load(i)
  }), { rootMargin: '200px' })
  cards.value.forEach((c, i) => {
    const dpr = Math.min(devicePixelRatio || 1, 2), S = Math.round(Math.min(560, c.getBoundingClientRect().width * dpr)); c.width = S; c.height = S
    io.observe(c)
    let px = 0, py = 0
    c.addEventListener('pointerdown', (ev) => { const v = views[i]; if (!v) return; v.drag = true; px = ev.clientX; py = ev.clientY; c.setPointerCapture(ev.pointerId) })
    c.addEventListener('pointermove', (ev) => { const v = views[i]; if (!v || !v.drag) return; v.yaw += (ev.clientX - px) * 0.01; v.tilt = Math.max(-1.2, Math.min(1.2, v.tilt + (ev.clientY - py) * 0.01)); px = ev.clientX; py = ev.clientY })
    c.addEventListener('pointerup', () => { const v = views[i]; if (v) v.drag = false })
    c.addEventListener('dblclick', () => { const v = views[i]; if (v) { v.src = 1 - v.src; v.t0 = performance.now() } })
  })
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
  const frame = (now: number) => {
    if (!alive) return
    const dark = isDark(), L = ramp(dark ? 'dark' : 'light')
    OBJ.forEach((_, i) => { if (visible[i]) draw(i, now, dark, L, reduce) })
    raf = requestAnimationFrame(frame)
  }
  raf = requestAnimationFrame(frame)
  stop = () => { alive = false; cancelAnimationFrame(raf); io.disconnect() }
})
onBeforeUnmount(() => stop())
</script>

<template>
  <div class="pp-gallery">
    <figure v-for="(o, i) in OBJ" :key="o.key" class="pp-g-card">
      <canvas :ref="(el) => { if (el) cards[i] = el as HTMLCanvasElement }" class="pp-g-canvas" :aria-label="o.name + ', drag to turn'"></canvas>
      <figcaption>
        <b>{{ o.name }}</b><span>{{ o.what }}</span>
        <span class="pp-g-time">all pairs: <strong>{{ o.time }}</strong>{{ o.time.includes('GPUs') ? '' : ' on one GPU' }}</span>
        <span class="pp-credit">{{ o.note }}</span>
      </figcaption>
    </figure>
  </div>
</template>
