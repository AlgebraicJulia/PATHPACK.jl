<script setup lang="ts">
// Isomap unrolling a swiss roll: 10,000 points, their nearest-neighbour graph, every geodesic distance from
// PATHPACK.jl and classical MDS (pathpack-film/site_data/swiss/isomap.jl). The points are colored by geodesic distance
// from one end of the roll, then move from the roll to their Isomap coordinates, hold, and roll back up.
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'

const cv = ref<HTMLCanvasElement>()
let stop = () => {}

onMounted(async () => {
  const [{ loadBin, nearView }, { ramp, isDark }] = await Promise.all([import('./lib/bin'), import('./lib/palette')])
  await nearView(cv.value!, '300px')
  const { meta, a } = await loadBin(withBase('/data/swiss.bin.gz'))
  const n = meta.n as number, p3 = a.p3 as Uint16Array, p2 = a.p2 as Uint16Array, D = a.d as Uint8Array
  const c = cv.value!; if (!c) return
  const dpr = Math.min(devicePixelRatio || 1, 2), W = Math.round(c.getBoundingClientRect().width * dpr), H = Math.round(W * 3 / 4)
  c.width = W; c.height = H
  const g = c.getContext('2d')!, img = g.createImageData(W, H), out = img.data, zb = new Float32Array(W * H)
  const X = new Float32Array(n), Y = new Float32Array(n), Z = new Float32Array(n), U = new Float32Array(n), V = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    X[i] = p3[3 * i] / 65535 - 0.5; Y[i] = p3[3 * i + 1] / 65535 - 0.5; Z[i] = p3[3 * i + 2] / 65535 - 0.5
    U[i] = p2[2 * i] / 65535 - 0.5; V[i] = p2[2 * i + 1] / 65535 - 0.5
  }
  const ease = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t))
  const seg = (t: number, a: number, b: number) => Math.min(1, Math.max(0, (t - a) / (b - a)))
  // one cycle: color sweep, unroll, hold flat, roll back up, rest
  const T = 10.5
  const sp = Math.max(2, Math.round(W / 210))
  function draw(t: number, dark: boolean, L: Uint8Array) {
    const u = t % T
    const reach = ease(seg(u, 0.2, 1.8)) * 255
    const q = ease(seg(u, 2.2, 4.8)) * (1 - ease(seg(u, 7.6, 9.8)))
    const ang = 0.55 + 0.35 * Math.sin(t * 0.25), tilt = 0.32, ca = Math.cos(ang), sa = Math.sin(ang), ct = Math.cos(tilt), st = Math.sin(tilt)
    const k3 = H * 0.86, k2 = W * 0.86, base = dark ? [70, 78, 102] : [198, 204, 216]
    out.fill(0); zb.fill(-9)
    for (let i = 0; i < n; i++) {
      // the roll: its axis (z) toward the viewer at an angle, spinning slowly
      const x0 = ca * X[i] + sa * Z[i], z0 = -sa * X[i] + ca * Z[i]
      const y1 = ct * Y[i] - st * z0, z1 = st * Y[i] + ct * z0
      const sx = W / 2 + ((1 - q) * x0 * k3 + q * U[i] * k2), sy = H / 2 - ((1 - q) * y1 * k3 + q * V[i] * k2)
      const zz = (1 - q) * z1
      const px = sx | 0, py = sy | 0
      if (px < 0 || py < 0 || px > W - sp || py > H - sp) continue
      const dv = D[i]; let r = base[0], gg = base[1], b = base[2]
      if (dv <= reach) { r = L[dv * 4]; gg = L[dv * 4 + 1]; b = L[dv * 4 + 2] }
      const sh = 0.62 + 0.38 * Math.min(1, Math.max(0, zz + 0.5)) * (1 - q) + 0.38 * q
      for (let dy = 0; dy < sp; dy++) for (let dx = 0; dx < sp; dx++) {
        const j = (py + dy) * W + px + dx
        if (zz > zb[j]) { zb[j] = zz; const o = j * 4; out[o] = r * sh; out[o + 1] = gg * sh; out[o + 2] = b * sh; out[o + 3] = 255 }
      }
    }
    g.putImageData(img, 0, 0)
  }
  let raf = 0, alive = true, visible = true, t0 = performance.now()
  const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting })
  io.observe(c)
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches
  const frame = (now: number) => {
    if (!alive) return
    const dark = isDark()
    if (visible || still) draw(still ? 6 : (now - t0) / 1000, dark, ramp(dark ? 'dark' : 'light'))
    if (!still) raf = requestAnimationFrame(frame)
  }
  raf = requestAnimationFrame(frame)
  const mo = new MutationObserver(() => { if (still) frame(0) })
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
  stop = () => { alive = false; cancelAnimationFrame(raf); io.disconnect(); mo.disconnect() }
})
onBeforeUnmount(() => stop())
</script>

<template>
  <canvas ref="cv" class="pp-av" role="img" aria-label="A swiss roll of 10,000 points unrolled by Isomap from PATHPACK.jl's geodesic distances"></canvas>
</template>
