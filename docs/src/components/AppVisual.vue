<script setup lang="ts">
// An application picture that colors itself: /media/apps/<name>.webp holds, per pixel, the distance level from the
// source (R, 1..255; 0 = always grey), a brightness (G) and the coverage (B), rendered from the real data
// (pathpack-film/site_assets/cards/render_cards.py). When it scrolls into view the distances sweep out from the
// source, with a bright front; hovering plays the sweep again. It follows the light/dark theme.
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{ name: string; ring?: boolean; label?: string }>()
const cv = ref<HTMLCanvasElement>()
let raf = 0, alive = true, replay = () => {}, mo: MutationObserver | null = null

onMounted(async () => {
  const [{ nearView }, { ramp, isDark }] = await Promise.all([import('./lib/bin'), import('./lib/palette')])
  await nearView(cv.value!, '300px')
  if (!alive) return
  const img = new Image(); img.decoding = 'async'; img.src = withBase(`/media/apps/${props.name}.webp`)
  await img.decode().catch(() => {})
  if (!alive || !img.naturalWidth) return
  const W = img.naturalWidth, H = img.naturalHeight, c = cv.value!
  c.width = W; c.height = H
  const g = c.getContext('2d')!
  g.drawImage(img, 0, 0)
  const src = g.getImageData(0, 0, W, H).data
  // pixels by distance level: buckets[0] are drawn grey, the others colored once the sweep reaches them
  const counts = new Uint32Array(256)
  for (let p = 0; p < src.length; p += 4) if (src[p + 2]) counts[src[p]]++
  const start = new Uint32Array(257); for (let v = 0; v < 256; v++) start[v + 1] = start[v] + counts[v]
  const pix = new Uint32Array(start[256]), fill = start.slice(0, 256)
  for (let p = 0; p < src.length; p += 4) if (src[p + 2]) pix[fill[src[p]]++] = p
  // the source: the covered pixel nearest the start of the sweep
  let ring: [number, number] | null = null
  if (props.ring) for (let v = 1; v < 256 && !ring; v++) for (let k = start[v]; k < start[v + 1]; k++) { const p = pix[k]; if (src[p + 2] > 160) { ring = [(p / 4) % W, Math.floor(p / 4 / W)]; break } }

  const out = g.createImageData(W, H), o = out.data
  let L = ramp('light'), grey = [198, 204, 216], ink = '#1a1f2e', dark = false
  function theme() {
    dark = isDark(); L = ramp(dark ? 'dark' : 'light')
    grey = dark ? [70, 78, 102] : [198, 204, 216]; ink = dark ? '#eef1f7' : '#1a1f2e'
  }
  // paint level v (glow 0..1 toward white at the sweep's front)
  function paint(v: number, colored: boolean, glow = 0) {
    const r0 = colored ? L[v * 4] : grey[0], g0 = colored ? L[v * 4 + 1] : grey[1], b0 = colored ? L[v * 4 + 2] : grey[2]
    const w = glow * (dark ? 0.55 : 0.4), top = dark ? 255 : 255
    const r = r0 + (top - r0) * w, gg = g0 + (top - g0) * w, b = b0 + (top - b0) * w
    for (let k = start[v]; k < start[v + 1]; k++) {
      const p = pix[k], s = src[p + 1] / 255
      o[p] = r * s; o[p + 1] = gg * s; o[p + 2] = b * s; o[p + 3] = Math.min(255, src[p + 2] * 1.05)
    }
  }
  function show() {
    g.putImageData(out, 0, 0)
    if (ring) {
      g.lineWidth = Math.max(3, W / 220); g.strokeStyle = ink; g.globalAlpha = 0.9
      g.beginPath(); g.arc(ring[0], ring[1], W / 70, 0, 6.2832); g.stroke(); g.globalAlpha = 1
    }
  }
  function full(cut: number) { for (let v = 0; v < 256; v++) paint(v, v > 0 && v <= cut) }

  const FRONT = 14, DUR = 2200
  let t0 = 0, cut = -1
  function frame(now: number) {
    if (!alive) return
    if (!t0) t0 = now
    const p = Math.min(1, (now - t0) / DUR), e = 1 - Math.pow(1 - p, 2.2)
    const next = Math.round(e * (255 + FRONT))
    // levels that entered the front, and the ones that left it
    for (let v = Math.max(1, cut - FRONT); v <= Math.min(255, next); v++) paint(v, true, Math.max(0, 1 - (next - v) / FRONT))
    cut = next; show()
    if (p < 1) raf = requestAnimationFrame(frame); else { raf = 0; full(255); show() }
  }
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  replay = () => {
    if (still || raf) return
    full(0); cut = 0; t0 = 0; raf = requestAnimationFrame(frame)
  }
  theme()
  if (still) { full(255); show() } else { full(0); show(); replay() }
  mo = new MutationObserver(() => { const d = dark; theme(); if (d !== dark && !raf) { full(255); show() } })
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
})
onBeforeUnmount(() => { alive = false; cancelAnimationFrame(raf); mo?.disconnect() })
</script>

<template>
  <canvas ref="cv" class="pp-av" role="img" :aria-label="label || name" @pointerenter="replay()"></canvas>
</template>
