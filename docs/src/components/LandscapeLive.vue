<script setup lang="ts">
// One landscape, three questions: north-central Florida as 67,411 land-cover cells (NLCD 2021), an 8-neighbour cost
// graph. Click a cell; "cheapest route" (min-plus) and "worst barrier" (min-max) are recomputed live in the browser
// from it; "where a wandering animal goes" (plus-times, a linear solve) is PATHPACK.jl's answer from the Ocala forest.
import { onMounted, onBeforeUnmount, ref, computed } from 'vue'
import { withBase } from 'vitepress'

type Mode = 'cost' | 'barrier' | 'walk'
const canvas = ref<HTMLCanvasElement>()
const mode = ref<Mode>('cost'), ms = ref(''), where = ref('the Ocala National Forest')
const CALLS: Record<Mode, [string, string, string, string, string]> = {
  cost: ['MinPlus', 'land', 'What is the cheapest route?', 'near', 'far'],
  barrier: ['MinMax', 'land', 'What is the worst barrier on the way?', 'mild', 'severe'],
  walk: ['PlusProd', 'walk', 'Where does a wandering animal go?', 'rarely', 'often'],
}
const GPU_MS: Record<Mode, number> = { cost: 98, barrier: 105, walk: 177 }
const info = computed(() => CALLS[mode.value])
let api: { set: (m: Mode) => void } | null = null
let stop = () => {}
function setMode(m: Mode) { mode.value = m; api?.set(m) }

onMounted(async () => {
  await (await import('./lib/bin')).nearView(canvas.value!)
  const [{ loadBin, bit }, { csr, dijkstra }, { ramp, isDark }] = await Promise.all([import('./lib/bin'), import('./lib/graph'), import('./lib/palette')])
  const { meta, a } = await loadBin(withBase('/data/landscape.bin.gz'))
  const n = meta.n, nx = meta.nx, ny = meta.ny, cls = a.cls as Uint8Array
  const rc = new Uint16Array(2 * n); for (let c = 0, i = 0; c < nx * ny; c++) if (bit(a.mask, c)) { rc[2 * i] = Math.floor(c / nx); rc[2 * i + 1] = c % nx; i++ }
  const w4 = new Float32Array(4 * n); for (let k = 0; k < 4 * n; k++) { const q = a.w4[k]; w4[k] = q ? Math.exp(meta.wlo + ((q - 1) / 65534) * (meta.whi - meta.wlo)) : 0 }
  const grid = new Int32Array(nx * ny).fill(-1); for (let i = 0; i < n; i++) grid[rc[2 * i] * nx + rc[2 * i + 1]] = i
  const DR = [[0, 1], [1, 1], [1, 0], [1, -1]], I: number[] = [], J: number[] = [], W: number[] = []
  for (let i = 0; i < n; i++) for (let k = 0; k < 4; k++) if (w4[4 * i + k] > 0) {
    const j = grid[(rc[2 * i] + DR[k][0]) * nx + rc[2 * i + 1] + DR[k][1]]; if (j >= 0) { I.push(i); J.push(j); W.push(w4[4 * i + k]) }
  }
  const g = csr(n, I, J, W)
  const LC: Record<number, number[]> = { 11: [141, 180, 212], 21: [216, 196, 196], 22: [201, 169, 169], 23: [176, 131, 131], 24: [138, 90, 90], 31: [217, 210, 196], 41: [143, 179, 131], 42: [111, 154, 107], 43: [130, 168, 120], 52: [200, 192, 138], 71: [220, 213, 158], 81: [227, 220, 150], 82: [217, 194, 122], 90: [127, 168, 146], 95: [156, 194, 180] }
  const off = document.createElement('canvas'); off.width = nx; off.height = ny
  const og = off.getContext('2d')!, img = og.createImageData(nx, ny), px = img.data
  const ctx = canvas.value!.getContext('2d')!
  const rank = new Uint8Array(n), d = new Float64Array(n)
  // the walk map, fixed: visits on a log scale, most visited = 1 (precomputed)
  const walkRank = a.level as Uint8Array
  let sources: number[] = Array.from(a.patch as Uint32Array), t0 = 0, raf = 0, alive = true, dirty = true, theme = ''
  const compute = () => {
    if (mode.value === 'walk') { rank.set(walkRank); return }
    const t = performance.now(); dijkstra(g, sources, mode.value === 'cost' ? 'sum' : 'max', d); ms.value = (performance.now() - t).toFixed(0)
    const o = Array.from({ length: n }, (_, i) => i).sort((x, y) => d[x] - d[y])
    o.forEach((v, r) => { rank[v] = 1 + Math.floor((254 * r) / (n - 1)) })
  }
  const paint = (cut: number, dark: boolean) => {
    const L = ramp(dark ? 'dark' : 'light'), gr = dark ? [18, 22, 33] : [255, 255, 255], mix = dark ? 0.55 : 0.3
    px.fill(0)
    for (let i = 0; i < n; i++) {
      const q = 4 * (rc[2 * i] * nx + rc[2 * i + 1]), b = LC[meta.classes[cls[i] - 1]] || [200, 200, 200], v = rank[i]
      let r = b[0] + (gr[0] - b[0]) * mix, g2 = b[1] + (gr[1] - b[1]) * mix, bl = b[2] + (gr[2] - b[2]) * mix
      if (v <= cut) { const k = (mode.value === 'walk' ? 256 - v : v) * 4, f = cut - v < 10 && cut < 255 ? 0.97 : 0.84
        r += (L[k] - r) * f; g2 += (L[k + 1] - g2) * f; bl += (L[k + 2] - bl) * f }
      px[q] = r; px[q + 1] = g2; px[q + 2] = bl; px[q + 3] = 255
    }
    og.putImageData(img, 0, 0)
    const c = canvas.value!, dpr = Math.min(devicePixelRatio || 1, 2), R = c.getBoundingClientRect()
    if (c.width !== Math.round(R.width * dpr)) { c.width = Math.round(R.width * dpr); c.height = Math.round(R.height * dpr) }
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; ctx.clearRect(0, 0, c.width, c.height)
    ctx.drawImage(off, 0, 0, c.width, c.height)
    // the source: a ring
    const sc = c.width / nx; let sx = 0, sy = 0; for (const s of sources) { sx += rc[2 * s + 1]; sy += rc[2 * s] } sx /= sources.length; sy /= sources.length
    ctx.lineWidth = 3 * dpr; ctx.strokeStyle = dark ? '#eef1f7' : '#1a1f2e'; ctx.fillStyle = dark ? '#121621' : '#ffffff'
    ctx.beginPath(); ctx.arc((sx + 0.5) * sc, (sy + 0.5) * sc, 7 * dpr, 0, 6.283); ctx.fill(); ctx.stroke()
  }
  api = { set: () => { compute(); t0 = performance.now() } }
  const frame = (now: number) => {
    if (!alive) return
    const dark = isDark(), th = dark ? 'd' : 'l', p = Math.min(1, (now - t0) / 1300), cut = Math.round((1 - Math.pow(1 - p, 3)) * 255)
    if (p < 1 || dirty || th !== theme) { paint(cut, dark); dirty = false; theme = th }
    raf = requestAnimationFrame(frame)
  }
  const onClick = (ev: MouseEvent) => {
    const r = canvas.value!.getBoundingClientRect(), cx = Math.floor(((ev.clientX - r.left) / r.width) * nx), cy = Math.floor(((ev.clientY - r.top) / r.height) * ny)
    let best = -1, bd = Infinity
    for (let dy = -6; dy <= 6; dy++) for (let dx = -6; dx <= 6; dx++) { const y = cy + dy, x = cx + dx; if (y < 0 || x < 0 || y >= ny || x >= nx) continue; const i = grid[y * nx + x]; if (i >= 0 && dx * dx + dy * dy < bd) { bd = dx * dx + dy * dy; best = i } }
    if (best < 0) return
    sources = [best]; where.value = 'your cell'
    if (mode.value === 'walk') mode.value = 'cost'
    compute(); t0 = performance.now()
  }
  canvas.value!.addEventListener('click', onClick)
  compute(); t0 = performance.now(); raf = requestAnimationFrame(frame)
  stop = () => { alive = false; cancelAnimationFrame(raf); canvas.value?.removeEventListener('click', onClick) }
})
onBeforeUnmount(() => stop())
</script>

<template>
  <div class="pp-demo">
    <div class="pp-demo-bar">
      <div class="pp-seg" role="tablist">
        <button :class="{ on: mode === 'cost' }" @click="setMode('cost')"><code>MinPlus</code> cheapest route</button>
        <button :class="{ on: mode === 'barrier' }" @click="setMode('barrier')"><code>MinMax</code> worst barrier</button>
        <button :class="{ on: mode === 'walk' }" @click="setMode('walk')"><code>PlusProd</code> wandering animal</button>
      </div>
    </div>
    <div class="pp-land">
      <canvas ref="canvas" class="pp-land-canvas" aria-label="A landscape colored by the chosen semiring's answer"></canvas>
      <div class="pp-land-side">
        <p class="pp-code">mstar(<span>{{ info[0] }}</span>(), {{ info[1] }})</p>
        <h3>{{ info[2] }}</h3>
        <p v-if="mode !== 'walk'">From {{ where }}, recomputed live in your browser: <b>{{ ms }} ms</b>. Click anywhere on the map to move the source.</p>
        <p v-else>Expected visits of a random walker that leaves the Ocala forest and survives each step with probability 0.9999: a sum over every walk, which PATHPACK.jl computes as the star <code>(I − βP)⁻¹</code>.</p>
        <div class="pp-legend pp-legend-inline"><span>{{ info[3] }}</span><i></i><span>{{ info[4] }}</span></div>
        <p class="pp-big"><b>{{ GPU_MS[mode] }} ms</b><span>for all 4.5 billion pairs of cells on one NVIDIA B200</span></p>
        <p class="pp-credit">Land cover: NLCD 2021 (USGS, public domain), 600 m cells; illustrative resistances for a forest mammal.</p>
      </div>
    </div>
  </div>
</template>
