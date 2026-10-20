<script setup lang="ts">
// Paint distances on a surface: the Stanford bunny (8,171 vertices) as a shaded WebGL mesh. Click a point: one Dijkstra
// along the mesh edges, in the browser, gives the distance from it to every vertex; the color and the moving contour
// bands follow it. Drag to turn.
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'

const canvas = ref<HTMLCanvasElement>(), ms = ref(''), ready = ref(false), failed = ref(false)
let stop = () => {}

const VS = `
attribute vec3 a_pos; attribute vec3 a_nrm; attribute float a_d;
uniform mat3 u_rot; uniform vec2 u_sc;
varying vec3 v_n; varying float v_d;
void main() { vec3 p = u_rot * a_pos; v_n = u_rot * a_nrm; v_d = a_d; gl_Position = vec4(p.x * u_sc.x, p.y * u_sc.y, -p.z * 0.5, 1.0); }`
const FS = `
precision highp float;
varying vec3 v_n; varying float v_d;
uniform sampler2D u_lut; uniform float u_reach; uniform float u_vmax; uniform float u_phase; uniform vec3 u_base; uniform float u_dark;
void main() {
  vec3 n = normalize(v_n); if (!gl_FrontFacing) n = -n;
  float dif = max(dot(n, normalize(vec3(0.35, 0.55, 0.75))), 0.0), rim = pow(1.0 - abs(n.z), 2.0);
  float t = clamp(v_d / u_vmax, 0.0, 1.0);
  vec3 c = v_d <= u_reach ? texture2D(u_lut, vec2(t, 0.5)).rgb : u_base;
  float f = fract(t * 14.0 - u_phase);
  if (v_d <= u_reach && f < 0.07) c = mix(c, u_dark > 0.5 ? vec3(0.95) : vec3(0.08, 0.1, 0.16), 0.55);
  float front = smoothstep(u_reach - 0.04 * u_vmax, u_reach, v_d) * step(v_d, u_reach);
  c = mix(c, vec3(1.0, 0.97, 0.9), front * 0.6);
  gl_FragColor = vec4(c * (0.42 + 0.62 * dif) + rim * 0.10, 1.0);
}`

onMounted(async () => {
  const [{ loadBin, nearView }, { csr, dijkstra }, { ramp, isDark }] = await Promise.all([import('./lib/bin'), import('./lib/graph'), import('./lib/palette')])
  await nearView(canvas.value!)
  const { meta, a } = await loadBin(withBase('/data/bunny.bin.gz'))
  const n = meta.n, nf = meta.f, tri = a.tri as Uint16Array
  const P = new Float32Array(3 * n); for (let i = 0; i < 3 * n; i++) P[i] = meta.lo[i % 3] + a.pos[i] / meta.scale
  // smooth normals, and the edges of the mesh as a graph weighted by length
  const N = new Float32Array(3 * n), seen = new Set<number>(), I: number[] = [], J: number[] = [], W: number[] = []
  for (let f = 0; f < nf; f++) {
    const [i, j, k] = [tri[3 * f], tri[3 * f + 1], tri[3 * f + 2]]
    const ux = P[3 * j] - P[3 * i], uy = P[3 * j + 1] - P[3 * i + 1], uz = P[3 * j + 2] - P[3 * i + 2]
    const vx = P[3 * k] - P[3 * i], vy = P[3 * k + 1] - P[3 * i + 1], vz = P[3 * k + 2] - P[3 * i + 2]
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx
    for (const v of [i, j, k]) { N[3 * v] += nx; N[3 * v + 1] += ny; N[3 * v + 2] += nz }
    for (const [p, q] of [[i, j], [j, k], [k, i]]) {
      const key = Math.min(p, q) * 65536 + Math.max(p, q); if (seen.has(key)) continue; seen.add(key)
      I.push(p); J.push(q); W.push(Math.hypot(P[3 * p] - P[3 * q], P[3 * p + 1] - P[3 * q + 1], P[3 * p + 2] - P[3 * q + 2]))
    }
  }
  for (let i = 0; i < n; i++) { const L = Math.hypot(N[3 * i], N[3 * i + 1], N[3 * i + 2]) || 1; N[3 * i] /= L; N[3 * i + 1] /= L; N[3 * i + 2] /= L }
  const g = csr(n, I, J, W), d = new Float64Array(n), D = new Float32Array(n)
  const cv = canvas.value!, gl = cv.getContext('webgl', { antialias: true })
  if (!gl) { failed.value = true; return }
  const sh = (t: number, s: string) => { const o = gl.createShader(t)!; gl.shaderSource(o, s); gl.compileShader(o); return o }
  const prog = gl.createProgram()!; gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog)
  const buf = (data: ArrayBufferView, target = gl.ARRAY_BUFFER) => { const b = gl.createBuffer()!; gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW); return b }
  const bP = buf(P), bN = buf(N), bD = buf(D), bI = buf(tri, gl.ELEMENT_ARRAY_BUFFER)
  const lut = gl.createTexture()!
  let theme = ''
  const setLut = (t: 'light' | 'dark') => { gl.bindTexture(gl.TEXTURE_2D, lut); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, ramp(t)); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE) }
  let yaw = 0.6, pitch = 0.1, vmax = 1, t0 = performance.now(), drag = false, moved = 0, last = performance.now(), raf = 0, alive = true
  const rot = () => { const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch)
    return new Float32Array([cy, sy * sp, -sy * cp, 0, cp, sp, sy, -cy * sp, cy * cp]) }   // column-major: yaw about y, then pitch about x
  const source = (s: number) => {
    const t = performance.now(); dijkstra(g, [s], 'sum', d); ms.value = (performance.now() - t).toFixed(1)
    vmax = 0; for (let i = 0; i < n; i++) { D[i] = d[i]; if (d[i] < Infinity && d[i] > vmax) vmax = d[i] }
    gl.bindBuffer(gl.ARRAY_BUFFER, bD); gl.bufferSubData(gl.ARRAY_BUFFER, 0, D); t0 = performance.now()
  }
  const fit = () => { const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2); const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr); if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h } return [w, h] }
  const scale = (w: number, h: number) => { const s = 0.92; return w > h ? [s * h / w, s] : [s, s * w / h] }
  const frame = (now: number) => {
    if (!alive) return
    const dark = isDark(), th = dark ? 'dark' : 'light'; if (th !== theme) { setLut(th); theme = th }
    if (!drag) yaw += (now - last) * 0.00012; last = now
    const [w, h] = fit(), [sx, sy] = scale(w, h), p = Math.min(1, (now - t0) / 1600), reach = (1 - Math.pow(1 - p, 3)) * vmax * 1.001
    gl.viewport(0, 0, w, h); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST)
    gl.useProgram(prog)
    for (const [name, b] of [['a_pos', bP], ['a_nrm', bN]] as const) { const l = gl.getAttribLocation(prog, name); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 3, gl.FLOAT, false, 0, 0) }
    { const l = gl.getAttribLocation(prog, 'a_d'); gl.bindBuffer(gl.ARRAY_BUFFER, bD); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 1, gl.FLOAT, false, 0, 0) }
    const u = (s: string) => gl.getUniformLocation(prog, s)
    gl.uniformMatrix3fv(u('u_rot'), false, rot()); gl.uniform2f(u('u_sc'), sx, sy)
    gl.uniform1f(u('u_reach'), reach); gl.uniform1f(u('u_vmax'), vmax); gl.uniform1f(u('u_phase'), ((now - t0) / 1000) * 0.6)
    gl.uniform3fv(u('u_base'), dark ? [0.36, 0.39, 0.48] : [0.8, 0.82, 0.87]); gl.uniform1f(u('u_dark'), dark ? 1 : 0)
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, lut); gl.uniform1i(u('u_lut'), 0)
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bI); gl.drawElements(gl.TRIANGLES, 3 * nf, gl.UNSIGNED_SHORT, 0)
    raf = requestAnimationFrame(frame)
  }
  // picking: the front-most vertex near the click, in screen space
  const pick = (cx: number, cy: number) => {
    const [w, h] = fit(), [sx, sy] = scale(w, h), R = rot(), r = cv.getBoundingClientRect(), dpr = w / r.width
    const X = cx * dpr, Y = cy * dpr, rad = 14 * dpr; let best = -1, bz = -Infinity, bd = Infinity
    for (let i = 0; i < n; i++) {
      const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2]
      const px = R[0] * x + R[3] * y + R[6] * z, py = R[1] * x + R[4] * y + R[7] * z, pz = R[2] * x + R[5] * y + R[8] * z
      const sxp = (px * sx * 0.5 + 0.5) * w, syp = (1 - (py * sy * 0.5 + 0.5)) * h, dd = Math.hypot(sxp - X, syp - Y)
      if (dd < rad && (pz > bz + 0.02 || (Math.abs(pz - bz) <= 0.02 && dd < bd))) { bz = pz; bd = dd; best = i }
    }
    return best
  }
  let px0 = 0, py0 = 0
  cv.addEventListener('pointerdown', (e) => { drag = true; moved = 0; px0 = e.clientX; py0 = e.clientY; cv.setPointerCapture(e.pointerId) })
  cv.addEventListener('pointermove', (e) => { if (!drag) return; const dx = e.clientX - px0, dy = e.clientY - py0; moved += Math.abs(dx) + Math.abs(dy); yaw += dx * 0.01; pitch = Math.max(-1.3, Math.min(1.3, pitch + dy * 0.01)); px0 = e.clientX; py0 = e.clientY })
  cv.addEventListener('pointerup', (e) => { drag = false; if (moved < 4) { const r = cv.getBoundingClientRect(), s = pick(e.clientX - r.left, e.clientY - r.top); if (s >= 0) source(s) } })
  // start from the tip of an ear: the highest vertex
  let top = 0; for (let i = 1; i < n; i++) if (P[3 * i + 1] > P[3 * top + 1]) top = i
  source(top); ready.value = true; raf = requestAnimationFrame(frame)
  stop = () => { alive = false; cancelAnimationFrame(raf) }
})
onBeforeUnmount(() => stop())
</script>

<template>
  <div class="pp-demo pp-mesh">
    <div class="pp-mesh-stage">
      <canvas ref="canvas" class="pp-mesh-canvas" aria-label="The Stanford bunny; click to measure distances from a point, drag to turn"></canvas>
      <p v-if="failed" class="pp-mesh-msg">This demo needs WebGL.</p>
    </div>
    <div class="pp-mesh-side">
      <p class="pp-code">mstar(<span>MinPlus</span>(), surface)</p>
      <h3>Click the bunny.</h3>
      <p>Distance along the surface from where you click to every other point, with moving contours. Drag to turn it.
        Your browser runs one Dijkstra over the mesh's 8,171 vertices: <b>{{ ms || '…' }} ms</b>.</p>
      <p class="pp-big"><b>55.6 ms</b><span>for every pair of the full 34,834-vertex scan on one NVIDIA B200, 2× faster than ROME and 31× faster than cuGraph</span></p>
      <p class="pp-credit">Stanford Bunny, Stanford 3D Scanning Repository. Distances along mesh edges, which run about 7% longer than exact surface distances.</p>
    </div>
  </div>
</template>
