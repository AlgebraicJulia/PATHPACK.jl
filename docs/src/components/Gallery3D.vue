<script setup lang="ts">
// Point clouds with their geodesic distances (distance along the surface or through atomic contacts) from a source,
// drawn with WebGL at the screen's full resolution, lit like the bunny: the surfaces as small discs facing their normals
// (from the points' neighbours, pathpack-film/site_data/gallery/normals.py), the molecules as shaded atoms.
// Drag to turn; double-click to switch the source. Each object renders only while it is on screen.
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'

const OBJ = [
  { key: 'spike', m: 21539, atoms: true, name: 'Coronavirus spike', what: '21,539 atoms, contacts within 4.5 Å', time: '48 ms', note: 'PDB 6VYB', a: 0.6, tilt: 0.05, step: 1, splat: 3, zoom: 0.95 },
  { key: 'brain', m: 110000, name: 'Folded human cortex', what: '163,842 vertices', time: '0.90 s', note: 'fsaverage (FreeSurfer, via neuromaps)', a: 0.3, tilt: 0.15, step: 1, splat: 2, zoom: 0.93 },
  { key: 'dragon', m: 90000, name: 'Stanford dragon', what: '436,846 vertices', time: '0.70 s on 8 GPUs', note: 'Stanford 3D Scanning Repository', a: -1.1, tilt: 0.2, step: 1, splat: 2, zoom: 0.95 },
  { key: 'capsid', m: 85620, atoms: true, name: 'Virus capsid, 60 copies', what: '85,620 atoms', time: '296 ms', note: 'PDB 2BUK', a: 0.6, tilt: 0.4, step: 1, splat: 2, zoom: 0.92 },
  { key: 'shuttle', m: 110000, name: 'Space Shuttle', what: '71,279 vertices', time: '114 ms', note: 'Smithsonian Open Access (CC0)', a: -1.1, tilt: 0.25, step: 1, splat: 2, zoom: 1.0 },
  { key: 'palm', m: 100000, name: 'Palm tree', what: '25,115 vertices', time: '31 ms', note: 'Quaternius (CC0)', a: 0.2, tilt: 0.2, step: 1, splat: 2, zoom: 0.95 },
]
const cards = ref<HTMLCanvasElement[]>([])
let stop = () => {}

const VS = `
attribute vec3 a_p; attribute float a_d0; attribute float a_d1; attribute vec2 a_n;
uniform mat3 u_rot; uniform float u_k; uniform float u_size; uniform float u_src;
varying float v_d; varying float v_z; varying vec3 v_n;
void main() {
  vec3 p = u_rot * a_p;
  gl_Position = vec4(p.x * u_k, p.y * u_k, -p.z * 0.9, 1.0);
  gl_PointSize = u_size;
  v_d = mix(a_d0, a_d1, u_src); v_z = p.z;
  // octahedral-decoded normal, turned with the object
  vec2 f = a_n * 2.0 - 1.0; vec3 n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
  if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * sign(n.xy);
  v_n = u_rot * normalize(n);
}`
const FS = `
precision highp float;
uniform sampler2D u_lut; uniform float u_reach; uniform float u_bands; uniform vec3 u_base; uniform float u_dark; uniform float u_atoms;
varying float v_d; varying float v_z; varying vec3 v_n;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0; c.y = -c.y;
  vec3 n;
  if (u_atoms > 0.5) {                         // a sphere
    float r2 = dot(c, c); if (r2 > 1.0) discard;
    n = vec3(c, sqrt(1.0 - r2));
  } else {                                     // a disc facing the normal: inside it if its back-projection is
    n = normalize(v_n); if (n.z < 0.0) n = -n; // within the unit disc (normals steeper than 75 degrees clamped)
    float nz = max(n.z, 0.26), qz = -(c.x * n.x + c.y * n.y) / nz;
    if (dot(c, c) + qz * qz > 1.0) discard;
  }
  vec3 col = u_base;
  if (v_d <= u_reach) {
    col = texture2D(u_lut, vec2((v_d + 0.5) / 256.0, 0.5)).rgb;
    if (mod(v_d + 2560.0 - u_bands, 16.0) < 2.5) col = u_dark > 0.5 ? col * 0.35 + 0.65 : col * 0.5;
  }
  float dif = max(dot(n, normalize(vec3(0.35, 0.55, 0.75))), 0.0), rim = pow(1.0 - abs(n.z), 2.0);
  gl_FragColor = vec4(col * (0.42 + 0.62 * dif) + rim * 0.10, 1.0);
}`

// the normals of a surface's points (two bytes each), gzip-compressed
async function normals(url: string) {
  const r = await fetch(url); if (!r.ok || !r.body) return null
  return new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer())
}

onMounted(async () => {
  const { ramp, isDark } = await import('./lib/palette')
  const views: any[] = OBJ.map(() => null), visible = OBJ.map(() => false)
  const load = async (i: number) => {
    views[i] = 'loading'
    const o = OBJ[i], img = new Image(); img.src = withBase(`/data/pts/${o.key}_pts.png`); await img.decode()
    const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight
    const g = c.getContext('2d')!; g.drawImage(img, 0, 0); const d = g.getImageData(0, 0, c.width, c.height).data
    const m = Math.min(o.m, Math.floor(d.length / 12)), P = new Float32Array(3 * m), D0 = new Float32Array(m), D1 = new Float32Array(m)
    for (let k = 0; k < m; k++) { const p = k * 12
      P[3 * k] = (d[p] * 256 + d[p + 1]) / 65535 - 0.5; P[3 * k + 1] = (d[p + 2] * 256 + d[p + 4]) / 65535 - 0.5; P[3 * k + 2] = (d[p + 5] * 256 + d[p + 6]) / 65535 - 0.5
      D0[k] = d[p + 8]; D1[k] = d[p + 9] }
    const N = o.atoms ? null : await normals(withBase(`/data/pts/${o.key}_n.bin.gz`))
    const cv = cards.value[i], gl = cv.getContext('webgl', { antialias: true, alpha: true, premultipliedAlpha: false })
    if (!gl) { views[i] = null; return }
    const sh = (t: number, src: string) => { const x = gl.createShader(t)!; gl.shaderSource(x, src); gl.compileShader(x); return x }
    const prog = gl.createProgram()!; gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog)
    const buf = (a: Float32Array | Uint8Array) => { const b = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, a, gl.STATIC_DRAW); return b }
    const lut = gl.createTexture()!
    views[i] = { gl, prog, m, bP: buf(P), bD0: buf(D0), bD1: buf(D1), bN: buf(N && N.length >= 2 * m ? N.subarray(0, 2 * m) : new Uint8Array(2 * m).fill(128)), lut, theme: '', yaw: o.a, tilt: o.tilt, src: 0, t0: performance.now(), drag: false, last: performance.now() }
  }
  const fit = (cv: HTMLCanvasElement) => {
    const dpr = Math.min(devicePixelRatio || 1, 2), S = Math.round(cv.getBoundingClientRect().width * dpr)
    if (S > 0 && cv.width !== S) { cv.width = S; cv.height = S }
    return cv.width
  }
  const draw = (i: number, now: number, dark: boolean, still = false) => {
    const v = views[i], o = OBJ[i]; if (!v || v === 'loading') return
    const { gl, prog } = v, cv = cards.value[i], S = fit(cv)
    if (!v.drag && !still) v.yaw += (now - v.last) * 0.00018; v.last = now
    const th = dark ? 'dark' : 'light'
    if (v.theme !== th) {
      gl.bindTexture(gl.TEXTURE_2D, v.lut); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, ramp(th))
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      v.theme = th
    }
    const reach = still ? 255 : Math.min(1, (now - v.t0) / 2600) * 255, bands = still ? -100 : ((now - v.t0) / 1000) * 30
    const ca = Math.cos(v.yaw), sa = Math.sin(v.yaw), ct = Math.cos(v.tilt), st = Math.sin(v.tilt)
    // screen x = ca x + sa z; screen y = ct y - st (-sa x + ca z); depth z = st y + ct (-sa x + ca z) (column-major)
    const R = new Float32Array([ca, st * sa, -ct * sa, 0, ct, st, sa, -st * ca, ct * ca])
    gl.viewport(0, 0, S, S); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST)
    gl.useProgram(prog)
    const at = (name: string, b: WebGLBuffer, k: number) => { const l = gl.getAttribLocation(prog, name); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, k, gl.FLOAT, false, 0, 0) }
    at('a_p', v.bP, 3); at('a_d0', v.bD0, 1); at('a_d1', v.bD1, 1)
    { const l = gl.getAttribLocation(prog, 'a_n'); gl.bindBuffer(gl.ARRAY_BUFFER, v.bN); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.UNSIGNED_BYTE, true, 0, 0) }
    const u = (n: string) => gl.getUniformLocation(prog, n)
    gl.uniformMatrix3fv(u('u_rot'), false, R); gl.uniform1f(u('u_k'), 2 * 0.92 * o.zoom)
    // discs and atoms overlap their neighbours, so the surface reads as closed
    gl.uniform1f(u('u_size'), Math.max(2, o.splat * (o.atoms ? 1.8 : 2.3) * S / 400)); gl.uniform1f(u('u_src'), v.src); gl.uniform1f(u('u_atoms'), o.atoms ? 1 : 0)
    gl.uniform1f(u('u_reach'), reach); gl.uniform1f(u('u_bands'), bands); gl.uniform1f(u('u_dark'), dark ? 1 : 0)
    gl.uniform3fv(u('u_base'), dark ? [92 / 255, 100 / 255, 124 / 255] : [205 / 255, 210 / 255, 222 / 255])
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, v.lut); gl.uniform1i(u('u_lut'), 0)
    gl.drawArrays(gl.POINTS, 0, v.m)
  }
  let raf = 0, alive = true
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    const i = cards.value.indexOf(e.target as HTMLCanvasElement); visible[i] = e.isIntersecting
    if (e.isIntersecting && !views[i]) load(i)
  }), { rootMargin: '200px' })
  cards.value.forEach((c, i) => {
    io.observe(c)
    let px = 0, py = 0
    c.addEventListener('pointerdown', (ev) => { const v = views[i]; if (!v || v === 'loading') return; v.drag = true; px = ev.clientX; py = ev.clientY; c.setPointerCapture(ev.pointerId) })
    c.addEventListener('pointermove', (ev) => { const v = views[i]; if (!v || !v.drag) return; v.yaw += (ev.clientX - px) * 0.01; v.tilt = Math.max(-1.2, Math.min(1.2, v.tilt + (ev.clientY - py) * 0.01)); px = ev.clientX; py = ev.clientY })
    c.addEventListener('pointerup', () => { const v = views[i]; if (v && v !== 'loading') v.drag = false })
    c.addEventListener('dblclick', () => { const v = views[i]; if (v && v !== 'loading') { v.src = 1 - v.src; v.t0 = performance.now() } })
  })
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
  const frame = (now: number) => {
    if (!alive) return
    const dark = isDark()
    OBJ.forEach((_, i) => { if (visible[i]) draw(i, now, dark, reduce) })
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
