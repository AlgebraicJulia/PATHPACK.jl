// Line segments colored by a value (a distance) on the GPU. Each segment is a quad with a real width in pixels; the
// value is interpolated along it. Segments with a value beyond `reach` are drawn in the base color; the band just
// inside the front glows. Values are updated without touching the geometry, so a new source costs one buffer upload.
import { ramp } from './palette'

const VS = `
attribute vec2 a_pos; attribute vec2 a_other; attribute float a_side; attribute float a_val; attribute float a_w;
uniform vec2 u_scale; uniform vec2 u_offset; uniform vec2 u_px;
varying float v_val;
void main() {
  vec2 p = a_pos * u_scale + u_offset, q = a_other * u_scale + u_offset;
  vec2 d = (q - p) / u_px; float L = length(d);
  vec2 nrm = L > 1e-6 ? vec2(-d.y, d.x) / L : vec2(0.0);
  gl_Position = vec4(p + nrm * a_side * a_w * 0.5 * u_px, 0.0, 1.0);
  v_val = a_val;
}`
const FS = `
precision highp float;
varying float v_val;
uniform float u_reach; uniform float u_vmax; uniform float u_front; uniform float u_alpha; uniform float u_glow;
uniform vec4 u_base; uniform sampler2D u_lut;
void main() {
  if (v_val > u_reach) { gl_FragColor = vec4(u_base.rgb, u_base.a * u_alpha); return; }
  vec3 c = texture2D(u_lut, vec2(clamp(v_val / u_vmax, 0.0, 1.0), 0.5)).rgb;
  float f = u_front > 0.0 ? smoothstep(u_reach - u_front, u_reach, v_val) : 0.0;
  gl_FragColor = vec4(mix(c, vec3(1.0, 0.97, 0.9), f * u_glow), u_alpha);
}`

export interface Layer { start: number; count: number; width: number; alpha: number }
export class FieldGL {
  gl: WebGLRenderingContext; prog: WebGLProgram; nseg = 0; layers: Layer[] = []
  bufGeo: WebGLBuffer; bufVal: WebGLBuffer; bufIdx: WebGLBuffer; lut: WebGLTexture
  box = [0, 0, 1, 1]; pad = 0.03; ro: ResizeObserver; theme: 'light' | 'dark' = 'light'
  onresize: () => void = () => {}
  constructor(public canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl', { antialias: true, premultipliedAlpha: false, alpha: true })!
    this.gl = gl
    const sh = (t: number, s: string) => { const o = gl.createShader(t)!; gl.shaderSource(o, s); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o) || 'shader'); return o }
    const p = gl.createProgram()!; gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p)
    this.prog = p; this.bufGeo = gl.createBuffer()!; this.bufVal = gl.createBuffer()!; this.bufIdx = gl.createBuffer()!
    gl.getExtension('OES_element_index_uint')
    this.lut = gl.createTexture()!; this.setTheme('light')
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(canvas); this.resize()
  }
  setTheme(t: 'light' | 'dark') {
    const gl = this.gl; this.theme = t
    gl.bindTexture(gl.TEXTURE_2D, this.lut)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, ramp(t))
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  }
  resize() {
    const r = this.canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr))
    if (w !== this.canvas.width || h !== this.canvas.height) { this.canvas.width = w; this.canvas.height = h; this.onresize() }
  }
  // seg: x0, y0, x1, y1 per segment (world units); w: per-segment width in CSS pixels
  setGeometry(seg: Float32Array, widths: Float32Array, layers: Layer[]) {
    const gl = this.gl, n = seg.length / 4; this.nseg = n; this.layers = layers
    const g = new Float32Array(n * 4 * 6)   // per vertex: pos(2) other(2) side(1) width(1)
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
    for (let k = 0; k < n; k++) {
      const ax = seg[4 * k], ay = seg[4 * k + 1], bx = seg[4 * k + 2], by = seg[4 * k + 3], w = widths[k]
      x0 = Math.min(x0, ax, bx); x1 = Math.max(x1, ax, bx); y0 = Math.min(y0, ay, by); y1 = Math.max(y1, ay, by)
      const vs = [[ax, ay, bx, by, 1], [ax, ay, bx, by, -1], [bx, by, ax, ay, -1], [bx, by, ax, ay, 1]]
      for (let v = 0; v < 4; v++) { const o = (k * 4 + v) * 6; g.set(vs[v], o); g[o + 5] = w }
    }
    this.box = [x0, y0, x1, y1]
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufGeo); gl.bufferData(gl.ARRAY_BUFFER, g, gl.STATIC_DRAW)
    const idx = new Uint32Array(n * 6)
    for (let k = 0; k < n; k++) { const b = 4 * k; idx.set([b, b + 1, b + 2, b, b + 2, b + 3], 6 * k) }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bufIdx); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufVal); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(n * 4), gl.DYNAMIC_DRAW)
  }
  // vals: value at the start and at the end of each segment (Infinity allowed)
  private vbuf?: Float32Array
  setValues(vals: Float64Array | Float32Array) {
    const n = this.nseg, v = this.vbuf && this.vbuf.length === 4 * n ? this.vbuf : (this.vbuf = new Float32Array(4 * n))
    for (let k = 0; k < n; k++) {
      const a = Math.min(vals[2 * k], 1e30), b = Math.min(vals[2 * k + 1], 1e30)
      v[4 * k] = a; v[4 * k + 1] = a; v[4 * k + 2] = b; v[4 * k + 3] = b
    }
    const gl = this.gl; gl.bindBuffer(gl.ARRAY_BUFFER, this.bufVal); gl.bufferSubData(gl.ARRAY_BUFFER, 0, v)
  }
  // world -> screen (CSS pixels), with the same fit the shader uses
  fit() {
    const W = this.canvas.width, H = this.canvas.height, [x0, y0, x1, y1] = this.box
    const s = Math.min((W * (1 - 2 * this.pad)) / (x1 - x0), (H * (1 - 2 * this.pad)) / (y1 - y0))
    return { s, ox: (W - s * (x1 - x0)) / 2 - s * x0, oy: (H - s * (y1 - y0)) / 2 - s * y0, W, H }
  }
  toWorld(cssX: number, cssY: number) {
    const dpr = this.canvas.width / this.canvas.getBoundingClientRect().width, f = this.fit()
    return [(cssX * dpr - f.ox) / f.s, (cssY * dpr - f.oy) / f.s]
  }
  toCss(x: number, y: number) {
    const dpr = this.canvas.width / this.canvas.getBoundingClientRect().width, f = this.fit()
    return [(x * f.s + f.ox) / dpr, (y * f.s + f.oy) / dpr]
  }
  draw(o: { reach: number; vmax: number; front?: number; glow?: number; base: [number, number, number, number] }) {
    const gl = this.gl, p = this.prog, f = this.fit(), dpr = this.canvas.width / Math.max(1, this.canvas.getBoundingClientRect().width)
    gl.viewport(0, 0, f.W, f.H); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT)
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    gl.useProgram(p)
    const at = (name: string) => gl.getAttribLocation(p, name), un = (name: string) => gl.getUniformLocation(p, name)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufGeo)
    const st = 6 * 4
    for (const [name, size, off] of [['a_pos', 2, 0], ['a_other', 2, 8], ['a_side', 1, 16], ['a_w', 1, 20]] as const) {
      const l = at(name); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, st, off)
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufVal); const lv = at('a_val'); gl.enableVertexAttribArray(lv); gl.vertexAttribPointer(lv, 1, gl.FLOAT, false, 4, 0)
    // world -> clip: x' = 2 (s x + ox) / W - 1, y' = 1 - 2 (s y + oy) / H
    gl.uniform2f(un('u_scale'), (2 * f.s) / f.W, (-2 * f.s) / f.H)
    gl.uniform2f(un('u_offset'), (2 * f.ox) / f.W - 1, 1 - (2 * f.oy) / f.H)
    gl.uniform2f(un('u_px'), (2 / f.W) * dpr, (2 / f.H) * dpr)
    gl.uniform1f(un('u_reach'), o.reach); gl.uniform1f(un('u_vmax'), o.vmax); gl.uniform1f(un('u_front'), o.front ?? 0)
    gl.uniform1f(un('u_glow'), o.glow ?? 0.55); gl.uniform4f(un('u_base'), ...o.base)
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.lut); gl.uniform1i(un('u_lut'), 0)
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bufIdx)
    for (const L of this.layers) {
      gl.uniform1f(un('u_alpha'), L.alpha)
      gl.drawElements(gl.TRIANGLES, L.count * 6, gl.UNSIGNED_INT, L.start * 6 * 4)
    }
  }
  destroy() { this.ro.disconnect() }
}
