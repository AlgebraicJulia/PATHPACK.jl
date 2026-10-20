// Read a demo data file (site_data/build_site_data.py): uint32 header length, JSON header, 4-byte-aligned arrays.
const CTORS: Record<string, any> = {
  float32: Float32Array, float64: Float64Array, uint8: Uint8Array, uint16: Uint16Array,
  uint32: Uint32Array, int8: Int8Array, int16: Int16Array, int32: Int32Array,
}
export interface Bin { meta: any; a: Record<string, any> }
const cache = new Map<string, Promise<Bin>>()
export function loadBin(url: string): Promise<Bin> {
  if (!cache.has(url)) cache.set(url, (async () => {
    const res = await fetch(url)
    // files are gzip-compressed (.gz); decompress in the browser
    const buf = url.endsWith('.gz') && 'DecompressionStream' in window
      ? await new Response(res.body!.pipeThrough(new (window as any).DecompressionStream('gzip'))).arrayBuffer()
      : await res.arrayBuffer()
    const hl = new Uint32Array(buf, 0, 1)[0]
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)))
    const base = 4 + hl, a: Record<string, any> = {}
    for (const [k, [t, off, n]] of Object.entries<any>(head.arrays)) a[k] = new CTORS[t](buf.slice(base + off, base + off + n * CTORS[t].BYTES_PER_ELEMENT))
    return { meta: head.meta, a }
  })())
  return cache.get(url)!
}

// unpack numpy's packbits (most significant bit first)
export const bit = (b: Uint8Array, k: number) => (b[k >> 3] >> (7 - (k & 7))) & 1
// resolve when the element comes within `margin` of the viewport (load a demo's data only when it is needed)
export function nearView(el: Element, margin = '600px'): Promise<void> {
  return new Promise((ok) => {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); ok() } }, { rootMargin: margin })
    io.observe(el)
  })
}
