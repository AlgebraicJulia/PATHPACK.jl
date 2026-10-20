// The distance ramp of the site (near = blue, far = gold), light and dark, as 256 RGB triples.
const STOPS = {
  light: [[0, [47, 79, 192]], [0.3, [79, 111, 214]], [0.6, [149, 88, 178]], [0.82, [203, 60, 51]], [1, [232, 163, 61]]],
  dark: [[0, [96, 136, 255]], [0.35, [150, 112, 244]], [0.6, [226, 84, 150]], [0.82, [248, 112, 64]], [1, [255, 204, 96]]],
} as const
export function ramp(theme: 'light' | 'dark'): Uint8Array {
  const S = STOPS[theme], L = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    const u = Math.pow(i / 255, 0.75)
    for (let k = 0; k < S.length - 1; k++) {
      const [a, ca] = S[k], [b, cb] = S[k + 1]
      if (u >= a && u <= b) { const t = (u - a) / (b - a); for (let c = 0; c < 3; c++) L[i * 4 + c] = ca[c] * (1 - t) + cb[c] * t; L[i * 4 + 3] = 255; break }
    }
  }
  return L
}
export const isDark = () => typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
