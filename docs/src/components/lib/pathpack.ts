// PATHPACK.jl's query step in the browser. docs/site_factors.jl factorizes a demo's graph once, F = mlu(MinPlus(), A)
// in Float32, and writes the factors to <name>-lu.bin.gz; this file runs rmul!(B, F) for a single row B, the
// distances from one source, the way PATHPACK runs it: the same supernodal triangular solves, the same loops in the
// same order, in Float32 (Math.fround after every ⊗). Each function names the Julia function it follows.
//
// Every array is kept 1-based, as in Julia: index 0 is unused, so the code below reads like PATHPACK's.
//
// rmul!(B, F)                       src/CPU/abstract_slu.jl     sgetrs!(F, Val(:R), Val(:N), B)
//   sgetrs!(F::ChordalSLU, ...)     src/CPU/chordal_slu.jl      permutation, blocked solve, permutation
//   sgetrs_mt!(... Bptr, Fptr ...)  src/CPU/blocked/sgetrs.jl   strongly connected components, in order
//   sgemx_sparse_st!                src/CPU/sparse/sgemx.jl     arcs into a component from the earlier ones
//   sgetrs_mt!(... fstrt, fstop)    src/CPU/chordal/sgetrs.jl   B ← B U*, then B ← B L*
//   strsx_mt!, strsx_fwd*, strsx_bwd*  src/CPU/chordal/strsx.jl the supernodal solves, front by front
//   strsx_st!, strsx2!              src/CPU/dense/strsx.jl      the dense solve on a large front's diagonal block
//   sgemx2!                         src/CPU/dense/sgemx.jl      the dense update of a large front's separator
import type { Bin } from './bin'

// ===== the semiring: MinPlus over Float32 (src/CPU/semiring/tropical.jl, semiring.jl) =====
const f32 = Math.fround
const INF = Infinity
// vmin(x, y) on x86: ifelse(x < y, x, y)
const vmin = (x: number, y: number) => (x < y ? x : y)
// splus(MinPlus(), a, b, Val(:N)) = vmin(a, b)
const splus = vmin
// smuladd(MinPlus(), a, b, c, Val(:N), Val(:N)) on x86 = splus(s, a + b, c): (a ⊗ b) ⊕ c
const smuladd = (a: number, b: number, c: number) => vmin(f32(a + b), c)
// sprod(MinPlus(), a::AbstractFloat, b::AbstractFloat, Val(:N), Val(:N)): c = a + b, szero if c is NaN
const sprod = (a: number, b: number) => { const c = f32(a + b); return c !== c ? INF : c }
// sstar(s::TropicalSemiring, a): sone if slte(s, a, sone) (a >= 0), else szero(s, T, Val(:C)) = -Inf
const sstar = (a: number) => (a >= 0 ? 0 : -INF)
// isintegral(MinPlus()) is false (docs/site_factors.jl checks it): the U solve applies sstar to its diagonal
const ISINTEGRAL = false
// STRSX_NB (src/CPU/chordal/strsx.jl) and CliqueTrees' THRESHOLD (src/CPU/dense/strsx.jl)
const STRSX_NB = 8, THRESHOLD = 64

export interface Factors {
  n: number; nf: number; nB: number
  cinvp: Int32Array; rperm: Int32Array        // F.cinvp, F.rperm
  Rptr: Int32Array                            // pointers(S.res): the residual of front f is Rptr[f]:Rptr[f + 1] - 1
  Sptr: Int32Array; Stgt: Int32Array          // pointers(S.sep), targets(S.sep): the separator of front f
  Dptr: Int32Array; Lptr: Int32Array          // S.Dptr, S.Lptr: where each front's blocks start
  Bptr: Int32Array; Fptr: Int32Array          // F.S.Bptr, F.S.Fptr: vertices and fronts of each component
  Nptr: Int32Array; Ntgt: Int32Array; Nval: Float32Array   // F.S.N, F.Nval: arcs between components
  UDval: Float32Array; LDval: Float32Array    // F.UDval (upper triangle, diagonal), F.LDval (strict lower triangle)
  ULval: Float32Array; LLval: Float32Array    // F.ULval, F.LLval
  C: Float32Array; work: Float32Array; Mval: Float32Array // the row being solved, permutecols!'s work, DivisionWorkspace's Mval
}

// a 1-based copy: index 0 unused
function one<T extends Int32Array | Float32Array>(a: ArrayLike<number>, ctor: { new (n: number): T }): T {
  const b = new ctor(a.length + 1); for (let i = 0; i < a.length; i++) b[i + 1] = a[i]; return b
}

export function fromBin({ meta, a }: Bin): Factors {
  const n = meta.n, nf = meta.nf
  const Rptr = one(a.Rptr, Int32Array), Sptr = one(a.Sptr, Int32Array)
  // S.Dptr and S.Lptr are the running sums of nn² and nn⋅na (CliqueTrees' ChordalSymbolic; site_factors.jl checks it)
  const Dptr = new Int32Array(nf + 2), Lptr = new Int32Array(nf + 2); Dptr[1] = 1; Lptr[1] = 1
  let maxna = 0
  for (let f = 1; f <= nf; f++) {
    const nn = Rptr[f + 1] - Rptr[f], na = Sptr[f + 1] - Sptr[f]
    Dptr[f + 1] = Dptr[f] + nn * nn; Lptr[f + 1] = Lptr[f] + nn * na; if (na > maxna) maxna = na
  }
  const D = one(a.D, Float32Array)
  return {
    n, nf, nB: meta.nB,
    cinvp: one(a.cinvp, Int32Array), rperm: one(a.rperm, Int32Array),
    Rptr, Sptr, Stgt: one(a.Stgt, Int32Array), Dptr, Lptr,
    Bptr: one(a.Bptr, Int32Array), Fptr: one(a.Fptr, Int32Array),
    Nptr: one(a.Nptr, Int32Array), Ntgt: one(a.Ntgt, Int32Array), Nval: one(a.Nval, Float32Array),
    UDval: D, LDval: D, ULval: one(a.UL, Float32Array), LLval: one(a.LL, Float32Array),
    C: new Float32Array(n + 1), work: new Float32Array(n + 1), Mval: new Float32Array(maxna + 1),
  }
}

// The distances from vertex s (0-based) to every vertex, into out (0-based): row s of A*,
//   B = fill(Inf, 1, n); B[s] = 0; rmul!(B, F)
export function distancesFrom(F: Factors, s: number, out: Float32Array | Float64Array) {
  const b = F.work
  b.fill(INF); b[s + 1] = 0
  rmul(F, b)
  for (let j = 1; j <= F.n; j++) out[j - 1] = b[j]
  return out
}

// rmul!(B, F) for a 1 × n row B, held 1-based in b (src/CPU/abstract_slu.jl), which is
// sgetrs!(F, Val(:R), Val(:N), B) (src/CPU/chordal_slu.jl). With SIDE = :R and TRANS = :N, isforward(:U, :N, :R)
// holds, so invp = F.cinvp and perm = F.rperm.
export function rmul(F: Factors, b: Float32Array) {
  const n = F.n, C = F.C
  // permutecols!(B, work, invp): B[1, invp[j]] = B[1, j] (src/CPU/utils.jl)
  for (let j = 1; j <= n; j++) C[F.cinvp[j]] = b[j]
  sgetrsBlocked(F, C)
  // permutecols!(B, work, perm)
  for (let j = 1; j <= n; j++) b[F.rperm[j]] = C[j]
  return b
}

// sgetrs_mt!(s, side, trans, L, U, Bptr, Fptr, nBptr, Nptr, Ntgt, Nval, B, W::DivisionWorkspace, pool, nt)
// (src/CPU/blocked/sgetrs.jl), branch isforward(:U, TRANS, SIDE), SIDE === :R
function sgetrsBlocked(F: Factors, B: Float32Array) {
  for (let c = 1; c <= F.nB; c++) {
    const fstrt = F.Fptr[c], fstop = F.Fptr[c + 1] - 1
    const jstrt = F.Bptr[c], jstop = F.Bptr[c + 1] - 1
    // sgemx_sparse!(s, Val(:N), trans, B, B, Nptr, Ntgt, Nval, jstrt, jstop, nt), one row: sgemx_sparse_st!
    // (src/CPU/sparse/sgemx.jl), saxpy!(s, tA, tB, Val(:R), Bval[p], view(A, :, k), view(C, :, j))
    for (let j = jstrt; j <= jstop; j++) {
      const pstrt = F.Nptr[j], pstop = F.Nptr[j + 1] - 1
      for (let p = pstrt; p <= pstop; p++) {
        const k = F.Ntgt[p]
        B[j] = smuladd(B[k], F.Nval[p], B[j])
      }
    }
    sgetrsChordal(F, B, fstrt, fstop)
  }
}

// sgetrs_mt!(s, side, trans, L, U, B, W, pool, nt, fstrt, fstop) (src/CPU/chordal/sgetrs.jl): isforward(:L, :N, :R)
// is false, so
//   strsx_mt!(s, side, trans, Val(:N), U, B, W.Mval, pool, nt, fstrt, fstop)
//   strsx_mt!(s, side, trans, Val(:U), L, B, W.Mval, pool, nt, fstrt, fstop)
function sgetrsChordal(F: Factors, B: Float32Array, fstrt: number, fstop: number) {
  // strsx_mt! (src/CPU/chordal/strsx.jl) with U: isforward(:U, :N, :R), fronts in order
  for (let f = fstrt; f <= fstop; f++) {
    const nn = F.Rptr[f + 1] - F.Rptr[f]
    if (nn === 1) strsxFwd1(F, B, f)
    else strsxFwd(F, B, f)
  }
  // strsx_mt! with L: not isforward(:L, :N, :R), fronts in reverse
  for (let f = fstop; f >= fstrt; f--) {
    const nn = F.Rptr[f + 1] - F.Rptr[f]
    if (nn === 1) strsxBwd1(F, B, f)
    else strsxBwd(F, B, f)
  }
}

// ===== U: forward, uplo = :U, diag = :N =====

// strsx_fwd_1!(s, C, Dval, Lval, Dptr, Lptr, res, sep, nrhs, f, trans, Val(:U), Val(:N), Val(:R), nothing)
function strsxFwd1(F: Factors, C: Float32Array, f: number) {
  const Dval = F.UDval, Lval = F.ULval
  const na = F.Sptr[f + 1] - F.Sptr[f], fsep0 = F.Sptr[f] - 1   // fsep[i] = Stgt[fsep0 + i]
  const Dp = F.Dptr[f], Lp = F.Lptr[f], Rp = F.Rptr[f]
  const d11 = Dval[Dp]                                          // l₂₁[i] = Lval[Lp + i - 1]
  //   c₁ ← d₁₁* c₁
  if (!ISINTEGRAL) {
    const v = sstar(d11)
    C[Rp] = sprod(C[Rp], v)
  }
  if (na > 0) {
    //   M₂ ← l₂₁ c₁       C₂ ← C₂ + M₂
    // strsx_fwd_upd_1! → strsx_fwd_upd_1_cols!: saxpy_kern!(..., pC + fsep[i], pr, l₂₁[i], nrhs)
    for (let i = 1; i <= na; i++) {
      const c = F.Stgt[fsep0 + i]
      C[c] = smuladd(C[Rp], Lval[Lp + i - 1], C[c])
    }
  }
}

// strsx_fwd!(s, C, Mval, Dval, Lval, Dptr, Lptr, res, sep, pool, nt, nrhs, f, trans, Val(:U), Val(:N), Val(:R), nothing)
function strsxFwd(F: Factors, C: Float32Array, f: number) {
  const Dval = F.UDval, Lval = F.ULval
  const nn = F.Rptr[f + 1] - F.Rptr[f], na = F.Sptr[f + 1] - F.Sptr[f]
  const Dp = F.Dptr[f], Lp = F.Lptr[f]
  // D₁₁[i, j] = Dval[Dp + (i - 1) + (j - 1) * nn], and for UPLO === :U, L₂₁ is nn × na:
  // L₂₁[i, j] = Lval[Lp + (i - 1) + (j - 1) * nn]
  if (nn < STRSX_NB) {
    // strsx_fwd_upd_small!(s, C::AbstractMatrix{T <: BScalarTypes}, ..., trans::N_OR_R, ::Val{:U}, ::Val{:R})
    const Rp = F.Rptr[f], fsep0 = F.Sptr[f] - 1
    for (let j = 1; j <= nn; j++) {
      const c = Rp + j - 1
      for (let i = 1; i <= j - 1; i++) {
        const d = Rp + i - 1
        C[c] = smuladd(C[d], Dval[Dp + (i - 1) + (j - 1) * nn], C[c])
      }
      if (!ISINTEGRAL) {
        const v = sstar(Dval[Dp + (j - 1) + (j - 1) * nn])
        C[c] = sprod(C[c], v)
      }
    }
    if (na > 0) {
      for (let j = 1; j <= na; j++) {
        const c = F.Stgt[fsep0 + j]
        for (let i = 1; i <= nn; i++) {
          const d = Rp + i - 1
          C[c] = smuladd(C[d], Lval[Lp + (i - 1) + (j - 1) * nn], C[c])
        }
      }
    }
  } else {
    // strsx_fwd_upd!(s, C, Mval, D₁₁, L₂₁, res, sep, na, nrhs, pool, nt, f, trans, uplo, diag, side, nothing)
    const Rp = F.Rptr[f], fsep0 = F.Sptr[f] - 1
    //   C₁ ← C₁ U₁₁*      (C₁ = view(C, 1:nrhs, fres): C[Rp:Rp + nn - 1])
    strsxSt(Dval, Dp, nn, true, C, Rp, nn)
    if (na > 0) {
      const M = F.Mval
      //   M₂ ← C₁ L₂₁      szerorec!(s, M₂, trans), then sgemx_mt!(s, Val(:N), trans, M₂, C₁, L₂₁, pool, nt)
      for (let k = 1; k <= na; k++) M[k] = INF
      sgemx(M, 1, C, Rp, Lval, Lp, nn, nn, na)
      //   C₂ ← C₂ + M₂     sscatteradd!(s, trans, C, M₂, fsep, Val(:R)) (src/CPU/utils.jl)
      for (let j = 1; j <= na; j++) {
        const c = F.Stgt[fsep0 + j]
        C[c] = splus(C[c], M[j])
      }
    }
  }
}

// ===== L: backward, uplo = :L, diag = :U =====

// strsx_bwd_1!(s, C, Dval, Lval, Dptr, Lptr, res, sep, nrhs, f, trans, Val(:L), Val(:U), Val(:R))
function strsxBwd1(F: Factors, C: Float32Array, f: number) {
  const Lval = F.LLval
  const na = F.Sptr[f + 1] - F.Sptr[f], fsep0 = F.Sptr[f] - 1
  const Lp = F.Lptr[f], Rp = F.Rptr[f]                          // u₁₂[i] = Lval[Lp + i - 1]
  if (na > 0) {
    //   M₂ ← C₂       c₁ ← u₁₂ M₂ + c₁
    // strsx_bwd_upd_1!, SIDE === :R, nrhs <= STRSX_1_NB
    let v = C[Rp]
    for (let i = 1; i <= na; i++) v = smuladd(C[F.Stgt[fsep0 + i]], Lval[Lp + i - 1], v)
    C[Rp] = v
  }
  //   c₁ ← d₁₁* c₁: not with DIAG === :U
}

// strsx_bwd!(s, C, Mval, Dval, Lval, Dptr, Lptr, res, sep, pool, nt, nrhs, f, trans, Val(:L), Val(:U), Val(:R))
function strsxBwd(F: Factors, C: Float32Array, f: number) {
  const Dval = F.LDval, Lval = F.LLval
  const nn = F.Rptr[f + 1] - F.Rptr[f], na = F.Sptr[f + 1] - F.Sptr[f]
  const Dp = F.Dptr[f], Lp = F.Lptr[f]
  // D₁₁[i, j] = Dval[Dp + (i - 1) + (j - 1) * nn], and for UPLO === :L, U₁₂ is na × nn:
  // U₁₂[i, j] = Lval[Lp + (i - 1) + (j - 1) * na]
  if (nn < STRSX_NB) {
    // strsx_bwd_upd_small!(s, C::AbstractMatrix{T <: BScalarTypes}, ..., trans::N_OR_R, ::Val{:L}, ::Val{:R})
    const Rp = F.Rptr[f], fsep0 = F.Sptr[f] - 1
    for (let j = 1; j <= nn; j++) {
      const c = Rp + j - 1
      for (let i = 1; i <= na; i++) {
        const d = F.Stgt[fsep0 + i]
        C[c] = smuladd(C[d], Lval[Lp + (i - 1) + (j - 1) * na], C[c])
      }
    }
    for (let j = nn; j >= 1; j--) {
      const c = Rp + j - 1
      for (let i = j + 1; i <= nn; i++) {
        const d = Rp + i - 1
        C[c] = smuladd(C[d], Dval[Dp + (i - 1) + (j - 1) * nn], C[c])
      }
      // diag === Val(:U): no star
    }
  } else {
    // strsx_bwd_upd!(s, C, Mval, D₁₁, U₁₂, res, sep, na, nrhs, pool, nt, f, trans, uplo, diag, side)
    const Rp = F.Rptr[f], fsep0 = F.Sptr[f] - 1
    if (na > 0) {
      const M = F.Mval
      //   M₂ ← C₂          copygatherrec!(M₂, C, fsep, side)
      for (let i = 1; i <= na; i++) M[i] = C[F.Stgt[fsep0 + i]]
      //   C₁ ← U₁₂ M₂ + C₁  sgemx_mt!(s, Val(:N), trans, C₁, M₂, U₁₂, pool, nt)
      sgemx(C, Rp, M, 1, Lval, Lp, na, na, nn)
    }
    //   C₁ ← C₁ L₁₁*
    strsxSt(Dval, Dp, nn, false, C, Rp, nn)
  }
}

// ===== the dense kernels, for one row =====

// strsx_st!(s, Val(:R), trans, uplo, diag, A, B, AP, BP, CP) (src/CPU/dense/strsx.jl) for a 1 × n row B, with
// uplo = :U, diag = :N (upper = true, forward) or uplo = :L, diag = :U (backward). A[i, j] = A0[a + (i - 1) + (j - 1) * lda],
// B[j] = Bv[b + j - 1].
function strsxSt(A0: Float32Array, a: number, lda: number, upper: boolean, Bv: Float32Array, b: number, n: number) {
  if (n <= THRESHOLD) {
    strsx2(A0, a, lda, upper, Bv, b, n)
  } else {
    // m = prevpow(2, n) >> 1
    let m = 1; while (m * 2 <= n) m *= 2; m >>= 1
    const a11 = a, a22 = a + m + m * lda
    if (upper) {
      // isforward(:U, :N, :R): A₂₁ = view(A, 1:m, m + 1:n)
      const a21 = a + m * lda
      strsxSt(A0, a11, lda, true, Bv, b, m)
      //   sgemx_st!(s, Val(:N), trans, B₂, B₁, A₂₁, AP, BP, CP)
      sgemx(Bv, b + m, Bv, b, A0, a21, lda, m, n - m)
      strsxSt(A0, a22, lda, true, Bv, b + m, n - m)
    } else {
      // A₂₁ = view(A, m + 1:n, 1:m)
      const a21 = a + m
      strsxSt(A0, a22, lda, false, Bv, b + m, n - m)
      //   sgemx_st!(s, Val(:N), trans, B₁, B₂, A₂₁, AP, BP, CP)
      sgemx(Bv, b, Bv, b + m, A0, a21, lda, n - m, m)
      strsxSt(A0, a11, lda, false, Bv, b, m)
    }
  }
}

// strsx2!(s, ::Val{:R}, trans::N_OR_R, ::Val{:U} or ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix{T <: BScalarTypes}, B)
// with m = 1: saxpy_kern! with one row applies its x's one after the other.
function strsx2(A0: Float32Array, a: number, lda: number, upper: boolean, Bv: Float32Array, b: number, n: number) {
  if (upper) {
    // ::Val{:U}, DIAG === :N
    for (let j = 1; j <= n; j++) {
      const bj = b + j - 1
      for (let k = 1; k <= j - 1; k++) Bv[bj] = smuladd(Bv[b + k - 1], A0[a + (k - 1) + (j - 1) * lda], Bv[bj])
      if (!ISINTEGRAL) {
        const sAjj = sstar(A0[a + (j - 1) + (j - 1) * lda])
        Bv[bj] = sprod(Bv[bj], sAjj)
      }
    }
  } else {
    // ::Val{:L}, DIAG === :U
    for (let j = n; j >= 1; j--) {
      const bj = b + j - 1
      for (let k = j + 1; k <= n; k++) Bv[bj] = smuladd(Bv[b + k - 1], A0[a + (k - 1) + (j - 1) * lda], Bv[bj])
    }
  }
}

// sgemx2!(s, Val(:N), Val(:N), C, A, B, ...) (src/CPU/dense/sgemx.jl) for a 1-row C and A:
//   C[1, k] = smuladd(s, A[1, j], B[j, k], C[1, k]) for k in 1:nk, j in 1:nj
// with C[k] = Cv[c + k - 1], A[j] = Av[ap + j - 1], B[j, k] = Bv[bp + (j - 1) + (k - 1) * ldb]. This is the loop of
// sgemx2!'s generic method; for Float32 PATHPACK runs a packed SIMD kernel (sgemx2_impl!, and sgemx_st!'s blocking)
// that takes the same ⊕ = min of the same Float32 sums C[1, k] and A[1, j] ⊗ B[j, k] in another order. min is exact,
// so the result is the same bit for bit.
function sgemx(Cv: Float32Array, c: number, Av: Float32Array, ap: number, Bv: Float32Array, bp: number, ldb: number, nj: number, nk: number) {
  for (let k = 1; k <= nk; k++) {
    let ck = Cv[c + k - 1]
    for (let j = 1; j <= nj; j++) ck = smuladd(Av[ap + j - 1], Bv[bp + (j - 1) + (k - 1) * ldb], ck)
    Cv[c + k - 1] = ck
  }
}
