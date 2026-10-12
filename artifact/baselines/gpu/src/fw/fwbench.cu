// Dense blocked Floyd-Warshall contenders on one GPU.
//
//   fwbench <graph.mtx> --algo {tuned|cuasr|lundsmith} [--b 128|256] [--budget 120] [--gpu NAME]
//
// algo=cuasr     : 3-phase blocked FW (Venkataraman et al.) in the GEMM formulation used by
//                  Sao et al. HPDC'21 / cuASR: phase 1 = closure of the b x b diagonal block,
//                  phase 2 = two in-place panel SRGEMMs, phase 3 = one in-place n x n x b SRGEMM.
//                  Phases 2/3 use cuASR's pre-tuned min-plus SIMT SRGEMM (hpcgarage/cuASR,
//                  CUTLASS 2.x, 128x128x8 threadblock, 64x32 warp tile, 2 stages).
// algo=tuned     : same 3-phase structure, our own register-tiled min-plus GEMM for phases 2/3
//                  (128x128 CTA tile, 8x8 per thread, k-pair interleaved smem, register prefetch,
//                  FADD2+FMNMX3 = 1 instr/update on sm_100, FADD+FMNMX on sm_120).  This is the
//                  "best dense FW we could write" upper bound, not a public code.
// algo=lundsmith : Lund & Smith "A Multi-Stage CUDA Kernel for Floyd-Warshall" (2010) as
//                  implemented in github.com/lppcom/CUDA_Blocked_Floyd-Warshall (f9958f4), kernels
//                  copied verbatim except that the predecessor matrix is dropped (distance only,
//                  like every other code here); int32 distances, INF = 0x3F3F3F3F, repo default
//                  config BLOCK_WIDTH=16, THREAD_WIDTH=2 (32x32 blocks).  Only other change: 64-bit
//                  cell indices (the repo's int indices overflow for n > 46340).
//
// The dense matrix is n_pad x n_pad (n_pad = n rounded up to the FW block b), padding = +inf
// (isolated vertices, no effect on real distances).  Timing (CUDA events / host clock):
//   ctx          : cudaFree(0) + kernel warm-up on a tiny problem (lazy module loading)
//   read         : MTX parse + CSR build (gb::read_mtx)
//   h2d          : cudaMalloc of the dense matrix + CSR upload
//   init         : fill +inf / diag 0 + scatter edges (device)
//   fw           : the FW rounds
//   gpu_compute  = init + fw            (graph (CSR) on device -> distances on device)
//   inproc       = h2d + init + fw      (host CSR -> distances on device; comparable to ours "plain")
// If the FW would exceed --budget seconds (estimated from the first rounds), only enough rounds
// to measure the per-round time are run and fw is extrapolated (rounds all cost n_pad^2 b).
#include "../common/mtx.hpp"
#include "../common/checksum.cuh"
#include "../common/result.hpp"
#include "../common/rowhash.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>

#ifdef WITH_CUASR
#include "cuasr/functional.h"
#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#endif

// ------------------------------------------------------------------------------------------
// common kernels
// ------------------------------------------------------------------------------------------
template <typename T>
__global__ void init_dense(T* D, int64_t ld, int64_t npad, int64_t n, T inf) {
  int64_t total = npad * ld;
  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
    int64_t i = idx / ld, j = idx - i * ld;
    D[idx] = (i == j && i < n) ? (T)0 : inf;
  }
}
template <typename T>
__global__ void scatter_edges(T* D, int64_t ld, int64_t n, const int64_t* rp, const int32_t* col, const float* w) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.y + threadIdx.y; i < n; i += (int64_t)gridDim.x * blockDim.y)
    for (int64_t e = rp[i] + threadIdx.x; e < rp[i + 1]; e += blockDim.x) D[i * ld + col[e]] = (T)w[e];
}

// Phase 1: closure of a bs x bs (bs <= 128) diagonal block on one SM, 1024 threads, register
// resident: thread (tx = lane, ty = warp) owns rows ty*4+a, cols tx+32c.  Iteration k: the owner
// warp of row k and the owner lane (per warp) of column k publish them to a double-buffered smem
// row/column (iteration k does not change row/column k since d(k,k) >= 0), one barrier, then every
// thread relaxes its 16 entries.  k = 32c + k1 + a with c, a compile-time (no dynamic indexing).
// Entries outside bs are +inf (padded vertices), so iterating all 128 k is harmless.
// Two builds of the same kernel: capped at 32 registers (diag_closure128, default) so that, with
// lookahead, the phase-1 CTA can start next to one resident bulk-GEMM CTA; and uncapped
// (diag_closure128_nocap, --p1cap 0; it then needs a nearly empty SM).
__device__ __forceinline__ void diag_closure_body(float* D, int64_t ld, int64_t k0, int bs) {
  __shared__ float rowb[2][128], colb[2][128];
  const int tid = threadIdx.x, tx = tid & 31, ty = tid >> 5;
  float r[4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      int i = ty * 4 + a, j = tx + 32 * c;
      r[a][c] = (i < bs && j < bs) ? D[(k0 + i) * ld + k0 + j] : INFINITY;
    }
#pragma unroll
  for (int c = 0; c < 4; ++c) {
#pragma unroll 1
    for (int k1 = 0; k1 < 32; k1 += 4) {
      const bool rowowner = (ty == ((32 * c + k1) >> 2));
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        const int q = a & 1;
        if (rowowner) {
#pragma unroll
          for (int cc = 0; cc < 4; ++cc) rowb[q][tx + 32 * cc] = r[a][cc];
        }
        if (tx == k1 + a) {
#pragma unroll
          for (int aa = 0; aa < 4; ++aa) colb[q][ty * 4 + aa] = r[aa][c];
        }
        __syncthreads();
        float rk[4], ck[4];
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) rk[cc] = rowb[q][tx + 32 * cc];
#pragma unroll
        for (int aa = 0; aa < 4; ++aa) ck[aa] = colb[q][ty * 4 + aa];
#pragma unroll
        for (int aa = 0; aa < 4; ++aa)
#pragma unroll
          for (int cc = 0; cc < 4; ++cc) r[aa][cc] = fminf(r[aa][cc], ck[aa] + rk[cc]);
      }
    }
  }
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      int i = ty * 4 + a, j = tx + 32 * c;
      if (i < bs && j < bs) D[(k0 + i) * ld + k0 + j] = r[a][c];
    }
}
__global__ void __maxnreg__(32) diag_closure128(float* D, int64_t ld, int64_t k0, int bs) { diag_closure_body(D, ld, k0, bs); }
__global__ void __launch_bounds__(1024) diag_closure128_nocap(float* D, int64_t ld, int64_t k0, int bs) { diag_closure_body(D, ld, k0, bs); }
static int g_p1cap = 1;
static void phase1_launch(float* D, int64_t ld, int64_t k0, int bs, cudaStream_t st) {
  if (g_p1cap) diag_closure128<<<1, 1024, 0, st>>>(D, ld, k0, bs); else diag_closure128_nocap<<<1, 1024, 0, st>>>(D, ld, k0, bs);
}

// ------------------------------------------------------------------------------------------
// tuned min-plus GEMM:  C[MxN] = min(C, A[MxK] (+) B[KxN]), row-major, M,N multiples of 128,
// K multiple of 8 (all guaranteed by padding n to a multiple of b and b in {128,256}).
// In-place use (A, B inside C) is safe for FW: the values A/B could be overwritten with are
// identical (closed panels), see README.
// ------------------------------------------------------------------------------------------
#define TBM 128
__device__ __forceinline__ float min3f(float a, float b, float c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)
  float d;
  asm("min.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
#else
  return fminf(fminf(a, b), c);
#endif
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)
  union { float2 f; unsigned long long u; } ua, ub, ud;
  ua.f = a; ub.f = b;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(ud.u) : "l"(ua.u), "l"(ub.u));
  return ud.f;
#else
  return make_float2(a.x + b.x, a.y + b.y);
#endif
}

// smem layout (k-pair interleaved): As[p][m][2] = (A[m][2p], A[m][2p+1]),  Bs[p][n][2] = (B[2p][n], B[2p+1][n])
// CTA tile 128 x WN (WN = 128: 256 threads, WN = 64: 128 threads); thread (tx, ty), TX = WN/8 threads
// along N, owns rows ty*4+(i&3)+64*(i>>2) and cols tx*2+(j&1)+2*TX*(j>>1) (8 x 8 per thread).
struct Prob { int M, N, K, swap; const float* A; int64_t lda; const float* B; int64_t ldb; float* C; int64_t ldc; int64_t skip_lo = -1, skip_hi = -1; };
template <int BK, int MINB, int WN>
__global__ void __launch_bounds__(2 * WN, MINB)
minplus_gemm_tuned(const Prob p0, const Prob p1) {
  constexpr int TX = WN / 8, NT = 16 * TX;
  constexpr int NA = (TBM * BK / 4) / NT;        // float4 of A per thread per k-tile
  constexpr int NB = (BK / 2) * (WN / 2) / NT;   // (row pair, column pair) items of B per thread
  const Prob& P = blockIdx.z ? p1 : p0;
  const int bx = P.swap ? blockIdx.y : blockIdx.x, by = P.swap ? blockIdx.x : blockIdx.y;
  if ((bx + 1) * WN > P.N || (by + 1) * TBM > P.M) return;
  const int64_t m0 = (int64_t)by * TBM, n0 = (int64_t)bx * WN;
  if ((n0 >= P.skip_lo && n0 < P.skip_hi) || (m0 >= P.skip_lo && m0 < P.skip_hi)) return;  // lookahead: done elsewhere
  const int K = P.K; const float* __restrict__ A = P.A; const float* __restrict__ B = P.B; float* C = P.C;
  const int64_t lda = P.lda, ldb = P.ldb, ldc = P.ldc;
  __shared__ __align__(16) float As[2][BK / 2][TBM][2];
  __shared__ __align__(16) float Bs[2][BK / 2][WN][2];
  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  int a_row[NA], a_k[NA], b_q[NB], b_c[NB];
#pragma unroll
  for (int u = 0; u < NA; ++u) { int it = tid + u * NT; a_row[u] = it / (BK / 4); a_k[u] = (it % (BK / 4)) * 4; }
#pragma unroll
  for (int u = 0; u < NB; ++u) { int it = tid + u * NT; b_q[u] = it / (WN / 2); b_c[u] = (it % (WN / 2)) * 2; }

  float acc[8][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int64_t r = m0 + ty * 4 + (i & 3) + (i >> 2) * 64;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float2 v = *reinterpret_cast<const float2*>(C + r * ldc + n0 + tx * 2 + 2 * TX * c);
      acc[i][2 * c] = v.x; acc[i][2 * c + 1] = v.y;
    }
  }
  float4 ra[NA]; float2 rb0[NB], rb1[NB];
  auto load = [&](int k0) {
#pragma unroll
    for (int u = 0; u < NA; ++u) ra[u] = *reinterpret_cast<const float4*>(A + (m0 + a_row[u]) * lda + k0 + a_k[u]);
#pragma unroll
    for (int u = 0; u < NB; ++u) {
      rb0[u] = *reinterpret_cast<const float2*>(B + (int64_t)(k0 + 2 * b_q[u]) * ldb + n0 + b_c[u]);
      rb1[u] = *reinterpret_cast<const float2*>(B + (int64_t)(k0 + 2 * b_q[u] + 1) * ldb + n0 + b_c[u]);
    }
  };
  auto store = [&](int buf) {
#pragma unroll
    for (int u = 0; u < NA; ++u) {
      *reinterpret_cast<float2*>(&As[buf][a_k[u] / 2][a_row[u]][0]) = make_float2(ra[u].x, ra[u].y);
      *reinterpret_cast<float2*>(&As[buf][a_k[u] / 2 + 1][a_row[u]][0]) = make_float2(ra[u].z, ra[u].w);
    }
#pragma unroll
    for (int u = 0; u < NB; ++u)
      *reinterpret_cast<float4*>(&Bs[buf][b_q[u]][b_c[u]][0]) = make_float4(rb0[u].x, rb1[u].x, rb0[u].y, rb1[u].y);
  };
  load(0);
  store(0);
  __syncthreads();
  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    bool more = k0 + BK < K;
    if (more) load(k0 + BK);
#pragma unroll
    for (int p = 0; p < BK / 2; ++p) {
      float2 a[8], b[8];
      {
        float4 x0 = *reinterpret_cast<const float4*>(&As[buf][p][ty * 4][0]);
        float4 x1 = *reinterpret_cast<const float4*>(&As[buf][p][ty * 4 + 2][0]);
        float4 x2 = *reinterpret_cast<const float4*>(&As[buf][p][ty * 4 + 64][0]);
        float4 x3 = *reinterpret_cast<const float4*>(&As[buf][p][ty * 4 + 66][0]);
        a[0] = make_float2(x0.x, x0.y); a[1] = make_float2(x0.z, x0.w);
        a[2] = make_float2(x1.x, x1.y); a[3] = make_float2(x1.z, x1.w);
        a[4] = make_float2(x2.x, x2.y); a[5] = make_float2(x2.z, x2.w);
        a[6] = make_float2(x3.x, x3.y); a[7] = make_float2(x3.z, x3.w);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          float4 y = *reinterpret_cast<const float4*>(&Bs[buf][p][tx * 2 + 2 * TX * c][0]);
          b[2 * c] = make_float2(y.x, y.y); b[2 * c + 1] = make_float2(y.z, y.w);
        }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float2 s = add2(a[i], b[j]);
          acc[i][j] = min3f(acc[i][j], s.x, s.y);
        }
    }
    if (more) {
      store(buf ^ 1);
      __syncthreads();
      buf ^= 1;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int64_t r = m0 + ty * 4 + (i & 3) + (i >> 2) * 64;
#pragma unroll
    for (int c = 0; c < 4; ++c)
      *reinterpret_cast<float2*>(C + r * ldc + n0 + tx * 2 + 2 * TX * c) = make_float2(acc[i][2 * c], acc[i][2 * c + 1]);
  }
}

static int g_tuned_variant = 0;  // 0: BK8/minB2, 1: BK8/minB1, 2: BK16/minB2, 3: BK16/minB1 (128x128 CTA tile)
                                 // 4: BK8/minB2, 5: BK8/minB4, 6: BK16/minB2 (128x64 CTA tile, 128 threads)
static int tuned_wn() { return g_tuned_variant >= 4 ? 64 : 128; }
static void launch_tuned(const Prob& p0, const Prob& p1, dim3 grid, cudaStream_t st) {
  switch (g_tuned_variant) {
    case 0: minplus_gemm_tuned<8, 2, 128><<<grid, 256, 0, st>>>(p0, p1); break;
    case 1: minplus_gemm_tuned<8, 1, 128><<<grid, 256, 0, st>>>(p0, p1); break;
    case 2: minplus_gemm_tuned<16, 2, 128><<<grid, 256, 0, st>>>(p0, p1); break;
    case 3: minplus_gemm_tuned<16, 1, 128><<<grid, 256, 0, st>>>(p0, p1); break;
    case 4: minplus_gemm_tuned<8, 2, 64><<<grid, 128, 0, st>>>(p0, p1); break;
    case 5: minplus_gemm_tuned<8, 4, 64><<<grid, 128, 0, st>>>(p0, p1); break;
    case 6: minplus_gemm_tuned<16, 2, 64><<<grid, 128, 0, st>>>(p0, p1); break;
  }
}
static void gemm_tuned(int M, int N, int K, const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc, cudaStream_t st) {
  Prob p{M, N, K, 0, A, lda, B, ldb, C, ldc};
  launch_tuned(p, p, dim3(N / tuned_wn(), M / TBM, 1), st);
}
// two independent problems in one launch (phase 2: row panel M=b,N=n and column panel M=n,N=b)
static void gemm2_tuned(int M0, int N0, int K0, const float* A0, const float* B0, float* C0,
                        int M1, int N1, int K1, const float* A1, const float* B1, float* C1, int64_t ld, cudaStream_t st) {
  Prob p0{M0, N0, K0, 0, A0, ld, B0, ld, C0, ld}, p1{M1, N1, K1, 1, A1, ld, B1, ld, C1, ld};
  // p1 is "swapped": its m-tiles run along grid x so both fit one grid
  const int wn = tuned_wn();
  unsigned gx = std::max(N0 / wn, M1 / TBM), gy = std::max(M0 / TBM, N1 / wn);
  launch_tuned(p0, p1, dim3(gx, gy, 2), st);
}

#ifdef WITH_CUASR
// cuASR's pre-tuned min-plus SRGEMM (DefaultSemiRingConfiguration<min, plus>, Sm50 SIMT tag as
// in its examples/benchmarks), all operands row-major, D = min(C, A (+) B) with D = C (in place).
using CuasrMinPlus = cuasr::gemm::device::Srgemm<
    cuasr::minimum<float>, cuasr::plus<float>, float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor, float>;
static void gemm_cuasr(int M, int N, int K, const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc, cudaStream_t st) {
  float alpha = cuasr::plus<float>::Identity, beta = cuasr::plus<float>::Identity;  // do_epilogue_min
  CuasrMinPlus::Arguments args({M, N, K}, {A, (int)lda}, {B, (int)ldb}, {C, (int)ldc}, {C, (int)ldc}, {alpha, beta});
  CuasrMinPlus op;
  cutlass::Status s = op(args, nullptr, st);
  if (s != cutlass::Status::kSuccess) { printf("STATUS cuasr_error %d\n", (int)s); fflush(stdout); exit(4); }
}
#endif

// ------------------------------------------------------------------------------------------
// GEMM-formulated blocked FW on the npad x npad square at (o,o) of D
// ------------------------------------------------------------------------------------------
typedef void (*GemmFn)(int, int, int, const float*, int64_t, const float*, int64_t, float*, int64_t, cudaStream_t);

static void fw_square(float* D, int64_t ld, int64_t o, int64_t npad, int b, GemmFn gemm, cudaStream_t st);
// phases 1 and 2 of round k (block rows/cols [k0, k0+b)) on the npad square at (o,o)
static void phase12(float* D, int64_t ld, int64_t o, int64_t npad, int64_t k0, int b, GemmFn gemm, cudaStream_t st) {
  float* Dk = D + (o + k0) * ld + (o + k0);       // diagonal block
  float* Row = D + (o + k0) * ld + o;             // b x npad row panel
  float* Col = D + o * ld + (o + k0);             // npad x b column panel
  if (b <= 128) phase1_launch(D, ld, o + k0, b, st);
  else fw_square(D, ld, o + k0, b, 128, gemm, st);
  // phase 2 (in place; includes the diagonal block, which is unchanged)
  if (gemm == gemm_tuned) {
    gemm2_tuned(b, (int)npad, b, Dk, Row, Row, (int)npad, b, b, Col, Dk, Col, ld, st);
  } else {
    gemm(b, (int)npad, b, Dk, ld, Row, ld, Row, ld, st);
    gemm((int)npad, b, b, Col, ld, Dk, ld, Col, ld, st);
  }
}
// phase 3 of round k on the whole square (the panels are unchanged by it), optionally skipping
// row/column block [s0, s0+b) (lookahead: that cross is updated separately by panel3)
static void bulk3(float* D, int64_t ld, int64_t o, int64_t npad, int64_t k0, int b, int64_t s0, GemmFn gemm, cudaStream_t st) {
  float* Row = D + (o + k0) * ld + o; float* Col = D + o * ld + (o + k0); float* Sq = D + o * ld + o;
  if (s0 < 0) { gemm((int)npad, (int)npad, b, Col, ld, Row, ld, Sq, ld, st); return; }
  if (gemm == gemm_tuned) {
    Prob p{(int)npad, (int)npad, b, 0, Col, ld, Row, ld, Sq, ld, s0, s0 + b};
    launch_tuned(p, p, dim3(npad / tuned_wn(), npad / TBM, 1), st);
    return;
  }
  int64_t e = s0 + b;  // 4 rectangles around the cross
  if (s0 > 0) gemm((int)s0, (int)s0, b, Col, ld, Row, ld, Sq, ld, st);
  if (s0 > 0 && e < npad) gemm((int)s0, (int)(npad - e), b, Col, ld, Row + e, ld, Sq + e, ld, st);
  if (s0 > 0 && e < npad) gemm((int)(npad - e), (int)s0, b, Col + e * ld, ld, Row, ld, Sq + e * ld, ld, st);
  if (e < npad) gemm((int)(npad - e), (int)(npad - e), b, Col + e * ld, ld, Row + e, ld, Sq + e * ld + e, ld, st);
}
// phase-3 update of round k restricted to row/column block j (the lookahead panel)
static void panel3(float* D, int64_t ld, int64_t o, int64_t npad, int64_t k0, int64_t j0, int b, GemmFn gemm, cudaStream_t st) {
  float* Sq = D + o * ld + o;
  float* Rowk = Sq + k0 * ld; float* Colk = Sq + k0;
  // row panel j: C = Sq[j0.., :] (b x npad) = min(C, Sq[j0.., k0..] (b x b) (+) Rowk)
  // col panel j: C = Sq[:, j0..] (npad x b) = min(C, Colk (+) Sq[k0.., j0..] (b x b))
  if (gemm == gemm_tuned) {
    gemm2_tuned(b, (int)npad, b, Sq + j0 * ld + k0, Rowk, Sq + j0 * ld, (int)npad, b, b, Colk, Sq + k0 * ld + j0, Sq + j0, ld, st);
  } else {
    gemm(b, (int)npad, b, Sq + j0 * ld + k0, ld, Rowk, ld, Sq + j0 * ld, ld, st);
    gemm((int)npad, b, b, Colk, ld, Sq + k0 * ld + j0, ld, Sq + j0, ld, st);
  }
}
static void fw_round(float* D, int64_t ld, int64_t o, int64_t npad, int64_t k0, int b, GemmFn gemm, cudaStream_t st) {
  phase12(D, ld, o, npad, k0, b, gemm, st);
  bulk3(D, ld, o, npad, k0, b, -1, gemm, st);
}
static void fw_square(float* D, int64_t ld, int64_t o, int64_t npad, int b, GemmFn gemm, cudaStream_t st) {
  for (int64_t k0 = 0; k0 < npad; k0 += b) fw_round(D, ld, o, npad, k0, b, gemm, st);
}
// Rounds [kbeg, kend) with one-round lookahead (as in right-looking LU): stream sB (high priority)
// runs panel3_k(k+1), phase1_{k+1}, phase2_{k+1} while stream sA runs the bulk phase 3 of round k
// minus the k+1 cross.  Requires rounds < kbeg complete in sA's order.  Ends joined into sA.
struct Lookahead { cudaStream_t sB; std::vector<cudaEvent_t> evA, evB; cudaEvent_t ev0; };
static void fw_lookahead(float* D, int64_t ld, int64_t npad, int b, GemmFn gemm, int kbeg, int kend, cudaStream_t sA, Lookahead& L) {
  int nb = (int)(npad / b);
  cudaEventRecord(L.ev0, sA); cudaStreamWaitEvent(L.sB, L.ev0, 0);
  phase12(D, ld, 0, npad, (int64_t)kbeg * b, b, gemm, L.sB);
  cudaEventRecord(L.evB[kbeg], L.sB);
  for (int k = kbeg; k < kend; ++k) {
    bool la = k + 1 < nb;
    cudaStreamWaitEvent(sA, L.evB[k], 0);
    bulk3(D, ld, 0, npad, (int64_t)k * b, b, la ? (int64_t)(k + 1) * b : -1, gemm, sA);
    cudaEventRecord(L.evA[k], sA);
    if (la) {
      if (k > kbeg) cudaStreamWaitEvent(L.sB, L.evA[k - 1], 0);
      panel3(D, ld, 0, npad, (int64_t)k * b, (int64_t)(k + 1) * b, b, gemm, L.sB);
      phase12(D, ld, 0, npad, (int64_t)(k + 1) * b, b, gemm, L.sB);
      cudaEventRecord(L.evB[k + 1], L.sB);
    }
  }
  if (kend < nb) cudaStreamWaitEvent(sA, L.evB[kend], 0);
}

// ------------------------------------------------------------------------------------------
// Lund & Smith multi-stage kernels (lppcom/CUDA_Blocked_Floyd-Warshall, predecessor removed)
// ------------------------------------------------------------------------------------------
#define LS_INF 1061109567  // 0x3F3F3F3F
#define LS_THREAD_WIDTH 2
#define LS_BLOCK_WIDTH 16
template <int BLOCK_SIZE> __global__ void ls_phase_one(const unsigned int block, const unsigned int n, const size_t pitch, int* const d) {
  int newPath;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int v1 = BLOCK_SIZE * block + ty, v2 = BLOCK_SIZE * block + tx;
  const size_t cell = (size_t)v1 * pitch + v2;
  __shared__ int primary_d[BLOCK_SIZE][BLOCK_SIZE];
  if (v1 < n && v2 < n) primary_d[ty][tx] = d[cell]; else primary_d[ty][tx] = LS_INF;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    newPath = primary_d[ty][i] + primary_d[i][tx];
    __syncthreads();
    if (newPath < primary_d[ty][tx]) primary_d[ty][tx] = newPath;
    __syncthreads();
  }
  if (v1 < n && v2 < n) d[cell] = primary_d[ty][tx];
}
template <int BLOCK_SIZE> __global__ void ls_phase_two(const unsigned int block, const unsigned int n, const size_t pitch, int* const d) {
  if (blockIdx.x == block) return;
  int newPath;
  int tx = threadIdx.x, ty = threadIdx.y;
  int v1 = BLOCK_SIZE * block + ty, v2 = BLOCK_SIZE * block + tx;
  __shared__ int primary_d[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int current_d[BLOCK_SIZE][BLOCK_SIZE];
  const size_t cell_primary = (size_t)v1 * pitch + v2;
  if (v1 < n && v2 < n) primary_d[ty][tx] = d[cell_primary]; else primary_d[ty][tx] = LS_INF;
  if (blockIdx.y == 0) { v1 = BLOCK_SIZE * block + ty; v2 = BLOCK_SIZE * blockIdx.x + tx; }
  else { v1 = BLOCK_SIZE * blockIdx.x + ty; v2 = BLOCK_SIZE * block + tx; }
  const size_t cell_current = (size_t)v1 * pitch + v2;
  if (v1 < n && v2 < n) current_d[ty][tx] = d[cell_current]; else current_d[ty][tx] = LS_INF;
  __syncthreads();
  if (blockIdx.y == 0) {
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      newPath = primary_d[ty][i] + current_d[i][tx];
      __syncthreads();
      if (newPath < current_d[ty][tx]) current_d[ty][tx] = newPath;
      __syncthreads();
    }
  } else {
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      newPath = current_d[ty][i] + primary_d[i][tx];
      __syncthreads();
      if (newPath < current_d[ty][tx]) current_d[ty][tx] = newPath;
      __syncthreads();
    }
  }
  if (v1 < n && v2 < n) d[cell_current] = current_d[ty][tx];
}
template <int BLOCK_SIZE, int THREAD_SIZE> __global__ void ls_phase_three(unsigned int block, const unsigned int n, const size_t pitch, int* const d) {
  if (blockIdx.x == block || blockIdx.y == block) return;
  int newPath, path;
  const int tx = threadIdx.x * THREAD_SIZE, ty = threadIdx.y * THREAD_SIZE;
  const int v1 = blockDim.y * blockIdx.y * THREAD_SIZE + ty;
  const int v2 = blockDim.x * blockIdx.x * THREAD_SIZE + tx;
  int idx, idy;
  __shared__ int primaryRow_d[BLOCK_SIZE * THREAD_SIZE][BLOCK_SIZE * THREAD_SIZE];
  __shared__ int primaryCol_d[BLOCK_SIZE * THREAD_SIZE][BLOCK_SIZE * THREAD_SIZE];
  int v1Row = BLOCK_SIZE * block * THREAD_SIZE + ty;
  int v2Col = BLOCK_SIZE * block * THREAD_SIZE + tx;
#pragma unroll
  for (int i = 0; i < THREAD_SIZE; ++i)
#pragma unroll
    for (int j = 0; j < THREAD_SIZE; ++j) {
      idx = tx + j; idy = ty + i;
      if (v1Row + i < n && v2 + j < n) primaryRow_d[idy][idx] = d[(size_t)(v1Row + i) * pitch + v2 + j];
      else primaryRow_d[idy][idx] = LS_INF;
      if (v1 + i < n && v2Col + j < n) primaryCol_d[idy][idx] = d[(size_t)(v1 + i) * pitch + v2Col + j];
      else primaryCol_d[idy][idx] = LS_INF;
    }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < THREAD_SIZE; ++i)
#pragma unroll
    for (int j = 0; j < THREAD_SIZE; ++j) {
      if (v1 + i < n && v2 + j < n) {
        size_t cell = (size_t)(v1 + i) * pitch + v2 + j;
        path = d[cell];
        idy = ty + i; idx = tx + j;
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE * THREAD_SIZE; ++k) {
          newPath = primaryCol_d[idy][k] + primaryRow_d[k][idx];
          if (path > newPath) path = newPath;
        }
        d[cell] = path;
      }
    }
}
static void ls_round(int* d, int64_t ld, int64_t n, int block, cudaStream_t st) {
  const int V = LS_BLOCK_WIDTH * LS_THREAD_WIDTH;
  dim3 gP1(1, 1, 1), gP2((n - 1) / V + 1, 2, 1), gP3((n - 1) / V + 1, (n - 1) / V + 1, 1);
  dim3 bP1(V, V, 1), bP2(V, V, 1), bP3(LS_BLOCK_WIDTH, LS_BLOCK_WIDTH, 1);
  ls_phase_one<V><<<gP1, bP1, 0, st>>>(block, n, ld, d);
  ls_phase_two<V><<<gP2, bP2, 0, st>>>(block, n, ld, d);
  ls_phase_three<LS_BLOCK_WIDTH, LS_THREAD_WIDTH><<<gP3, bP3, 0, st>>>(block, n, ld, d);
}

// ------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  double T0 = gb::now_s();
  if (argc < 2) { fprintf(stderr, "usage: fwbench graph.mtx --algo tuned|cuasr|lundsmith [--b 128] [--budget 120]\n"); return 1; }
  std::string path = argv[1], algo = "tuned", gpuname = "";
  int b = 128; double budget = 120.0; int force_rounds = -1; int64_t gemmbench_n = 0; int lookahead = 1;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--algo")) algo = argv[++i];
    else if (!strcmp(argv[i], "--b")) b = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--budget")) budget = atof(argv[++i]);
    else if (!strcmp(argv[i], "--rounds")) force_rounds = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--gpu")) gpuname = argv[++i];
    else if (!strcmp(argv[i], "--kv")) g_tuned_variant = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--lookahead")) lookahead = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--p1cap")) g_p1cap = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--gemmbench")) gemmbench_n = atoll(argv[++i]);
  }
  gb::Result R;
  R.code = "fw_" + algo + (algo == "lundsmith" ? std::string("") : "_b" + std::to_string(b));
  R.graph = gb::basename_noext(path);
  GemmFn gemm = nullptr;
  if (algo == "tuned") gemm = gemm_tuned;
#ifdef WITH_CUASR
  else if (algo == "cuasr") gemm = gemm_cuasr;
#endif
  else if (algo != "lundsmith") { fprintf(stderr, "unknown/unbuilt algo %s\n", algo.c_str()); return 1; }
  if (algo != "lundsmith" && (b % 128 != 0 || b > 256)) { fprintf(stderr, "b must be 128 or 256\n"); return 1; }

  // context + warm-up (lazy module loading) on a tiny problem
  double t = gb::now_s();
  GB_CUDA(cudaFree(0));
  cudaDeviceProp prop; GB_CUDA(cudaGetDeviceProperties(&prop, 0));
  R.gpu = gpuname.empty() ? prop.name : gpuname;
  cudaStream_t st; GB_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Lookahead L;
  { int lo, hi; GB_CUDA(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    GB_CUDA(cudaStreamCreateWithPriority(&L.sB, cudaStreamNonBlocking, hi));
    GB_CUDA(cudaEventCreateWithFlags(&L.ev0, cudaEventDisableTiming)); }
  auto ensure_events = [&](int nbr) {
    while ((int)L.evA.size() < nbr + 1) {
      cudaEvent_t a, c; cudaEventCreateWithFlags(&a, cudaEventDisableTiming); cudaEventCreateWithFlags(&c, cudaEventDisableTiming);
      L.evA.push_back(a); L.evB.push_back(c);
    }
  };
  {
    int64_t wn = 256;
    if (algo == "lundsmith") {
      int* w; GB_CUDA(cudaMalloc(&w, wn * wn * sizeof(int)));
      init_dense<int><<<64, 256, 0, st>>>(w, wn, wn, wn, LS_INF);
      for (int k = 0; k < (int)(wn / 32); ++k) ls_round(w, wn, wn, k, st);
      GB_CUDA(cudaStreamSynchronize(st)); cudaFree(w);
    } else {
      float* w; GB_CUDA(cudaMalloc(&w, wn * wn * sizeof(float)));
      init_dense<float><<<64, 256, 0, st>>>(w, wn, wn, wn, INFINITY);
      fw_square(w, wn, 0, wn, b, gemm, st);
      ensure_events((int)(wn / b));
      fw_lookahead(w, wn, wn, b, gemm, 0, (int)(wn / b), st, L);
      GB_CUDA(cudaStreamSynchronize(st)); cudaFree(w);
    }
    { gb::DevChecksum c; c.add<float>(nullptr, 0, 0, 0, 1e30f); }
    GB_CUDA(cudaGetLastError());
  }
  double t_ctx = gb::now_s() - t;
  if (gemmbench_n > 0) {  // time the phase-3 GEMM (n x n x b, in place) and phase-2 GEMMs alone
    int64_t gn = gemmbench_n; float* w; GB_CUDA(cudaMalloc(&w, gn * gn * sizeof(float)));
    init_dense<float><<<4096, 256, 0, st>>>(w, gn, gn, gn, 1.0f);
    cudaEvent_t a0, a1; cudaEventCreate(&a0); cudaEventCreate(&a1);
    for (int rep = 0; rep < 3; ++rep) {
      cudaEventRecord(a0, st);
      for (int it = 0; it < 10; ++it) gemm((int)gn, (int)gn, b, w, gn, w, gn, w, gn, st);
      cudaEventRecord(a1, st); cudaEventSynchronize(a1);
      float ms; cudaEventElapsedTime(&ms, a0, a1);
      printf("GEMMBENCH %s kv=%d n=%lld b=%d phase3 %.3f ms  %.2f Tupd/s\n", algo.c_str(), g_tuned_variant, (long long)gn, b, ms / 10, (double)gn * gn * b / (ms / 10 / 1e3) / 1e12);
      cudaEventRecord(a0, st);
      for (int it = 0; it < 10; ++it) {
        if (gemm == gemm_tuned) gemm2_tuned(b, (int)gn, b, w, w, w, (int)gn, b, b, w, w, w, gn, st);
        else { gemm(b, (int)gn, b, w, gn, w, gn, w, gn, st); gemm((int)gn, b, b, w, gn, w, gn, w, gn, st); }
      }
      cudaEventRecord(a1, st); cudaEventSynchronize(a1);
      cudaEventElapsedTime(&ms, a0, a1);
      printf("GEMMBENCH %s kv=%d n=%lld b=%d phase2(both) %.3f ms\n", algo.c_str(), g_tuned_variant, (long long)gn, b, ms / 10);
      cudaEventRecord(a0, st);
      for (int it = 0; it < 10; ++it) diag_closure128<<<1, 1024, 0, st>>>(w, gn, 0, 128);
      cudaEventRecord(a1, st); cudaEventSynchronize(a1);
      cudaEventElapsedTime(&ms, a0, a1);
      printf("GEMMBENCH phase1(128) %.3f ms\n", ms / 10);
    }
    GB_CUDA(cudaGetLastError());
    return 0;
  }

  double t_read; gb::Csr G = gb::read_mtx(path, &t_read);
  int64_t n = G.n;
  R.n = n; R.nnz = G.nnz_file;
  int bpad = (algo == "lundsmith") ? 32 : b;
  int64_t npad = (n + bpad - 1) / bpad * bpad;
  // Lund-Smith works on n x n with bounds checks; its repo allocates with cudaMallocPitch (512-byte
  // aligned rows), so use the same pitch (128 ints)
  int64_t ld = (algo == "lundsmith") ? (n + 127) / 128 * 128 : npad;
  int64_t rowsalloc = (algo == "lundsmith") ? n : npad;
  int nb = (int)(npad / bpad);
  size_t esz = (algo == "lundsmith") ? sizeof(int) : sizeof(float);

  // ---- inproc starts: allocation + upload
  double ti0 = gb::now_s();
  void* D = nullptr; int64_t *drp; int32_t* dcol; float* dw;
  cudaError_t ae = cudaMalloc(&D, (size_t)rowsalloc * ld * esz);
  if (ae != cudaSuccess) {
    R.status = "oom"; R.note = "dense matrix allocation failed"; R.phase("ctx", t_ctx); R.phase("read", t_read);
    R.e2e_inproc = gb::now_s() - T0; R.print(); return 0;
  }
  GB_CUDA(cudaMalloc(&drp, (n + 1) * sizeof(int64_t)));
  GB_CUDA(cudaMalloc(&dcol, std::max<int64_t>(G.m, 1) * sizeof(int32_t)));
  GB_CUDA(cudaMalloc(&dw, std::max<int64_t>(G.m, 1) * sizeof(float)));
  GB_CUDA(cudaMemcpyAsync(drp, G.rowptr.data(), (n + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, st));
  GB_CUDA(cudaMemcpyAsync(dcol, G.col.data(), G.m * sizeof(int32_t), cudaMemcpyHostToDevice, st));
  GB_CUDA(cudaMemcpyAsync(dw, G.w.data(), G.m * sizeof(float), cudaMemcpyHostToDevice, st));
  GB_CUDA(cudaStreamSynchronize(st));
  double t_h2d = gb::now_s() - ti0;

  cudaEvent_t e0, e1, e2, e3; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
  GB_CUDA(cudaEventRecord(e0, st));
  int blocks = (int)std::min<int64_t>((rowsalloc * ld + 255) / 256, 65536);
  if (algo == "lundsmith") {
    init_dense<int><<<blocks, 256, 0, st>>>((int*)D, ld, n, n, LS_INF);
    scatter_edges<int><<<4096, dim3(32, 8), 0, st>>>((int*)D, ld, n, drp, dcol, dw);
  } else {
    init_dense<float><<<blocks, 256, 0, st>>>((float*)D, ld, npad, n, INFINITY);
    scatter_edges<float><<<4096, dim3(32, 8), 0, st>>>((float*)D, ld, n, drp, dcol, dw);
  }
  GB_CUDA(cudaEventRecord(e1, st));
  bool use_la = lookahead && algo != "lundsmith";
  if (use_la) ensure_events(nb);
  auto round = [&](int k) {
    if (algo == "lundsmith") ls_round((int*)D, ld, n, k, st);
    else fw_round((float*)D, ld, 0, npad, (int64_t)k * b, b, gemm, st);
  };
  auto rounds = [&](int kb, int ke) {  // rounds [kb, ke), after which everything is joined into st
    if (kb >= ke) return;
    if (use_la) fw_lookahead((float*)D, ld, npad, b, gemm, kb, ke, st, L);
    else for (int k = kb; k < ke; ++k) round(k);
  };
  // probe: first round (or forced count), then decide full vs extrapolated
  int probe = std::min(nb, force_rounds > 0 ? force_rounds : 1);
  for (int k = 0; k < probe; ++k) round(k);
  GB_CUDA(cudaEventRecord(e2, st));
  GB_CUDA(cudaEventSynchronize(e2));
  float ms_init, ms_probe; cudaEventElapsedTime(&ms_init, e0, e1); cudaEventElapsedTime(&ms_probe, e1, e2);
  double est_fw = ms_probe / 1e3 / probe * nb;
  int done = probe;
  bool extrap = false;
  if (force_rounds > 0 && force_rounds < nb) extrap = true;
  else if (est_fw > budget) {
    // run more rounds until >= 5 s of measurement (or all), then extrapolate
    extrap = true;
    int target = std::min(nb, std::max(probe + 1, (int)ceil(5.0 / (ms_probe / 1e3 / probe))));
    rounds(done, target);
    done = target;
    if (done == nb) extrap = false;
  } else {
    rounds(done, nb);
    done = nb;
  }
  GB_CUDA(cudaEventRecord(e3, st));
  GB_CUDA(cudaEventSynchronize(e3));
  GB_CUDA(cudaGetLastError());
  float ms_rest; cudaEventElapsedTime(&ms_rest, e2, e3);
  double t_fw_meas = (ms_probe + ms_rest) / 1e3;
  double t_fw = extrap ? t_fw_meas / done * nb : t_fw_meas;
  double t_inproc_meas = gb::now_s() - ti0;

  // checksum over the n x n part
  double tc = gb::now_s();
  gb::DevChecksum cs;
  if (!extrap) {
    if (algo == "lundsmith") cs.add<int>((int*)D, ld, n, n, LS_INF, st);
    else cs.add<float>((float*)D, ld, n, n, 1e30f, st);
    gb::Checksum c = cs.get(st);
    R.have_cs = true; R.cs_sum = c.sum; R.cs_cnt = c.cnt;
  }
  double t_cs = gb::now_s() - tc;
  // position-sensitive per-row hashes (D is in original labels: row = source, column = target)
  double th = gb::now_s();
  if (!extrap) {
    gb::RowHasher H(R.graph);
    if (algo == "lundsmith") H.hash_rows<int>((int*)D, ld, n, n, LS_INF, nullptr, st);
    else H.hash_rows<float>((float*)D, ld, n, n, 1e30f, nullptr, st);
    gb::RowHashResult hr = H.compare();
    R.extra.push_back({"rows_hashed", std::to_string(hr.rows_hashed)});
    R.extra.push_back({"rowhash_bad", hr.have_ref ? std::to_string(hr.rows_bad) : std::string("null")});
    R.extra.push_back({"rowhash_ok", hr.have_ref ? (hr.rows_bad == 0 ? "true" : "false") : "null"});
  }
  R.phase("rowhash", gb::now_s() - th);

  R.gpu_compute = ms_init / 1e3 + t_fw;
  R.inproc = extrap ? t_h2d + ms_init / 1e3 + t_fw : t_inproc_meas;
  R.extrapolated = extrap;
  R.phase("ctx_warmup", t_ctx); R.phase("read", t_read); R.phase("alloc_h2d", t_h2d);
  R.phase("init_dense", ms_init / 1e3); R.phase("fw", t_fw); R.phase("checksum", t_cs);
  R.extra.push_back({"npad", std::to_string(npad)});
  R.extra.push_back({"lookahead", use_la ? "1" : "0"});
  R.extra.push_back({"kv", std::to_string(g_tuned_variant)});
  R.extra.push_back({"p1cap", std::to_string(g_p1cap)});
  R.extra.push_back({"rounds_total", std::to_string(nb)});
  R.extra.push_back({"rounds_run", std::to_string(done)});
  R.extra.push_back({"fw_measured_s", std::to_string(t_fw_meas)});
  double upd = (algo == "lundsmith") ? (double)n * n * n : (double)npad * npad * npad;
  R.extra.push_back({"rate_Tupd_s", std::to_string(upd / t_fw / 1e12)});
  R.e2e_inproc = gb::now_s() - T0;
  R.print();
  cudaFree(D); cudaFree(drp); cudaFree(dcol); cudaFree(dw);
  return 0;
}
