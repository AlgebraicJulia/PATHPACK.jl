// CPU Dijkstra reference for sampled rows (verification of extrapolated runs).
// Integer weights stored as float -> distances are exact in int64.
#pragma once
#include "mtx.hpp"
#include <queue>
#include <vector>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gb {

struct RowCheck { double sum; unsigned long long cnt; };

inline std::vector<RowCheck> dijkstra_rows(const Csr& G, const std::vector<int64_t>& sources) {
  std::vector<RowCheck> out(sources.size());
#pragma omp parallel
  {
    std::vector<int64_t> dist(G.n);
    std::vector<char> done(G.n);
#pragma omp for schedule(dynamic, 1)
    for (int64_t q = 0; q < (int64_t)sources.size(); ++q) {
      const int64_t INF = INT64_MAX;
      std::fill(dist.begin(), dist.end(), INF);
      std::fill(done.begin(), done.end(), 0);
      using P = std::pair<int64_t, int32_t>;
      std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
      int64_t s = sources[q];
      dist[s] = 0; pq.push({0, (int32_t)s});
      while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (done[u]) continue;
        done[u] = 1;
        for (int64_t e = G.rowptr[u]; e < G.rowptr[u + 1]; ++e) {
          int32_t v = G.col[e];
          int64_t nd = d + (int64_t)G.w[e];
          if (nd < dist[v]) { dist[v] = nd; pq.push({nd, v}); }
        }
      }
      double sum = 0; unsigned long long cnt = 0;
      for (int64_t v = 0; v < G.n; ++v) { if (dist[v] == INF) cnt++; else sum += (double)dist[v]; }
      out[q] = {sum, cnt};
    }
  }
  return out;
}

// Deterministic sample of k sources spread over [0,n) (stride + offset), always includes 0.
inline std::vector<int64_t> sample_sources(int64_t n, int64_t k, uint64_t seed = 12345) {
  std::vector<int64_t> s;
  if (k >= n) { for (int64_t i = 0; i < n; ++i) s.push_back(i); return s; }
  uint64_t x = seed;
  for (int64_t q = 0; q < k; ++q) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    int64_t base = (q * n) / k, span = ((q + 1) * n) / k - base;
    s.push_back(base + (int64_t)((x >> 33) % (uint64_t)std::max<int64_t>(span, 1)));
  }
  return s;
}

}  // namespace gb
