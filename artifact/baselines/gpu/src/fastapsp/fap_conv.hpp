// Timed "conversion" for the canonical scope (FAIRNESS.md): our CSR (graph in host memory) ->
// FastAPSP's own CSR (adj_size, row_offset, col_val, weight).  FastAPSP builds its CSR inside its file
// reader (readMatFile: linked-list insertion of both directions of every file line, then a walk of
// the lists).  Here we run exactly that build from the in-memory edge list that mtx2fap writes
// (i < j, row-major order of our CSR), so the arrays are identical to FastAPSP's (checked by the
// driver: conv_identical) and only the parsing, which is file reading, is left out.
#pragma once
#include <cmath>
#include <vector>
#include "../common/mtx.hpp"

namespace fap_conv {

struct FapCsr { std::vector<int> adj_size, row_offset, col_val; std::vector<float> weight; };

inline FapCsr convert(const gb::Csr& G) {
  const int n = (int)G.n; const long long m = G.m;
  std::vector<int> h(n + 10, -1), e(m + 10), ne(m + 10);
  std::vector<float> w(m + 10);
  int idx = 0;
  auto add = [&](int a, int b, float c) { e[idx] = b; w[idx] = c; ne[idx] = h[a]; h[a] = idx++; };
  for (int i = 0; i < n; ++i)
    for (int64_t q = G.rowptr[i]; q < G.rowptr[i + 1]; ++q) {
      int j = G.col[q];
      if (j > i) { float c = (float)std::fabs((double)G.w[q]); add(i, j, c); add(j, i, c); }
    }
  FapCsr F; F.adj_size.resize(n); F.row_offset.resize(n + 1); F.col_val.resize(m); F.weight.resize(m);
  F.row_offset[0] = 0; int edges = 0;
  for (int i = 0; i < n; i++) {
    int count = 0;
    for (int j = h[i]; j != -1; j = ne[j]) { F.col_val[edges] = e[j]; F.weight[edges] = w[j]; edges++; count++; }
    F.adj_size[i] = count; F.row_offset[i + 1] = F.row_offset[i] + count;
  }
  return F;
}

template <class GraphT>
inline bool identical(const FapCsr& F, const GraphT& G) {
  return F.adj_size == G.adj_size && F.row_offset == G.row_offset && F.col_val == G.col_val && F.weight == G.weight;
}

}  // namespace fap_conv
