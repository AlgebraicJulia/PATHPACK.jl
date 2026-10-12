// Shared MatrixMarket reader for the GPU-contender drivers.
// Input: "coordinate real general", 1-based, entry (i,j,w) = edge i->j with weight w.
// Duplicates take the min, self loops are dropped (d(i,i)=0 for positive weights).
// Output: CSR by source row (row i = out-edges of i), 0-based, sorted by column.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <stdexcept>

namespace gb {

inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Csr {
  int64_t n = 0, m = 0;          // vertices, directed edges after dedup/self-loop removal
  int64_t nnz_file = 0;          // entries in the file
  std::vector<int64_t> rowptr;   // n+1
  std::vector<int32_t> col;      // m
  std::vector<float>   w;        // m
};

inline std::string basename_noext(const std::string& p) {
  size_t s = p.find_last_of('/');
  std::string b = (s == std::string::npos) ? p : p.substr(s + 1);
  size_t d = b.rfind(".mtx");
  if (d != std::string::npos) b = b.substr(0, d);
  return b;
}

static inline const char* skip_ws(const char* p, const char* e) {
  while (p < e && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
  return p;
}
static inline const char* parse_i64(const char* p, const char* e, int64_t& v) {
  p = skip_ws(p, e);
  int64_t x = 0; bool neg = false;
  if (p < e && *p == '-') { neg = true; ++p; }
  while (p < e && *p >= '0' && *p <= '9') { x = x * 10 + (*p - '0'); ++p; }
  v = neg ? -x : x;
  return p;
}
static inline const char* parse_f(const char* p, const char* e, float& v) {
  p = skip_ws(p, e);
  // fast path: integer possibly followed by ".0..." ; fall back to strtod otherwise
  const char* q = p; int64_t x = 0; bool neg = false;
  if (q < e && *q == '-') { neg = true; ++q; }
  const char* ds = q;
  while (q < e && *q >= '0' && *q <= '9') { x = x * 10 + (*q - '0'); ++q; }
  if (q < e && (*q == '.' || *q == 'e' || *q == 'E')) {
    char* endp; double d = strtod(p, &endp); v = (float)d; return endp;
  }
  if (q == ds) throw std::runtime_error("bad float");
  v = (float)(neg ? -x : x);
  return q;
}

// Reads the file; returns CSR. t_read receives the seconds spent (parse + CSR build).
inline Csr read_mtx(const std::string& path, double* t_read = nullptr) {
  double t0 = now_s();
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> buf(sz + 1);
  if (fread(buf.data(), 1, sz, f) != (size_t)sz) throw std::runtime_error("short read");
  fclose(f); buf[sz] = 0;
  const char* p = buf.data(); const char* e = p + sz;
  bool symmetric = false;
  // header + comments
  while (p < e && *p == '%') {
    const char* nl = (const char*)memchr(p, '\n', e - p);
    std::string line(p, nl ? nl - p : e - p);
    if (line.rfind("%%MatrixMarket", 0) == 0 && line.find("symmetric") != std::string::npos) symmetric = true;
    p = nl ? nl + 1 : e;
  }
  int64_t nr, nc, nz;
  p = parse_i64(p, e, nr); p = parse_i64(p, e, nc); p = parse_i64(p, e, nz);
  if (nr != nc) throw std::runtime_error("not square");
  Csr G; G.n = nr; G.nnz_file = nz;
  std::vector<int32_t> I, J; std::vector<float> W;
  I.reserve(nz * (symmetric ? 2 : 1)); J.reserve(I.capacity()); W.reserve(I.capacity());
  for (int64_t k = 0; k < nz; ++k) {
    int64_t i, j; float w;
    p = parse_i64(p, e, i); p = parse_i64(p, e, j); p = parse_f(p, e, w);
    --i; --j;
    if (i == j) continue;
    I.push_back((int32_t)i); J.push_back((int32_t)j); W.push_back(w);
    if (symmetric) { I.push_back((int32_t)j); J.push_back((int32_t)i); W.push_back(w); }
  }
  // counting sort by row, then sort each row by col and dedup (min)
  int64_t n = G.n, m0 = (int64_t)I.size();
  std::vector<int64_t> rp(n + 1, 0);
  for (int64_t k = 0; k < m0; ++k) rp[I[k] + 1]++;
  for (int64_t i = 0; i < n; ++i) rp[i + 1] += rp[i];
  std::vector<std::pair<int32_t, float>> tmp(m0);
  { std::vector<int64_t> pos(rp.begin(), rp.end() - 1);
    for (int64_t k = 0; k < m0; ++k) tmp[pos[I[k]]++] = {J[k], W[k]}; }
  G.rowptr.assign(n + 1, 0); G.col.reserve(m0); G.w.reserve(m0);
  for (int64_t i = 0; i < n; ++i) {
    auto b = tmp.begin() + rp[i], en = tmp.begin() + rp[i + 1];
    std::sort(b, en, [](auto& a, auto& c) { return a.first < c.first || (a.first == c.first && a.second < c.second); });
    int32_t last = -1;
    for (auto it = b; it != en; ++it) {
      if (it->first == last) continue;  // first occurrence is the min weight
      G.col.push_back(it->first); G.w.push_back(it->second); last = it->first;
    }
    G.rowptr[i + 1] = (int64_t)G.col.size();
  }
  G.m = (int64_t)G.col.size();
  if (t_read) *t_read = now_s() - t0;
  return G;
}

}  // namespace gb
