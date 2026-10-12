// Host version of src/common/rowhash.cuh (same formula, for FastAPSP's host-resident result):
//   uint64 mod 2^64; i = source, j = target, 1-based ORIGINAL labels;
//   v(i,j) = d(i,j) as integer if finite (d < 5e7, FastAPSP's infinity is 1e8) else 0xFFFFFFFF;
//   rowhash(i) = sum_j v(i,j) * (j*0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019).
// Reference: rowhash/<graph>.rowhash (line i = rowhash(i), hex; GB_ROWHASH_DIR after gpu_fingerprint.py).
#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace fap_rh {

inline std::vector<unsigned long long> load_ref(const std::string& graph) {
  std::vector<unsigned long long> ref;
  std::ifstream f("rowhash/" + graph + ".rowhash");
  std::string line;
  while (f && std::getline(f, line)) {
    size_t a = line.find_first_not_of(" \t\r");
    if (a == std::string::npos) continue;
    ref.push_back(std::stoull(line.substr(a), nullptr, 16));
  }
  return ref;
}

// Columns of `row` are original 0-based labels 0..n-1 (FastAPSP keeps original column order).
inline unsigned long long hash_row(const float* row, int64_t n) {
  const unsigned long long C1 = 0x9E3779B97F4A7C15ull, C2 = 0x632BE59BD9B4E019ull;
  unsigned long long acc = 0;
  for (int64_t j = 0; j < n; ++j) {
    float x = row[j];
    unsigned long long v = (x >= 5e7f || x != x) ? 0xFFFFFFFFull : (unsigned long long)(long long)x;
    acc += v * ((unsigned long long)(j + 1) * C1 + C2);
  }
  return acc;
}

// rows [0, nrows) of D (row stride n); row_label[r] = 0-based original source label. Returns #bad rows.
inline long long check_rows(const float* D, int64_t nrows, int64_t n, const int32_t* row_label,
                            const std::vector<unsigned long long>& ref) {
  long long bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic, 64)
  for (int64_t r = 0; r < nrows; ++r) {
    int64_t i = row_label[r];
    unsigned long long h = hash_row(D + r * n, n);
    if (i < 0 || i >= (int64_t)ref.size() || ref[i] != h) bad++;
  }
  return bad;
}

}  // namespace fap_rh
