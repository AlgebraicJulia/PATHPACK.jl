// One-line JSON result emitter. Every driver prints exactly one "RESULT {...}" line per run;
// the job script adds process wall time / status and the merge script checks checksums.
#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <iomanip>

namespace gb {

struct Result {
  std::string code, graph, gpu, status = "ok", note;
  long long n = 0, nnz = 0;
  double gpu_compute = -1;   // graph already on device -> distances finished on device (s)
  double inproc = -1;        // host graph in memory -> distances on device (s), comparable to ours "plain"
  double e2e_inproc = -1;    // whole process from main() (s), excluding nothing
  std::vector<std::pair<std::string, double>> phases;
  double cs_sum = 0; unsigned long long cs_cnt = 0; bool have_cs = false;
  bool extrapolated = false;
  long long sources_run = -1;          // for sampled SSSP runs
  long long rows_checked = -1, rows_bad = -1;  // CPU-Dijkstra row verification
  std::vector<std::pair<std::string, std::string>> extra;  // raw json fragments

  void phase(const std::string& k, double v) { phases.push_back({k, v}); }
  void print() const {
    std::ostringstream o; o << std::setprecision(10);
    o << "RESULT {\"code\":\"" << code << "\",\"graph\":\"" << graph << "\",\"gpu\":\"" << gpu
      << "\",\"n\":" << n << ",\"nnz\":" << nnz << ",\"status\":\"" << status << "\""
      << ",\"gpu_compute\":" << gpu_compute << ",\"inproc\":" << inproc << ",\"e2e_inproc\":" << e2e_inproc
      << ",\"phases\":{";
    for (size_t i = 0; i < phases.size(); ++i) o << (i ? "," : "") << "\"" << phases[i].first << "\":" << phases[i].second;
    o << "}";
    if (have_cs) o << ",\"checksum\":[" << std::setprecision(17) << cs_sum << "," << cs_cnt << "]" << std::setprecision(10);
    o << ",\"extrapolated\":" << (extrapolated ? "true" : "false");
    if (sources_run >= 0) o << ",\"sources_run\":" << sources_run;
    if (rows_checked >= 0) o << ",\"rows_checked\":" << rows_checked << ",\"rows_bad\":" << rows_bad;
    for (auto& kv : extra) o << ",\"" << kv.first << "\":" << kv.second;
    if (!note.empty()) o << ",\"note\":\"" << note << "\"";
    o << "}";
    printf("%s\n", o.str().c_str()); fflush(stdout);
  }
};

}  // namespace gb
