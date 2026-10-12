// PHAST (Delling et al. 2013) on RoutingKit's contraction hierarchy: one-to-all shortest paths as an upward
// Dijkstra search followed by one linear downward sweep in decreasing rank. The k-source variant stores each
// vertex's k labels contiguously (PHAST's multiple-trees layout). Validated against plain Dijkstra.
// Usage: phast_driver graph.gr [nsrc=100] [k=16,64] [threads=1]
#include <routingkit/contraction_hierarchy.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace RoutingKit;
typedef unsigned W;
static const W INF = 0x3fffffffu;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Graph { unsigned n = 0; std::vector<unsigned> tail, head; std::vector<W> w; };
static Graph read_gr(const char* path) {
    Graph g; std::ifstream in(path); std::string line; char c;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == 'p') { std::istringstream s(line); std::string p, sp; unsigned n, m; s >> p >> sp >> n >> m; g.n = n; g.tail.reserve(m); g.head.reserve(m); g.w.reserve(m); }
        else if (line[0] == 'a') { std::istringstream s(line); unsigned u, v; W x; s >> c >> u >> v >> x; g.tail.push_back(u - 1); g.head.push_back(v - 1); g.w.push_back(x); }
    }
    return g;
}
// reference: plain Dijkstra from s on the input graph (CSR by tail)
static std::vector<W> dijkstra(const Graph& g, const std::vector<unsigned>& fo, const std::vector<unsigned>& hd, const std::vector<W>& wt, unsigned s) {
    std::vector<W> d(g.n, INF); d[s] = 0;
    std::priority_queue<std::pair<W, unsigned>, std::vector<std::pair<W, unsigned>>, std::greater<>> q; q.push({0, s});
    while (!q.empty()) { auto [du, u] = q.top(); q.pop(); if (du > d[u]) continue;
        for (unsigned a = fo[u]; a < fo[u + 1]; ++a) { W nd = du + wt[a]; if (nd < d[hd[a]]) { d[hd[a]] = nd; q.push({nd, hd[a]}); } } }
    return d;
}
// PHAST with k sources; dist is n*k, labels of rank x at dist[x*k .. x*k+k-1]; sources given as ranks.
struct Phast {
    const ContractionHierarchy& ch; unsigned n;
    std::vector<std::pair<W, unsigned>> heap; std::vector<unsigned> touched; std::vector<W> up;  // upward search scratch
    Phast(const ContractionHierarchy& ch_) : ch(ch_), n(ch_.node_count()), up(ch_.node_count(), INF) {}
    void run(const unsigned* src, unsigned k, W* dist) {
        std::fill(dist, dist + (size_t)n * k, INF);
        for (unsigned j = 0; j < k; ++j) {                                   // upward searches
            unsigned s = src[j]; up[s] = 0; touched.clear(); touched.push_back(s);
            std::priority_queue<std::pair<W, unsigned>, std::vector<std::pair<W, unsigned>>, std::greater<>> q; q.push({0, s});
            while (!q.empty()) { auto [du, x] = q.top(); q.pop(); if (du > up[x]) continue;
                dist[(size_t)x * k + j] = du;
                for (unsigned a = ch.forward.first_out[x]; a < ch.forward.first_out[x + 1]; ++a) {
                    unsigned y = ch.forward.head[a]; W nd = du + ch.forward.weight[a];
                    if (nd < up[y]) { if (up[y] == INF) touched.push_back(y); up[y] = nd; q.push({nd, y}); } } }
            for (unsigned x : touched) up[x] = INF;
        }
        for (unsigned x = n; x-- > 0;) {                                      // downward sweep, decreasing rank
            W* dx = dist + (size_t)x * k;
            for (unsigned a = ch.backward.first_out[x]; a < ch.backward.first_out[x + 1]; ++a) {
                const W* dy = dist + (size_t)ch.backward.head[a] * k; W w = ch.backward.weight[a];
                for (unsigned j = 0; j < k; ++j) { W c = dy[j] + w; dx[j] = c < dx[j] ? c : dx[j]; }
            }
        }
    }
};
// Level-parallel single-tree sweep (Delling et al. 2013, Sec. 5): vertices grouped so that each reads only vertices
// of earlier levels; the vertices of one level are processed in parallel. As in Delling et al., vertices are renumbered by
// level so each level is contiguous in memory. Upward search stays sequential (small).
struct Levels { std::vector<unsigned> first, order, newid, fo, head; std::vector<W> w; };
static Levels levels(const ContractionHierarchy& ch) {
    unsigned n = ch.node_count(); std::vector<unsigned> lev(n, 0); unsigned L = 0;
    for (unsigned x = n; x-- > 0;) {
        unsigned l = 0;
        for (unsigned a = ch.backward.first_out[x]; a < ch.backward.first_out[x + 1]; ++a) l = std::max(l, lev[ch.backward.head[a]] + 1);
        lev[x] = l; L = std::max(L, l + 1);
    }
    Levels r; r.first.assign(L + 1, 0); r.order.resize(n);
    for (unsigned x = 0; x < n; ++x) r.first[lev[x] + 1]++;
    for (unsigned l = 0; l < L; ++l) r.first[l + 1] += r.first[l];
    std::vector<unsigned> pos(r.first.begin(), r.first.end() - 1);
    for (unsigned x = 0; x < n; ++x) r.order[pos[lev[x]]++] = x;
    r.newid.resize(n); for (unsigned i = 0; i < n; ++i) r.newid[r.order[i]] = i;
    r.fo.assign(n + 1, 0);                                             // downward arcs of vertex order[i], renumbered
    for (unsigned i = 0; i < n; ++i) { unsigned x = r.order[i]; r.fo[i + 1] = r.fo[i] + (ch.backward.first_out[x + 1] - ch.backward.first_out[x]); }
    r.head.resize(r.fo[n]); r.w.resize(r.fo[n]);
    for (unsigned i = 0; i < n; ++i) { unsigned x = r.order[i], p = r.fo[i];
        for (unsigned a = ch.backward.first_out[x]; a < ch.backward.first_out[x + 1]; ++a, ++p) { r.head[p] = r.newid[ch.backward.head[a]]; r.w[p] = ch.backward.weight[a]; } }
    return r;
}
static void run_levels(const ContractionHierarchy& ch, const Levels& lv, Phast& ph, const unsigned* src, unsigned k, W* dist, int nt) {
    unsigned n = ch.node_count(); std::fill(dist, dist + (size_t)n * k, INF);
    for (unsigned j = 0; j < k; ++j) {                                    // upward searches, sequential (small)
        unsigned s0 = src[j]; ph.up[s0] = 0; ph.touched.clear(); ph.touched.push_back(s0);
        std::priority_queue<std::pair<W, unsigned>, std::vector<std::pair<W, unsigned>>, std::greater<>> q; q.push({0, s0});
        while (!q.empty()) { auto [du, x] = q.top(); q.pop(); if (du > ph.up[x]) continue; dist[(size_t)lv.newid[x] * k + j] = du;
            for (unsigned a = ch.forward.first_out[x]; a < ch.forward.first_out[x + 1]; ++a) {
                unsigned y = ch.forward.head[a]; W nd = du + ch.forward.weight[a];
                if (nd < ph.up[y]) { if (ph.up[y] == INF) ph.touched.push_back(y); ph.up[y] = nd; q.push({nd, y}); } } }
        for (unsigned x : ph.touched) ph.up[x] = INF;
    }
    for (size_t l = 0; l + 1 < lv.first.size(); ++l) {                   // downward sweep, level by level
        long lo = lv.first[l], hi = lv.first[l + 1];
        if (k == 1) {                                                     // scalar path for a single tree
            #pragma omp parallel for num_threads(nt) schedule(static) if(nt > 1 && hi - lo > 2048)
            for (long i = lo; i < hi; ++i) {
                W dx = dist[i];
                for (unsigned a = lv.fo[i]; a < lv.fo[i + 1]; ++a) { W c = dist[lv.head[a]] + lv.w[a]; dx = c < dx ? c : dx; }
                dist[i] = dx;
            }
            continue;
        }
        #pragma omp parallel for num_threads(nt) schedule(static) if(nt > 1 && hi - lo > 2048)
        for (long i = lo; i < hi; ++i) {
            W* dx = dist + (size_t)i * k;
            for (unsigned a = lv.fo[i]; a < lv.fo[i + 1]; ++a) {
                const W* dy = dist + (size_t)lv.head[a] * k; W w = lv.w[a];
                for (unsigned j = 0; j < k; ++j) { W c = dy[j] + w; dx[j] = c < dx[j] ? c : dx[j]; }
            }
        }
    }
}
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s graph.gr [nsrc] [threads]\n", argv[0]); return 1; }
    unsigned nsrc = argc > 2 ? atoi(argv[2]) : 100; int nt = argc > 3 ? atoi(argv[3]) : 1;
    Graph g = read_gr(argv[1]); unsigned n = g.n;
    double t0 = now();
    ContractionHierarchy ch = ContractionHierarchy::build(n, g.tail, g.head, g.w);
    double t_ch = now() - t0;
    // CSR of the input graph for the reference Dijkstra
    std::vector<unsigned> fo(n + 1, 0), hd(g.tail.size()); std::vector<W> wt(g.tail.size());
    for (unsigned t : g.tail) fo[t + 1]++; for (unsigned i = 0; i < n; ++i) fo[i + 1] += fo[i];
    { std::vector<unsigned> pos(fo.begin(), fo.end() - 1); for (size_t a = 0; a < g.tail.size(); ++a) { unsigned p = pos[g.tail[a]]++; hd[p] = g.head[a]; wt[p] = g.w[a]; } }
    std::mt19937 rng(42); std::uniform_int_distribution<unsigned> U(0, n - 1);
    std::vector<unsigned> srcs(std::max(nsrc, 64u)); for (auto& s : srcs) s = U(rng);
    Phast ph(ch);
    {   // validation: 3 sources, k = 1 and k = 3
        std::vector<unsigned> r(3); for (int j = 0; j < 3; ++j) r[j] = ch.rank[srcs[j]];
        std::vector<W> D((size_t)n * 3); ph.run(r.data(), 3, D.data());
        for (int j = 0; j < 3; ++j) { auto ref = dijkstra(g, fo, hd, wt, srcs[j]);
            for (unsigned v = 0; v < n; ++v) { W a = ref[v], b = D[(size_t)ch.rank[v] * 3 + j];
                if ((a >= INF ? INF : a) != (b >= INF ? INF : b)) { fprintf(stderr, "MISMATCH source %u vertex %u: %u vs %u\n", srcs[j], v, a, b); return 2; } } }
    }
    auto bench = [&](unsigned k) {   // per-query seconds, median over blocks; blocks run on nt threads (one block per thread)
        unsigned nb = std::max(1u, nsrc / k); std::vector<double> t;
        std::vector<std::vector<W>> buf(nt, std::vector<W>((size_t)n * k)); std::vector<Phast> P; for (int i = 0; i < nt; ++i) P.emplace_back(ch);
        std::vector<unsigned> r(srcs.size()); for (size_t i = 0; i < srcs.size(); ++i) r[i] = ch.rank[srcs[i]];
        P[0].run(r.data(), k, buf[0].data());                                // warm-up
        if (nt == 1) { for (unsigned b = 0; b < nb; ++b) { double s = now(); P[0].run(r.data() + (b * k) % (r.size() - k + 1), k, buf[0].data()); t.push_back((now() - s) / k); }
                       std::sort(t.begin(), t.end()); return t[t.size() / 2]; }
        double s = now();
        #pragma omp parallel for num_threads(nt) schedule(dynamic)
        for (int b = 0; b < (int)(nb * nt); ++b) { int id = omp_get_thread_num(); P[id].run(r.data() + (b * k) % (r.size() - k + 1), k, buf[id].data()); }
        return (now() - s) / (nb * nt * k);                                  // throughput-based per-query time
    };
    double t1 = bench(1), t16 = bench(16), t64 = bench(64);
    Levels lv = levels(ch); Phast phl(ch); std::vector<W> Dl((size_t)n * 3), Ds((size_t)n * 3);
    {   std::vector<unsigned> r(3); for (int j = 0; j < 3; ++j) r[j] = ch.rank[srcs[j]];
        phl.run(r.data(), 3, Ds.data()); run_levels(ch, lv, phl, r.data(), 3, Dl.data(), nt);      // k = 3, nt threads
        for (unsigned x = 0; x < n; ++x) for (int j = 0; j < 3; ++j)
            if (Dl[(size_t)lv.newid[x] * 3 + j] != Ds[(size_t)x * 3 + j]) { fprintf(stderr, "MISMATCH level-ordered sweep\n"); return 2; } }
    auto bench_lv = [&](unsigned k) {   // per-query seconds; nt == 1: sequential; nt > 1: one block per thread (throughput)
        unsigned nb = std::max(1u, nsrc / k); std::vector<double> t;
        std::vector<std::vector<W>> buf(nt, std::vector<W>((size_t)n * k)); std::vector<Phast> P; for (int i = 0; i < nt; ++i) P.emplace_back(ch);
        std::vector<unsigned> r(srcs.size()); for (size_t i = 0; i < srcs.size(); ++i) r[i] = ch.rank[srcs[i]];
        run_levels(ch, lv, P[0], r.data(), k, buf[0].data(), 1);
        if (nt == 1) { for (unsigned b = 0; b < nb; ++b) { double s0 = now(); run_levels(ch, lv, P[0], r.data() + (b * k) % (r.size() - k + 1), k, buf[0].data(), 1); t.push_back((now() - s0) / k); }
                       std::sort(t.begin(), t.end()); return t[t.size() / 2]; }
        double s0 = now();
        #pragma omp parallel for num_threads(nt) schedule(dynamic)
        for (int b = 0; b < (int)(nb * nt); ++b) { int id = omp_get_thread_num(); run_levels(ch, lv, P[id], r.data() + (b * k) % (r.size() - k + 1), k, buf[id].data(), 1); }
        return (now() - s0) / (nb * nt * k);
    };
    double l1 = bench_lv(1), l16 = bench_lv(16), l64 = bench_lv(64);
    std::vector<double> tl; std::vector<W> D1(n);
    for (unsigned b = 0; b < nsrc; ++b) { unsigned r = ch.rank[srcs[b]]; double s0 = now(); run_levels(ch, lv, phl, &r, 1, D1.data(), nt); tl.push_back(now() - s0); }
    std::sort(tl.begin(), tl.end()); double tlat = tl[tl.size() / 2];
    printf("%s n=%u m=%zu threads=%d | CH build %.2f s, arcs up %zu down %zu | rank order per query: k=1 %.3f, k=16 %.3f, k=64 %.3f ms | level order per query: k=1 %.3f, k=16 %.3f, k=64 %.3f ms | one tree on %d threads (level-parallel): %.3f ms, %zu levels | validated\n", argv[1], n, g.tail.size(), nt, t_ch, ch.forward.head.size(), ch.backward.head.size(), 1e3 * t1, 1e3 * t16, 1e3 * t64, 1e3 * l1, 1e3 * l16, 1e3 * l64, nt, 1e3 * tlat, lv.first.size() - 1);
    return 0;
}
