// Widest (maximum-capacity) paths with Boost's generic Dijkstra, unmodified: combine = min, compare = greater,
// unreached = -inf, source = +inf. Boost's compressed-sparse-row graph (its fastest representation).
// Usage: bottleneck_boost GRAPH.wel SOURCES [DUMP]   (GRAPH.wel: 0-based "u v capacity" per arc)
// Each source's result is copied into its row of an output matrix allocated and first-touched before the timer, like the
// output of our closure. Prints the total and mean time; with DUMP, writes the values from the first source.
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/dijkstra_shortest_paths_no_color_map.hpp>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <algorithm>
#include <vector>
struct Cap { double w; };
typedef boost::compressed_sparse_row_graph<boost::directedS, boost::no_property, Cap> Graph;
int main(int argc, char **argv) {
    std::ifstream in(argv[1]); std::vector<std::pair<int, int>> E; std::vector<Cap> W; int u, v, n = 0; double w;
    while (in >> u >> v >> w) { E.emplace_back(u, v); W.push_back({w}); n = std::max(n, std::max(u, v) + 1); }
    Graph g(boost::edges_are_unsorted_multi_pass, E.begin(), E.end(), W.begin(), n);
    std::vector<int> src; { std::ifstream s(argv[2]); int x; while (s >> x) src.push_back(x); }
    std::vector<double> dist(n); const double inf = std::numeric_limits<double>::infinity();
    auto run = [&](int s) {
        boost::dijkstra_shortest_paths_no_color_map(g, s,
            boost::weight_map(boost::get(&Cap::w, g))
            .distance_map(boost::make_iterator_property_map(dist.begin(), boost::get(boost::vertex_index, g)))
            .distance_compare(std::greater<double>())
            .distance_combine([](double a, double b) { return std::min(a, b); })
            .distance_inf(-inf).distance_zero(inf));
    };
    std::vector<double> out(src.size() * size_t(n), 0.0);           // the output, one row per source, touched before the timer
    run(src[0]);                                                     // warm-up
    auto t0 = std::chrono::steady_clock::now();
    for (size_t r = 0; r < src.size(); r++) { run(src[r]); std::copy(dist.begin(), dist.end(), out.begin() + r * n); }
    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), t = total / src.size();
    if (argc > 3) { FILE *f = fopen(argv[3], "w"); for (int i = 0; i < n; i++) fprintf(f, "%.17g\n", out[i]); fclose(f); }
    printf("boost widest-path Dijkstra: n=%d arcs=%zu sources=%zu total_s=%.6f mean %.1f us per source\n", n, E.size(), src.size(), total, 1e6 * t);
    return 0;
}
