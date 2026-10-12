// RoutingKit CCH baseline on DIMACS road networks.
// Parses USA-road-t .gr (travel-time arcs) and USA-road-d .co (coordinates), computes an
// inertial-flow nested-dissection order, builds a CCH, customizes it with the travel times,
// and runs random point-to-point queries. Times the four phases separately.
// Build: clang++ -std=c++17 -O3 -mcpu=apple-m1 -Iinclude cch_driver.cpp lib/libroutingkit.a \
//        -L$(brew --prefix libomp)/lib -lomp -o cch_driver
// Run:   ./cch_driver <file.gr> <file.co> [nqueries=1000] [seed=1]
#include <routingkit/customizable_contraction_hierarchy.h>
#include <routingkit/nested_dissection.h>
#include <routingkit/timer.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

using namespace RoutingKit;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s file.gr file.co [nq] [seed]\n", argv[0]); return 1; }
    int nq = argc > 3 ? std::atoi(argv[3]) : 1000;
    unsigned seed = argc > 4 ? (unsigned)std::atoi(argv[4]) : 1u;

    unsigned n = 0; long m = 0;
    std::vector<unsigned> tail, head, weight;
    char line[256];
    std::FILE* fg = std::fopen(argv[1], "r");
    if (!fg) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    while (std::fgets(line, sizeof(line), fg)) {
        if (line[0] == 'p') { std::sscanf(line, "p sp %u %ld", &n, &m); tail.reserve(m); head.reserve(m); weight.reserve(m); }
        else if (line[0] == 'a') { unsigned u, v, w; std::sscanf(line, "a %u %u %u", &u, &v, &w); tail.push_back(u-1); head.push_back(v-1); weight.push_back(w); }
    }
    std::fclose(fg);

    std::vector<float> lat(n, 0.f), lon(n, 0.f);
    std::FILE* fc = std::fopen(argv[2], "r");
    if (!fc) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    while (std::fgets(line, sizeof(line), fc)) {
        if (line[0] == 'v') { unsigned id; long x, y; std::sscanf(line, "v %u %ld %ld", &id, &x, &y); if (id>=1 && id<=n) { lon[id-1] = x/1e6f; lat[id-1] = y/1e6f; } }
    }
    std::fclose(fc);

    long t0 = get_micro_time();
    std::vector<unsigned> order = compute_nested_node_dissection_order_using_inertial_flow(n, tail, head, lat, lon);
    long t_order = get_micro_time() - t0;

    t0 = get_micro_time();
    CustomizableContractionHierarchy cch(order, tail, head);
    long t_build = get_micro_time() - t0;

    t0 = get_micro_time();
    CustomizableContractionHierarchyMetric metric(cch, weight);
    metric.customize();
    long t_cust = get_micro_time() - t0;

    CustomizableContractionHierarchyQuery query(metric);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<unsigned> pick(0, n-1);
    query.reset().add_source(pick(rng)).add_target(pick(rng)).run();  // warm-up
    t0 = get_micro_time();
    unsigned long long acc = 0;
    for (int i = 0; i < nq; ++i) { acc += query.reset().add_source(pick(rng)).add_target(pick(rng)).run().get_distance(); }
    long t_query = get_micro_time() - t0;

    std::printf("n=%u m=%ld | order_ms=%.1f cch_build_ms=%.1f customize_ms=%.1f | query_us=%.2f (nq=%d, acc=%llu)\n",
                n, m, t_order/1e3, t_build/1e3, t_cust/1e3, (double)t_query/nq, nq, acc);
    return 0;
}
