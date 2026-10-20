<script setup lang="ts">
// Head-to-heads, measured: same answers checked. Bars are seconds on a log scale (or linear, on request); PATHPACK.jl
// is the first bar of each group. Bars grow when the chart scrolls into view.
import { ref, onMounted, onBeforeUnmount } from 'vue'
const GROUPS = [
  { title: 'All 15.9 billion distances of the US road network', where: 'one NVIDIA B200 each',
    rows: [['PATHPACK.jl', 0.1389], ['ROME', 0.7503], ['cuGraph', 18.05], ['Fast-APSP', 105.54]] },
  { title: 'Closeness of every street in Paris (11,840 junctions)', where: 'one 32-core AMD EPYC node; igraph and NetworkX single-threaded',
    rows: [['PATHPACK.jl', 0.2824], ['graph-tool', 0.9518], ['NetworKit', 2.4903], ['igraph', 19.861], ['NetworkX', 289.75]] },
  { title: 'Isomap on 10,000 points: the shortest-path step', where: 'one 32-core AMD EPYC node',
    rows: [['PATHPACK.jl', 0.2064], ['NetworKit', 1.2725], ['SciPy, 16 processes', 1.6758], ['scikit-learn', 21.777]] },
  { title: 'Exact personalized PageRank (GDC) on a 47k-node mesh', where: '16 CPU cores, float64',
    rows: [['PATHPACK.jl', 4.58], ['PyTorch dense inverse', 545.43]] },
  { title: 'Every pair of a 32k-vertex cortex', where: 'one 32-core AMD EPYC node',
    rows: [['PATHPACK.jl', 2.64], ['NetworKit', 17.2], ['neuromaps, 16 workers', 24.1]] },
]
const log = ref(true), shown = ref(false), root = ref<HTMLElement>()
const fmt = (s: number) => (s < 1 ? (s * 1000 < 10 ? (s * 1000).toFixed(1) : Math.round(s * 1000)) + ' ms' : s < 100 ? s.toFixed(s < 10 ? 2 : 1) + ' s' : Math.round(s) + ' s')
const frac = (s: number, mx: number) => (log.value ? Math.max(0.02, Math.log10(s / 0.05) / Math.log10(mx / 0.05)) : Math.max(0.004, s / mx))
let io: IntersectionObserver | null = null
onMounted(() => { io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { shown.value = true; io?.disconnect() } }, { threshold: 0.15 }); io.observe(root.value!) })
onBeforeUnmount(() => io?.disconnect())
</script>

<template>
  <div class="pp-bench" ref="root">
    <div class="pp-bench-head">
      <div class="pp-seg"><button :class="{ on: log }" @click="log = true">log scale</button><button :class="{ on: !log }" @click="log = false">linear</button></div>
    </div>
    <div v-for="G in GROUPS" :key="G.title" class="pp-bench-group">
      <h4>{{ G.title }} <span>{{ G.where }}</span></h4>
      <div v-for="([who, s], i) in G.rows" :key="who" class="pp-bench-row" :class="{ us: i === 0 }">
        <span class="pp-bench-who">{{ who }}</span>
        <span class="pp-bench-track"><span class="pp-bench-bar" :style="{ width: (shown ? 100 * frac(s as number, Math.max(...G.rows.map(r => r[1] as number))) : 0) + '%' }"></span></span>
        <span class="pp-bench-val">{{ fmt(s as number) }}<em v-if="i > 0">{{ ((s as number) / (G.rows[0][1] as number)) < 10 ? ((s as number) / (G.rows[0][1] as number)).toFixed(1) : Math.round((s as number) / (G.rows[0][1] as number)) }}× slower</em></span>
      </div>
    </div>
    <p class="pp-credit">Wall-clock times from the graph in memory to the full answer in memory, the same answers checked between tools. Scripts and raw measurements are public.</p>
  </div>
</template>
