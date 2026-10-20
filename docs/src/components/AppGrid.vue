<script setup lang="ts">
// Where all-pairs answers are used, each with a picture drawn from its real data (AppVisual) and its measured
// head-to-head (one 32-core AMD EPYC node with the other tool on the same machine, or one NVIDIA B200).
const APPS = [
  { key: 'brain', field: 'Neuroscience', title: 'Brain maps on the folded cortex',
    text: 'Spatial null models compare brain maps through the distance along the cortex between every pair of its points.',
    big: '2,000×', vs: 'faster than BrainSMASH\'s distance step: 1.6 hours to 2.6 seconds', call: 'MinPlus' },
  { key: 'swiss', field: 'Machine learning', title: 'Manifold learning',
    text: 'Isomap unrolls data by the shortest paths of its nearest-neighbour graph: every pair of points.',
    big: '105×', vs: 'faster than scikit-learn\'s Isomap shortest-path step', call: 'MinPlus' },
  { key: 'gnn', field: 'Graph neural networks', title: 'Exact diffusion',
    text: 'Personalized PageRank diffusion, as PyTorch Geometric\'s GDC computes it, is a sum over every walk.',
    big: '119×', vs: 'faster than PyTorch Geometric\'s exact diffusion on a 47k-node mesh; 23× on PubMed', call: 'PlusProd', ring: true },
  { key: 'paris', field: 'Urban planning', title: 'How central is every street?',
    text: 'Closeness: each street\'s mean walking distance to all the others, from all 140 million distances of Paris.',
    big: '70×', vs: 'faster than igraph; 3.4× faster than graph-tool on the same 16 cores', call: 'MinPlus' },
  { key: 'landscape', field: 'Ecology', title: 'Landscape connectivity',
    text: 'Least-cost distances between every pair of habitat cells, and where a wandering animal goes.',
    big: '15×', vs: 'faster than NetworKit on the same 16 cores, every cell to every cell', call: 'MinPlus · MinMax · PlusProd', ring: true },
  { key: 'amazon', field: 'Geography', title: 'River networks',
    text: 'Every pair of points on the Amazon river system, by boat, from OpenStreetMap waterways.',
    big: '9.5 ms', vs: 'for all 12,785 junctions, every pair, on one GPU', call: 'MinPlus', ring: true },
]
</script>

<template>
  <div class="pp-apps">
    <article v-for="a in APPS" :key="a.key" class="pp-app">
      <div class="pp-app-media"><SwissUnroll v-if="a.key === 'swiss'" /><AppVisual v-else :name="a.key" :ring="a.ring" :label="a.title" /></div>
      <div class="pp-app-body">
        <p class="pp-app-field">{{ a.field }}</p>
        <h3>{{ a.title }}</h3>
        <p class="pp-app-text">{{ a.text }}</p>
        <p class="pp-app-num"><b>{{ a.big }}</b><span>{{ a.vs }}</span></p>
        <p class="pp-app-call"><code>{{ a.call }}</code></p>
      </div>
    </article>
  </div>
</template>
