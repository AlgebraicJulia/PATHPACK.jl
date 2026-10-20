<script setup lang="ts">
// How PATHPACK.jl computes A*, in five steps, on a small street grid: the graph and its matrix; a nested-dissection
// reordering; elimination with fill-in (three orders compared); every row of A* solved independently; and one row
// from the stored factors. Everything drawn is computed in the browser for the graph shown (hiw/model.ts).
import { onMounted, onBeforeUnmount, ref, reactive, computed, watch, nextTick } from 'vue'
import type { Model, Key } from './hiw/model'
import type { UI, QueryInfo } from './hiw/scene'

const STEPS = ['The graph is a matrix', 'Reorder', 'Eliminate', 'Solve', 'Query']
const ORDERS: { key: Key; label: string }[] = [
  { key: 'input', label: 'Input order' }, { key: 'sweep', label: 'Row by row' }, { key: 'nd', label: 'Nested dissection' }]
const ui = reactive<UI>({ step: 0, ord: 'nd', elim: 0, workers: 4, src: 0, hoverV: -1, hoverR: -1, hoverC: -1 })
const ready = ref(false), playing = ref(false), reduce = ref(false)
const gcan = ref<HTMLCanvasElement>(), mcan = ref<HTMLCanvasElement>(), tabs = ref<HTMLButtonElement[]>([])
const info = reactive({ n: 144, m: 232, nnz: 464 })
const stats = ref<Record<Key, { fill: number; work: number; height: number }>>({
  input: { fill: 0, work: 0, height: 0 }, sweep: { fill: 0, work: 0, height: 0 }, nd: { fill: 0, work: 0, height: 0 } })
const q = ref<QueryInfo>({ up: 0, touchU: 0, touchL: 0 })
interface TNode { id: number; x: number; y: number; r: number; sep: boolean; size: number; first: number; px: number; py: number; depth: number }
const tree = ref<{ nodes: TNode[]; h: number; split: number }>({ nodes: [], h: 120, split: 0 })
// separators get a tone by their level: the top one darkest red (--pp-sep0), each level below lighter
const sepLevels = ref<number[]>([])
const sepRank = (depth: number) => Math.min(3, Math.max(0, sepLevels.value.indexOf(depth)))
let md: Model | null = null
let scene: ReturnType<typeof import('./hiw/scene').createScene> | null = null

const TW = 640
const fmt = (x: number) => x.toLocaleString('en-US')
const ratio = (a: number, b: number) => (b ? (a / b).toFixed(1) : '')
const maxFill = computed(() => Math.max(1, ...ORDERS.map(o => stats.value[o.key].fill)))
const label = (v: number, key?: Key) => (md ? md.ords[key ?? (ui.step === 0 ? 'input' : ui.step === 2 ? ui.ord : 'nd')].pos[v] + 1 : 0)
const nodeDone = (x: TNode) => (ui.step === 2 ? Math.max(0, Math.min(1, (ui.elim - x.first) / x.size)) : 1)
const upNodes = computed(() => {
  const s = new Set<number>(); if (!ready.value || !md || ui.step !== 4) return s
  let v = ui.src; while (v >= 0) { s.add(md.nd.nodeOf[v]); v = md.sym.nd.parent[v] }
  return s
})

// ---- readouts under the two panels ----
const graphRead = computed(() => {
  if (!ready.value || !md) return ''
  const v = ui.hoverV, g = md.g
  if (ui.step === 4) {
    if (v >= 0 && v !== ui.src) return `Corner ${label(v)}: distance ${md.D[ui.src * g.n + v].toFixed(2)} from corner ${label(ui.src)}.`
    return `Distances from corner ${label(ui.src)}. Click another corner.`
  }
  if (ui.step === 3) {
    const per = g.n / ui.workers
    if (v >= 0) return `Row ${label(v)} of A*: the distances from this corner, computed by ${ui.workers === 4 ? 'CPU thread' : 'GPU'} ${Math.floor(md.ords.nd.pos[v] / per) + 1}.`
    return `Each corner's row of A* is computed by one ${ui.workers === 4 ? 'thread' : 'GPU'} (its color).`
  }
  if (v >= 0) {
    const deg = g.nbr[v].length
    if (ui.step === 2) {
      const P = md.ords[ui.ord].pos
      return P[v] < ui.elim ? `Corner ${label(v)}: eliminated (step ${P[v] + 1}).` : `Corner ${label(v)}: not yet eliminated; ${deg} street${deg === 1 ? '' : 's'}.`
    }
    if (ui.step === 1) {
      const x = md.nd.nodes[md.nd.nodeOf[v]]
      return `Corner ${label(v)}: ${x.sep ? `on a separator (level ${x.depth + 1})` : 'in a part'}; row and column ${label(v)}.`
    }
    return `Corner ${label(v)}: ${deg} street${deg === 1 ? '' : 's'}, so row ${label(v)} has ${deg} nonzero${deg === 1 ? '' : 's'}.`
  }
  return ui.step === 2 ? 'Gray rings: eliminated. Red lines: fill, links made by elimination.' : 'Hover a corner.'
})
const matRead = computed(() => {
  if (!ready.value || !md) return ''
  const a = ui.hoverR, b = ui.hoverC, g = md.g
  if (ui.step === 4) return `Blue: rows of U read by the forward solve (${fmt(q.value.touchU)} entries). Red: L, read by the backward solve (${fmt(q.value.touchL)}). Bottom: the row of A*.`
  if (a < 0) return ui.step === 3 ? 'Hover an entry.' : 'Hover an entry. Empty entries are ∞: no street.'
  const i = label(a), j = label(b)
  if (ui.step === 3) return `A*[${i}, ${j}] = ${md.D[a * g.n + b].toFixed(2)}: the distance from ${i} to ${j}.`
  if (a === b) return `Diagonal entry [${i}, ${i}].`
  const e = g.edges.find(([p, r]) => (p === a && r === b) || (p === b && r === a))
  if (e) return `A[${i}, ${j}] = ${e[2].toFixed(2)}: the street from ${i} to ${j}.`
  if (ui.step === 2) {
    const S = md.sym[ui.ord], f = S.fill.findIndex(([p, r]) => (p === a && r === b) || (p === b && r === a))
    if (f >= 0 && S.fill[f][2] < ui.elim) return `[${i}, ${j}]: fill, made when corner ${S.fill[f][2] + 1} was eliminated.`
  }
  return `A[${i}, ${j}] = ∞: no street from ${i} to ${j}.`
})
const graphHead = computed(() => ['144 corners, 232 streets', 'separators in red, darker at the top', 'with fill', 'colored by worker', 'distance from the source'][ui.step])
const matHead = computed(() => ['A, input order', 'A, nested-dissection order', `L + U, ${ORDERS.find(o => o.key === ui.ord)!.label.toLowerCase()}`, 'A*, one row per corner', 'L + U, and one row of A*'][ui.step])

// ---- stepping ----
function go(i: number, focus = false) {
  if (i < 0 || i >= STEPS.length) return
  ui.step = i; ui.hoverV = ui.hoverR = ui.hoverC = -1; playing.value = false
  if (i === 2) { ui.elim = 0; if (!reduce.value) playing.value = true; else ui.elim = info.n }
  if (i === 4 && md) q.value = scene!.query(ui.src)
  scene?.restart()
  if (focus) nextTick(() => tabs.value[i]?.focus())
}
function onTabKey(e: KeyboardEvent) {
  const k = e.key
  if (k === 'ArrowRight' || k === 'ArrowDown') go(Math.min(STEPS.length - 1, ui.step + 1), true)
  else if (k === 'ArrowLeft' || k === 'ArrowUp') go(Math.max(0, ui.step - 1), true)
  else if (k === 'Home') go(0, true)
  else if (k === 'End') go(STEPS.length - 1, true)
  else return
  e.preventDefault()
}
function setOrd(k: Key) { ui.ord = k; scene?.touch() }
function togglePlay() {
  if (playing.value) { playing.value = false; return }
  if (ui.elim >= info.n) ui.elim = 0
  if (reduce.value) { ui.elim = info.n; return }
  playing.value = true
}
function onSlider(e: Event) { playing.value = false; ui.elim = +(e.target as HTMLInputElement).value }
watch(() => ui.elim, () => scene?.elimChanged())
watch(() => [ui.workers, ui.ord], () => scene?.touch())

// ---- pointer ----
const local = (e: PointerEvent) => { const r = (e.target as HTMLElement).getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top] }
function gMove(e: PointerEvent) { if (!scene) return; const [x, y] = local(e); ui.hoverV = scene.vertexAt(x, y); ui.hoverR = ui.hoverC = -1; scene.touch() }
function mMove(e: PointerEvent) {
  if (!scene) return; const [x, y] = local(e)
  if (ui.step === 4) { ui.hoverV = -1; ui.hoverR = ui.hoverC = -1; scene.touch(); return }
  const [a, b] = scene.cellAt(x, y); ui.hoverR = a; ui.hoverC = b; ui.hoverV = -1; scene.touch()
}
function leave() { ui.hoverV = ui.hoverR = ui.hoverC = -1; scene?.touch() }
function gClick(e: PointerEvent) {
  if (!scene || ui.step !== 4) return
  const [x, y] = local(e), v = scene.vertexAt(x, y, 30); if (v >= 0) { ui.src = v; q.value = scene.query(v) }
}
function mClick(e: PointerEvent) {
  if (!scene || ui.step !== 4) return
  const [x, y] = local(e), [a] = scene.cellAt(x, y); if (a >= 0) { ui.src = a; q.value = scene.query(a) }
}
function randomSource() { if (!md || !scene) return; let v = ui.src; while (v === ui.src) v = Math.floor(Math.random() * md.g.n); ui.src = v; q.value = scene.query(v) }

let stop = () => {}
onMounted(async () => {
  reduce.value = matchMedia('(prefers-reduced-motion: reduce)').matches
  const [{ build }, { createScene }] = await Promise.all([import('./hiw/model'), import('./hiw/scene')])
  md = build()
  const g = md.g
  info.n = g.n; info.m = g.edges.length; info.nnz = 2 * g.edges.length
  for (const k of ['input', 'sweep', 'nd'] as Key[]) stats.value[k] = { fill: md.sym[k].fill.length * 2, work: md.sym[k].work, height: md.sym[k].height }
  // the source of the query step: a corner near the lower left
  let best = 0; for (let v = 0; v < g.n; v++) if (g.x[v] - g.y[v] < g.x[best] - g.y[best]) best = v
  ui.src = best
  // the elimination tree, one node per part: leaves spread evenly, each separator above the middle of its two parts
  const nodes: TNode[] = [], maxDepth = Math.max(...md.nd.nodes.map(x => x.depth)), leaves = md.nd.nodes.filter(x => !x.sep).length
  const W = TW, top = 16, gap = 30
  let slot = 0
  const xs = new Map<number, number>()
  const place = (x: Model['nd']['nodes'][0]): number => {
    const cx = x.kids.length ? x.kids.map(place).reduce((p, c) => p + c, 0) / x.kids.length : 96 + ((slot++ + 0.5) / leaves) * (W - 106)
    xs.set(x.id, cx); return cx
  }
  place(md.nd.root)
  const yOf = (d: number) => top + d * gap
  const visit = (x: Model['nd']['nodes'][0], par: Model['nd']['nodes'][0] | null) => {
    nodes.push({ id: x.id, x: xs.get(x.id)!, y: yOf(x.depth), r: 3 + Math.sqrt(x.size) * 2.3, sep: x.sep, size: x.size, first: x.first,
      px: par ? xs.get(par.id)! : NaN, py: par ? yOf(par.depth) : NaN, depth: x.depth })
    for (const k of x.kids) visit(k, x)
  }
  visit(md.nd.root, null)
  sepLevels.value = [...new Set(nodes.filter(t => t.sep).map(t => t.depth))].sort((a, b) => a - b)
  tree.value = { nodes, h: top + maxDepth * gap + 16, split: top + 1.5 * gap }
  ready.value = true
  await nextTick()
  scene = createScene(gcan.value!, mcan.value!, md, ui, reduce.value)
  let last = 0
  scene.onFrame = (now: number) => {
    if (!playing.value || ui.step !== 2) { last = now; return }
    const rate = 1000 / 34 // ms per elimination
    if (now - last >= rate) { last = now; ui.elim = Math.min(info.n, ui.elim + 1); if (ui.elim >= info.n) playing.value = false }
  }
  stop = () => scene?.destroy()
})
onBeforeUnmount(() => stop())
</script>

<template>
  <div class="pp-hiw">
    <div class="pp-hiw-top">
    <div class="pp-hiw-steps" role="tablist" aria-label="Steps of the explainer" @keydown="onTabKey">
      <button v-for="(s, i) in STEPS" :key="s" ref="tabs" type="button" role="tab" :id="'pp-hiw-tab-' + i"
        :aria-selected="ui.step === i" aria-controls="pp-hiw-panel" :tabindex="ui.step === i ? 0 : -1"
        :class="{ on: ui.step === i, past: i < ui.step }" @click="go(i)">
        <span class="pp-hiw-num">{{ i + 1 }}</span><span class="pp-hiw-lbl">{{ s }}</span>
      </button>
    </div>
    <div class="pp-hiw-arrows">
      <button type="button" class="pp-hiw-btn" :disabled="ui.step === 0" @click="go(ui.step - 1)" aria-label="Previous step">
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3 5 8l5 5" /></svg>
      </button>
      <button type="button" class="pp-hiw-btn brand" :disabled="ui.step === STEPS.length - 1" @click="go(ui.step + 1)" aria-label="Next step">
        <span>Next</span><svg viewBox="0 0 16 16" aria-hidden="true"><path d="m6 3 5 5-5 5" /></svg>
      </button>
    </div>
    </div>
    <div class="pp-hiw-body" id="pp-hiw-panel" role="tabpanel" :aria-labelledby="'pp-hiw-tab-' + ui.step">
      <div class="pp-hiw-text">
        <p class="pp-hiw-kicker">Step {{ ui.step + 1 }} of {{ STEPS.length }}</p>
        <template v-if="ui.step === 0">
          <h3>The graph is a matrix</h3>
          <p>{{ info.n }} street corners and {{ info.m }} streets. Row <i>i</i> of the matrix <i>A</i> holds the streets
            out of corner <i>i</i>: <i>A</i>[<i>i</i>, <i>j</i>] is the length of the street from <i>i</i> to <i>j</i>.
            An empty entry is ∞: no street.</p>
          <p>PATHPACK.jl computes the star <i>A</i>* = <i>I</i> ⊕ <i>A</i> ⊕ <i>A</i>² ⊕ ⋯. With ⊕ = min and ⊗ = +, its
            entry <i>A</i>*[<i>i</i>, <i>j</i>] is the distance from <i>i</i> to <i>j</i>. Floyd–Warshall finds it with
            <i>n</i>³ updates on a dense matrix. PATHPACK.jl keeps <i>A</i> sparse for as long as it can.</p>
          <p class="pp-hiw-hint">The corners are numbered in the order a data file might list them, so the nonzeros
            look scattered. Hover a corner or an entry.</p>
        </template>
        <template v-else-if="ui.step === 1">
          <h3>Reorder: nested dissection</h3>
          <p>First, renumber the corners so that the factors stay sparse. Nested dissection, one such order, cuts the
            graph in two with a small separator (red), numbers both halves first and the separator last, then cuts each
            half the same way. Each level of cuts has its own shade: the first separator is the darkest red, the
            separators of its two halves a lighter one, and so on down the tree.</p>
          <p>The matrix is the same, its rows and columns permuted. No street joins two halves, so the blocks between them
            are empty, and they stay empty when the matrix is factorized.</p>
        </template>
        <template v-else-if="ui.step === 2">
          <h3>Eliminate: fill-in</h3>
          <p>Gaussian elimination over the semiring, one corner at a time. Eliminating a corner links its remaining
            neighbors to each other, through it. Each new link is a new nonzero, called fill (red). <i>A</i> plus the
            fill is the shape of the factors <i>L</i> and <i>U</i>.</p>
          <div class="pp-hiw-orders" role="group" aria-label="Elimination order">
            <button v-for="o in ORDERS" :key="o.key" type="button" :aria-pressed="ui.ord === o.key" :class="{ on: ui.ord === o.key }" @click="setOrd(o.key)">
              <span class="pp-hiw-oname">{{ o.label }}</span>
              <span class="pp-hiw-obar"><i :style="{ width: (100 * stats[o.key].fill / maxFill) + '%' }"></i></span>
              <span class="pp-hiw-oval">+{{ fmt(stats[o.key].fill) }}</span>
            </button>
          </div>
          <p class="pp-hiw-small">New nonzeros. Nested dissection makes {{ ratio(stats.input.fill, stats.nd.fill) }}× less
            fill than the input order and needs {{ ratio(stats.input.work, stats.nd.work) }}× fewer updates
            ({{ fmt(stats.nd.work) }} vs {{ fmt(stats.input.work) }}).</p>
          <div class="pp-hiw-play">
            <button type="button" class="pp-hiw-btn" :aria-label="playing ? 'Pause elimination' : 'Play elimination'" @click="togglePlay">
              <svg v-if="!playing" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" /></svg>
              <svg v-else viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" /></svg>
            </button>
            <input type="range" min="0" :max="info.n" :value="ui.elim" @input="onSlider" aria-label="Corners eliminated" />
            <span class="pp-hiw-count">{{ ui.elim }} / {{ info.n }}</span>
          </div>
        </template>
        <template v-else-if="ui.step === 3">
          <h3>Solve: every row on its own</h3>
          <p>The factors give the star, <i>A</i>* = <i>U</i>* <i>L</i>*. Each row of <i>A</i>* is a forward solve with
            <i>U</i> and a backward solve with <i>L</i>, starting from a row of the identity. Most of the work is here.</p>
          <p>No row depends on another and no two rows write the same entry, so the rows split across CPU threads, or
            across GPUs, without locks.</p>
          <div class="pp-hiw-ctrl">
            <div class="pp-seg" role="group" aria-label="Workers">
              <button type="button" :class="{ on: ui.workers === 4 }" :aria-pressed="ui.workers === 4" @click="ui.workers = 4; scene?.restart()">4 CPU threads</button>
              <button type="button" :class="{ on: ui.workers === 2 }" :aria-pressed="ui.workers === 2" @click="ui.workers = 2; scene?.restart()">2 GPUs</button>
            </div>
            <button type="button" class="pp-hiw-link" @click="scene?.restart()">Replay</button>
          </div>
          <div class="pp-hiw-chips">
            <span v-for="w in ui.workers" :key="w"><i :style="{ background: 'var(--pp-hiw-w' + w + ')' }"></i>{{ ui.workers === 4 ? 'thread' : 'GPU' }} {{ w }}</span>
          </div>
        </template>
        <template v-else>
          <h3>Query: reuse the factors</h3>
          <p>Keep <i>L</i> and <i>U</i>, and the distances from one corner, one row of <i>A</i>*, cost two triangular
            solves. Click any corner.</p>
          <p>The forward solve with <i>U</i> only climbs the elimination tree, from the source to the root: here
            {{ q.up }} of {{ info.n }} corners (dashed). The backward solve with <i>L</i> comes back down, separators
            first.</p>
          <div class="pp-hiw-ctrl">
            <button type="button" class="pp-hiw-link" @click="randomSource">Another source</button>
            <div class="pp-legend pp-legend-inline"><span>near</span><i></i><span>far</span></div>
          </div>
        </template>
      </div>
      <figure class="pp-hiw-panel">
        <figcaption class="pp-hiw-head"><b>Graph</b><span>{{ graphHead }}</span></figcaption>
        <div class="pp-hiw-stage">
          <canvas ref="gcan" class="pp-hiw-canvas" :class="{ pick: ui.step === 4 }" role="img"
            :aria-label="'The street graph, ' + graphHead" @pointermove="gMove" @pointerleave="leave" @click="gClick"></canvas>
        </div>
        <p class="pp-hiw-readout" aria-live="polite">{{ graphRead }}</p>
      </figure>
      <figure class="pp-hiw-panel">
        <figcaption class="pp-hiw-head"><b>Matrix</b><span>{{ matHead }}</span></figcaption>
        <div class="pp-hiw-stage">
          <canvas ref="mcan" class="pp-hiw-canvas" :class="{ pick: ui.step === 4 }" role="img"
            :aria-label="'The matrix, ' + matHead" @pointermove="mMove" @pointerleave="leave" @click="mClick"></canvas>
        </div>
        <p class="pp-hiw-readout" aria-live="polite">{{ matRead }}</p>
      </figure>
      <figure v-if="ready && (ui.step === 1 || ui.step === 4 || (ui.step === 2 && ui.ord === 'nd'))" class="pp-hiw-tree pp-hiw-extra">
        <svg :viewBox="'0 0 ' + TW + ' ' + tree.h" role="img" aria-label="The elimination tree, one node per part">
          <g v-if="ui.step === 2" class="pp-hiw-split">
            <line x1="0" :y1="tree.split" :x2="TW" :y2="tree.split" />
            <text x="0" :y="tree.split - 6">GPU</text>
            <text x="0" :y="tree.split + 14">CPU threads</text>
          </g>
          <g class="pp-hiw-edges">
            <line v-for="x in tree.nodes.filter(t => !Number.isNaN(t.px))" :key="'e' + x.id" :x1="x.px" :y1="x.py" :x2="x.x" :y2="x.y"
              :class="{ hot: upNodes.has(x.id) }" />
          </g>
          <g v-for="x in tree.nodes" :key="x.id" :class="['pp-hiw-node', x.sep ? 'sep' : 'part', { hot: upNodes.has(x.id) }]"
            :style="x.sep ? { '--sep': `var(--pp-sep${sepRank(x.depth)})` } : undefined">
            <title>{{ x.sep ? 'separator' : 'part' }}: {{ x.size }} corner{{ x.size === 1 ? '' : 's' }}</title>
            <circle :cx="x.x" :cy="x.y" :r="x.r" class="ring" />
            <circle :cx="x.x" :cy="x.y" :r="x.r * Math.sqrt(nodeDone(x))" class="fill" />
          </g>
        </svg>
        <figcaption v-if="ui.step === 1">The parts form a tree. Parts in different branches never touch, so they can be
          eliminated at the same time.</figcaption>
        <figcaption v-else-if="ui.step === 2">The elimination tree, one node per part, filling in as it is eliminated.
          Branches run in parallel: CPU threads take the many small parts; with a GPU, the GPU takes the large dense
          separator blocks at the top.</figcaption>
        <figcaption v-else>The parts the forward solve climbs through, from the source's part to the top separator.</figcaption>
      </figure>
      <p v-else-if="ready && ui.step === 2" class="pp-hiw-note pp-hiw-extra">With this order the elimination tree is
        <b>{{ stats[ui.ord].height }}</b> levels tall; with nested dissection, <b>{{ stats.nd.height }}</b>. A tall tree is a
        long chain of steps that wait for one another, so little of the elimination can run in parallel.</p>
      <div class="pp-hiw-foot">
        <button type="button" class="pp-hiw-btn wide" :disabled="ui.step === 0" @click="go(ui.step - 1)">Back</button>
        <span>Step {{ ui.step + 1 }} of {{ STEPS.length }}</span>
        <button type="button" class="pp-hiw-btn wide brand" :disabled="ui.step === STEPS.length - 1" @click="go(ui.step + 1)">Next</button>
      </div>
    </div>
  </div>
</template>
