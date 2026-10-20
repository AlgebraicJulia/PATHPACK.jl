<script setup lang="ts">
// An explorable code cell: Julia code whose highlighted parts are controls. Every combination was run with
// PATHPACK.jl when the docs were built (docs/explore.jl); changing a control shows that run's output at once.
import { ref, computed, onMounted } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{ id: string }>()
const cell = ref<any>(null), sel = ref<Record<string, string>>({}), copied = ref(false)
onMounted(async () => {
  cell.value = await (await fetch(withBase(`/explore/${props.id}.json`))).json()
  for (const p of cell.value.params) sel.value[p.name] = p.values[0]
})
const KW = /\b(using|function|end|for|in|if|else|return|const|let|do|begin)\b/g
const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
function hl(s: string) {   // light highlighting of a code fragment (no slots inside)
  return esc(s).replace(/(#.*)$/gm, '<span class="c">$1</span>').replace(KW, '<span class="k">$1</span>')
    .replace(/\b(\d+(\.\d+)?)\b/g, '<span class="n">$1</span>')
}
// the code split into text and slots
const parts = computed(() => {
  if (!cell.value) return []
  const out: { t: 'txt' | 'slot'; v: string }[] = [], re = /\{(\w+)\}/g
  let last = 0, m: RegExpExecArray | null
  const code = cell.value.code as string
  while ((m = re.exec(code))) { out.push({ t: 'txt', v: code.slice(last, m.index) }); out.push({ t: 'slot', v: m[1] }); last = m.index + m[0].length }
  out.push({ t: 'txt', v: code.slice(last) })
  return out
})
const param = (name: string) => cell.value?.params.find((p: any) => p.name === name)
const output = computed(() => cell.value ? cell.value.outputs[cell.value.params.map((p: any) => sel.value[p.name]).join('|')] ?? '' : '')
function cycle(name: string) { const p = param(name); const i = p.values.indexOf(sel.value[name]); sel.value[name] = p.values[(i + 1) % p.values.length] }
function copy() {
  const code = parts.value.map((x) => (x.t === 'txt' ? x.v : sel.value[x.v])).join('')
  navigator.clipboard?.writeText(code).then(() => { copied.value = true; setTimeout(() => (copied.value = false), 1400) }).catch(() => {})
}
</script>

<template>
  <div class="pp-x" v-if="cell">
    <div class="pp-x-head">
      <span class="pp-x-badge">Explorable</span>
      <span class="pp-x-hint">Change the highlighted parts; the output is PATHPACK.jl's own.</span>
      <button class="pp-x-copy" type="button" @click="copy">{{ copied ? 'copied' : 'copy code' }}</button>
    </div>
    <pre class="pp-x-code"><code><template v-for="(x, i) in parts" :key="i"><span v-if="x.t === 'txt'" v-html="hl(x.v)"></span><span v-else class="pp-x-slot"><select v-model="sel[x.v]" :aria-label="param(x.v).label" @dblclick="cycle(x.v)"><option v-for="v in param(x.v).values" :key="v" :value="v">{{ v }}</option></select></span></template></code></pre>
    <pre class="pp-x-out"><code>{{ output }}</code></pre>
    <p class="pp-x-note" v-if="cell.note">{{ cell.note }}</p>
  </div>
  <div class="pp-x pp-x-loading" v-else>Loading…</div>
</template>
