<script setup lang="ts">
// The Enhanced Readability menu (page width, spotlight), with the accessible name its button lacks and without the
// stray menuitem role on its wrapper (it sits in the nav bar, not inside a menu).
import { onMounted, ref, nextTick } from 'vue'
import { NolebaseEnhancedReadabilitiesMenu } from '@nolebase/vitepress-plugin-enhanced-readabilities/client'
const root = ref<HTMLElement>()
onMounted(async () => {
  await nextTick()
  const el = root.value; if (!el) return
  el.querySelectorAll('[role="menuitem"]').forEach((e) => e.removeAttribute('role'))
  el.querySelectorAll('button').forEach((b) => { if (!b.getAttribute('aria-label') && !b.textContent?.trim()) b.setAttribute('aria-label', 'Reading layout') })
})
</script>

<template>
  <div ref="root" class="pp-readability"><NolebaseEnhancedReadabilitiesMenu /></div>
</template>
