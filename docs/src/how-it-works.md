```@raw html
---
title: How It Works
aside: false
pageClass: pp-wide
---
```

```@raw html
<div class="pp-hiw-page">
<p class="pp-eyebrow">How it works</p>
<h1>All pairs, from one sparse factorization.</h1>
<div class="pp-hiw-intro">
<p>An algebraic path problem asks for the <em>star</em> of a graph's weighted adjacency matrix, <span style="white-space: nowrap">A* = I ⊕ A ⊕ A² ⊕ ⋯</span>, over a semiring. With ⊕ = min and ⊗ = +, A*[i, j] is the shortest distance from i to j. Other semirings ask other questions with the same formula: widest paths (max, min), bottleneck paths (min, max), reachability (or, and), sums over all walks (+, ×).</p>
<p>Floyd–Warshall computes A* with the loops of Gaussian elimination: n³ updates on a dense matrix. PATHPACK.jl uses a sparse direct method instead. It orders the vertices to limit fill, factorizes A into sparse triangular factors L and U over the semiring, and then solves against the identity, row by row, to get A*.</p>
<p>The steps below run on a small street grid. Every order, factor and distance in them is computed in your browser for the graph shown.</p>
</div>
<HowItWorks />
<div class="pp-hiw-after">
<h2>In code</h2>
<p>One call does all of it. To answer many single-source questions, keep the factorization: each query is two triangular solves.</p>
<pre class="pp-codeblock"><code><span class="k">using</span> PATHPACK, SparseArrays, LinearAlgebra
<span class="k">using</span> PATHPACK.CPU: MinPlus, mstar, mlu
D = <span class="f">mstar</span>(<span class="s">MinPlus</span>(), A)                   <span class="c"># all of A*: order, factorize, solve</span>
F = <span class="f">mlu</span>(<span class="s">MinPlus</span>(), A)                     <span class="c"># or keep the factors</span>
B = <span class="f">fill</span>(Inf, 1, size(A, 1)); B[s] = 0.0  <span class="c"># the unit row vector at s</span>
<span class="f">rmul!</span>(B, F)                               <span class="c"># row s of A*: the distances from s</span></code></pre>
<p>The keyword <code>alg</code> of <code>mlu</code> and <code>mstar</code> picks the ordering, for example <code>METIS()</code> (nested dissection) from CliqueTrees.jl. On NVIDIA GPUs, <code>apsp_gpu(A)</code> runs the same method: CPU threads factorize the many small parts, the GPU the large separator blocks, and the rows of A* are split across one GPU or several.</p>
</div>
</div>
```
