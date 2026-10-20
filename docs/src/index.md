```@raw html
---
layout: page
title: PATHPACK.jl
sidebar: false
aside: false
---
```

```@raw html
<div class="pp-page">

<LogoHero />

<div class="pp-stats2">
  <div><b>0.14 s</b><span>all 15.9 billion distances of the US road network, on one GPU</span></div>
  <div><b>760×</b><span>faster than Fast-APSP on that network; 130× faster than cuGraph</span></div>
  <div><b>70×</b><span>faster than igraph for the closeness of every street in Paris</span></div>
  <div><b>8 GPUs</b><span>for 190 billion distances in 0.39 s; one GPU or many, one call</span></div>
</div>

<section class="pp-sec">
  <p class="pp-eyebrow">Try it</p>
  <h2 class="pp-h">Distance to everywhere, at once.</h2>
  <p class="pp-sec-lead">Move across Paris. Your browser recomputes one source's distances on every move. PATHPACK.jl
  answers the whole question, the distance from every junction to every other one, in one call.</p>
  <ParisLive />
</section>

<section class="pp-sec">
  <p class="pp-eyebrow">One call, any semiring</p>
  <h2 class="pp-h">Same graph. Different question.</h2>
  <p class="pp-sec-lead">The star of a matrix depends on the semiring it is computed in. Switch the semiring and the
  same landscape answers a different question: the cheapest route, the worst barrier on the way, or where a wandering
  animal spends its time.</p>
  <LandscapeLive />
</section>

<section class="pp-sec">
  <p class="pp-eyebrow">Built for real workloads</p>
  <h2 class="pp-h">Faster inside the tools you already use.</h2>
  <p class="pp-sec-lead">Many methods need the distance between every pair of points. PATHPACK.jl computes it
  directly, and measured head to head, same answers, it is faster than the call each field uses today.</p>
  <AppGrid />
</section>

<section class="pp-sec">
  <p class="pp-eyebrow">Any graph</p>
  <h2 class="pp-h">Atoms, scans, surfaces.</h2>
  <p class="pp-sec-lead">Proteins through their atomic contacts, surfaces through their meshes. Click the bunny to
  measure from anywhere on it; below, drag to turn an object and double-click to move the source of its bands.</p>
  <MeshPaint />
  <div style="height: 22px"></div>
  <Gallery3D />
</section>

<section class="pp-sec">
  <p class="pp-eyebrow">Measured</p>
  <h2 class="pp-h">Faster than what you use now.</h2>
  <p class="pp-sec-lead">On one GPU against the GPU state of the art, and on the same CPU cores against the libraries
  people use.</p>
  <BenchChart />
</section>

<section class="pp-sec">
  <p class="pp-eyebrow">Get started</p>
  <h2 class="pp-h">Two lines.</h2>
</section>

</div>
```


```@raw html
<div class="pp-page">
<pre class="pp-codeblock"><code><span class="k">using</span> PATHPACK
<span class="k">using</span> PATHPACK.CPU: MinPlus, MaxMin, mstar
<span></span>
D = <span class="f">mstar</span>(<span class="s">MinPlus</span>(), A)    <span class="c"># shortest paths between all pairs</span>
W = <span class="f">mstar</span>(<span class="s">MaxMin</span>(), A)     <span class="c"># widest paths, same call</span></code></pre>
<div class="pp-features">
  <div class="pp-feature"><h4>Any semiring</h4><p>Shortest, widest, bottleneck, most reliable, reachability, sums over all walks: one solver.</p></div>
  <div class="pp-feature"><h4>Multithreaded CPUs</h4><p>One sparse factorization, then independent rows: every core busy, no write conflicts.</p></div>
  <div class="pp-feature"><h4>One GPU or many</h4><p>The CPU factorizes the small parts, the GPU the large ones; the rows split over every GPU you have.</p></div>
  <div class="pp-feature"><h4>Queries, fast</h4><p>Factorize once, then answer single-source and single-target queries with two triangular solves.</p></div>
</div>
<div class="pp-hero-actions" style="margin-top: 34px">
  <a class="pp-btn pp-btn-brand" href="./generated/road_networks">Road networks tutorial</a>
  <a class="pp-btn" href="./generated/semirings">One graph, many questions</a>
  <a class="pp-btn" href="./api">Library reference</a>
</div>
</div>
```
