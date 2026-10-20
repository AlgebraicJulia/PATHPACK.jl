using Documenter
using DocumenterVitepress
using Literate

const literate_dir = joinpath(@__DIR__, "literate")
const generated_dir = joinpath(@__DIR__, "src", "generated")

@info "Loading PATHPACK"
using PATHPACK

# Tutorials whose code needs hardware the docs build does not have: shown, not run.
const not_executed = ("gpus",)

# A notebook of each tutorial, served next to the site (src/public is copied to its root), and a download link
# under the tutorial's title.
const notebook_dir = joinpath(@__DIR__, "src", "public", "notebooks")

function notebook_link(content, name)
  note = name in not_executed ? "The same code, ready to run in Jupyter on a machine with an NVIDIA GPU." :
    "The same code as on this page, ready to run in Jupyter with IJulia."
  icon = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12m0 0l-5-5m5 5l5-5M5 21h14"/></svg>"""
  link = """
  #md # ```@raw html
  #md # <p class="pp-t-nb"><a class="pp-t-nb-btn" href="../notebooks/$name.ipynb" download>$(icon)Download notebook</a><span>$note</span></p>
  #md # ```
  """
  title, rest = split(content, "\n"; limit = 2)
  return string(title, "\n", link, rest)
end

const no_literate = "--no-literate" in ARGS
if !no_literate
  @info "Building Literate.jl docs"
  for (root, dirs, files) in walkdir(literate_dir)
    out_dir = joinpath(generated_dir, relpath(root, literate_dir))
    for file in files
      f, l = splitext(file)
      if l == ".jl" && !startswith(f, "_")
        path = joinpath(root, file)
        fence = f in not_executed ? (; codefence = "```julia" => "```") : (;)
        Literate.markdown(path, out_dir; documenter = true, credit = false, preprocess = s -> notebook_link(s, f), fence...)
        Literate.notebook(path, notebook_dir; execute = false, credit = false)
      end
    end
  end
end

@info "Running the explorable code cells"
include(joinpath(@__DIR__, "explore.jl"))

@info "Building Documenter.jl docs"
makedocs(
  modules = [PATHPACK],
  repo = "https://github.com/AlgebraicJulia/PATHPACK.jl",
  sitename = "PATHPACK.jl",
  format = DocumenterVitepress.MarkdownVitepress(
    repo = "https://github.com/AlgebraicJulia/PATHPACK.jl",
    devbranch = "main",
    devurl = "dev",
    # there is no tagged release yet, so dev is the only version: let search engines index it (turn this back on, the
    # default, once there is a stable version, so that only stable is indexed)
    noindex_non_stable = false,
    description = "All-pairs shortest paths, widest paths, reachability and more, on CPUs and NVIDIA GPUs",
  ),
  doctest = false,
  checkdocs = :exports,
  pages = Any[
    "Home" => "index.md",
    "Guide" => Any[
      "Get Started" => "getting-started.md",
      "How It Works" => "how-it-works.md",
    ],
    "Tutorials" => Any[
      "Overview" => "tutorials.md",
      "Road Networks" => "generated/road_networks.md",
      "One Graph, Many Questions" => "generated/semirings.md",
      "Shortest and Widest Paths" => "generated/shortest_paths.md",
      "Longest Paths and Schedules" => "generated/critical_paths.md",
      "Most Reliable Routes" => "generated/reliability.md",
      "Bandwidth and Widest Paths" => "generated/widest_paths.md",
      "Access Audits" => "generated/access_audits.md",
      "Factorize Once, Query Many Times" => "generated/queries.md",
    ],
    "Applications" => Any[
      "Exact Diffusion for GNNs" => "generated/gnn_diffusion.md",
      "Isomap and Manifold Learning" => "generated/isomap.md",
      "Distances on Surfaces" => "generated/brain_maps.md",
    ],
    "Scaling Up" => Any[
      "GPUs, One or Many" => "generated/gpus.md",
      "Performance Guide" => "generated/performance.md",
      "Benchmarks" => "benchmarks.md",
    ],
    "Reference" => Any[
      "Library Reference" => "api.md",
      "Release Notes" => "changelog.md",
    ],
  ],
)

@info "Deploying docs"
DocumenterVitepress.deploydocs(
  repo = "github.com/AlgebraicJulia/PATHPACK.jl.git",
  target = joinpath(@__DIR__, "build"),
  branch = "gh-pages",
  devbranch = "main",
  push_preview = true,
)
