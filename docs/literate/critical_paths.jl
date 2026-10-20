# # Longest Paths and Schedules
#
# A project is a set of tasks, each with a duration, and a set of precedences: a task can start only when the tasks
# before it are done. The earliest a task can start is the length of the longest chain of work that leads to it, and
# the project's length is the longest chain of all, the critical path. In the max-plus semiring (``\oplus = \max``,
# ``\otimes = +``) the star of the precedence graph holds all of these longest paths at once.

# ## A Small Project
#
# Building a house: each arc `a → b` means `b` starts after `a` is done, and its weight is the duration of `a`, in
# days. Two zero-length tasks, `start` and `finish`, mark the ends of the project.

using PATHPACK, SparseArrays, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: MaxPlus, mstar

tasks = ["start", "permits", "foundation", "framing", "roof", "plumbing",
    "electrical", "insulation", "drywall", "paint", "floors", "finish"]
days = [0, 5, 7, 10, 4, 6, 5, 3, 6, 3, 4, 0]
after = [("start", "permits"), ("permits", "foundation"),
    ("foundation", "framing"), ("framing", "roof"), ("framing", "plumbing"),
    ("framing", "electrical"), ("roof", "insulation"),
    ("plumbing", "insulation"), ("electrical", "insulation"),
    ("insulation", "drywall"), ("drywall", "paint"), ("drywall", "floors"),
    ("paint", "finish"), ("floors", "finish")]

id = Dict(t => i for (i, t) in enumerate(tasks))
n = length(tasks)
src = [id[a] for (a, b) in after]
dst = [id[b] for (a, b) in after]
A = sparse(src, dst, Float64.(days[src]), n, n);

# ## Every Longest Path
#
# `L[i, j]` is the length of the longest chain of work from the start of `i` to the start of `j`, and `-Inf` (the
# semiring's zero) when `j` does not depend on `i`.

L = mstar(MaxPlus(), A)
s, f = id["start"], id["finish"]
L[s, f]    # the project takes this many days

# Row `s` holds the earliest start of every task, and column `f` the length of the work that must follow each task
# (its own duration included). A task's slack is how long it can slip without delaying the project; the tasks with no
# slack form the critical path.

earliest = L[s, :]
slack = L[s, f] .- earliest .- L[:, f]
[tasks earliest slack]

#-

tasks[slack .== 0]    # the critical path

#-

figure(panel("the schedule", gantt(tasks, earliest, days, slack .== 0; slack = slack); sub = "days"),  #hide
    caption = "Each task starts at its earliest time, a row of the max-plus star. Tasks on the critical path, in red, have no slack; the grey tails show how long the others can slip.")  #hide

# The roof can slip two days, and the electrical work and the painting one day each. Every other task is critical.
#
# Because the star holds every pair, other questions need no new computation: `L[id["roof"], id["paint"]]` is the
# least time between starting the roof and starting the paint, and a column of `L` lists everything a task waits on.

L[id["roof"], id["paint"]]

# ## A Check on a Larger Project
#
# On a directed acyclic graph the longest paths from one source also follow from a single pass in topological order.
# Here is a random project of 2,000 tasks, numbered so that every precedence goes from a lower to a higher number,
# and the pass checks the first row of the star.

Random.seed!(1)
m = 2000
src = Int[]; dst = Int[]; len = Float64[]
for j in 2:m, _ in 1:3
    i = rand(max(1, j - 50):(j - 1))
    push!(src, i); push!(dst, j); push!(len, rand(1:10))
end
B = sparse(src, dst, len, m, m, max)
LB = mstar(MaxPlus(), B)

longest = fill(-Inf, m); longest[1] = 0.0
for j in 2:m, p in nzrange(B, j)
    i = rowvals(B)[p]
    longest[j] = max(longest[j], longest[i] + nonzeros(B)[p])
end
LB[1, :] == longest

# ## Cycles
#
# A longest path is well defined only when no cycle has positive length: otherwise a path could go around it forever.
# In a precedence graph there are no cycles at all. If a graph has a cycle of positive length, the max-plus star is
# `Inf` for every pair whose paths can pass through it, which flags the circular dependency.

C = sparse([1, 2, 3], [2, 3, 1], [1.0, 1.0, 1.0], 3, 3)    # a cycle of length 3
mstar(MaxPlus(), C)

# The GPU solver also takes `MaxPlus`, but only for undirected graphs and for directed graphs whose strongly connected
# components do not reach one another. A precedence graph is neither, so schedules run on the CPU solver, `mstar`.
