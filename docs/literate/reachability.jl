# # Reachability and Labels
#
# Sometimes the question is not how far, but whether: can `j` be reached from `i` at all? And when arcs carry
# permissions, which permissions can make the whole trip? In the semiring `OrAnd` an arc's weight is a set of labels,
# stored as the bits of an integer. Along a path the labels are intersected (``\otimes = \cap``, bitwise and): a label
# survives only if every arc carries it. Over all paths they are united (``\oplus = \cup``, bitwise or). The star
# holds, for every pair `i`, `j`, the set of labels that can travel from `i` to `j`.

# ## Plain Reachability
#
# With one label, the star is the transitive closure of the graph. Take a random directed graph with 400 vertices
# and 600 arcs, every arc carrying label 1.

using PATHPACK, SparseArrays, Random
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: OrAnd, MinPlus, mstar

Random.seed!(5)
n = 400
src = rand(1:n, 600); dst = rand(1:n, 600)
A = sparse(src, dst, fill(UInt32(1), 600), n, n, |)
R = mstar(OrAnd(), A)
R[1:4, 1:4]

# The diagonal holds `0xffffffff`, all labels: the semiring's one, the empty path. Off the diagonal an entry is `1`
# when a path exists and `0` (the empty set, the semiring's zero) when none does. The shortest-path star must agree:
# a distance is finite exactly when a path exists.

reach = R .!= 0
D = mstar(MinPlus(), sparse(src, dst, ones(600), n, n, min))
reach == isfinite.(D)

#-

count(reach) - n    # the number of ordered pairs (i, j), i ≠ j, with a path from i to j

# ## Permissions Along Paths
#
# Now a building: rooms joined by doors, some of them one-way, and staff in 8 groups. Each door opens for some of the
# groups. The star tells which groups can get from any room to any other.

groups = ["visitors", "staff", "cleaning", "IT", "lab", "security", "medical", "admin"]
rooms = 120
src = Int[]; dst = Int[]; allowed = UInt32[]
for _ in 1:300
    a, b = rand(1:rooms), rand(1:rooms)
    a == b && continue
    bits = UInt32(0)
    for g in 1:8
        rand() < 0.6 && (bits |= UInt32(1) << (g - 1))
    end
    push!(src, a); push!(dst, b); push!(allowed, bits)
    if rand() < 0.8    # most doors open both ways
        push!(src, b); push!(dst, a); push!(allowed, bits)
    end
end
doors = sparse(src, dst, allowed, rooms, rooms, |)
G = mstar(OrAnd(), doors)

who(bits) = [g for (k, g) in enumerate(groups) if bits & (UInt32(1) << (k - 1)) != 0]
who(G[1, 2])    # the groups that can get from room 1 to room 2

#-

pairs = [count(x -> x & (UInt32(1) << (g - 1)) != 0, G) - rooms for g in 1:8]  #hide
figure(panel("ordered room pairs each group can travel", bars(groups, pairs; fmt = x -> string(Int(x)))),  #hide
    caption = "Counted from the one label matrix G: bit g of G[i, j] says whether group g can get from room i to room j.")  #hide

# Each label can be checked on its own: group `g` can get from `i` to `j` exactly when `j` is reachable from `i`
# through the doors that open for `g`.

all(1:8) do g
    mask = UInt32(1) << (g - 1)
    keep = (allowed .& mask) .!= 0
    Ag = sparse(src[keep], dst[keep], ones(count(keep)), rooms, rooms, min)
    Dg = mstar(MinPlus(), Ag)
    ((G .& mask) .!= 0) == isfinite.(Dg)
end

# One `OrAnd` star answers all 8 groups at once, where the check above needed one reachability computation per group.
#
# ## How Many Labels
#
# A label is a bit, so the integer type sets the number of labels: 32 with `UInt32`, 64 with `UInt64`. With a signed
# type the semiring's one is `typemax`, which leaves out the sign bit: `Int32` holds 31 labels. Prefer unsigned types.
#
# The dual semiring `AndOr` swaps the two operations: ``\oplus = \cap`` and ``\otimes = \cup``. There a path collects
# the union of its arcs' labels, and the star keeps the labels common to every path, for example the checkpoints that
# no route from `i` to `j` can avoid when each arc is labelled with the checkpoints it passes.
