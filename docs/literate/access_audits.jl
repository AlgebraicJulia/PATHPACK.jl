# # Access Audits: Who Can Reach What
#
# A security audit asks a simple question about a system's permissions: who can actually reach what? The permission
# on one file is easy to read. Access is a property of whole routes. On a Linux file system, a user can open a file
# only if they are allowed through every directory on the way to it, and a second route to the same file, through a
# shared folder or a mount, can let in people its owner never meant to. The question is about every user and every
# file at once: an all-pairs problem.
#
# In the semiring `OrAnd`, the weight of an arc is a set of users, stored as the bits of an integer. Along a path
# the sets are intersected (``\otimes = \cap``, bitwise and): a user gets through only if every step lets them
# through. Over all paths they are united (``\oplus = \cup``, bitwise or): one open route is enough. The star `G`
# holds, for every pair `i`, `j`, the set of users who can get from `i` to `j`.
#
# !!! note "A teaching model"
#     The model below follows the rules that matter most for audits: to look up a name in a directory you need
#     execute permission (`x`) on that directory, and to read a file you need read permission (`r`) on it. It leaves
#     out access control lists, the sticky bit, listing directories (`r` on a directory) and `root`, who bypasses
#     permissions. Symbolic links add no routes, since the kernel checks the target's own path; bind mounts and hard
#     links do, and the example uses a bind mount.

# ## A File System as a Graph
#
# Five accounts, and the groups they belong to.

using PATHPACK, SparseArrays, LinearAlgebra
Base.include(@__MODULE__, joinpath(pkgdir(PATHPACK), "docs", "literate", "_figs.jl"))  #hide
using PATHPACK.CPU: OrAnd, MinPlus, mstar, mlu

users = ["alice", "bob", "carol", "dave", "www"]
members = Dict("alice" => ["alice"], "bob" => ["bob"], "carol" => ["carol"], "dave" => ["dave"],
    "www" => ["www"], "root" => String[], "lab" => ["alice", "bob", "dave"], "ops" => ["carol", "dave"])
bit(u) = UInt32(1) << (findfirst(==(u), users) - 1)
who(bits) = [u for u in users if bits & bit(u) != 0]

# The directories and files, each with an owner, a group and a mode, as `ls -l` shows them. Alice's home is private
# (`700`), so she keeps a dataset in it world-readable (`755`, `644`), trusting the home directory to protect it.
# The lab's shared folder is open to the lab group only (`770`).

fs = [
    ("/", "root", "root", 0o755),
    ("/etc", "root", "root", 0o755),
    ("/etc/passwd", "root", "root", 0o644),
    ("/etc/shadow", "root", "root", 0o600),
    ("/home", "root", "root", 0o755),
    ("/home/alice", "alice", "alice", 0o700),
    ("/home/alice/thesis.pdf", "alice", "alice", 0o600),
    ("/home/alice/datasets", "alice", "alice", 0o755),
    ("/home/alice/datasets/patients.csv", "alice", "alice", 0o644),
    ("/home/bob", "bob", "lab", 0o750),
    ("/home/bob/notes.txt", "bob", "lab", 0o640),
    ("/home/bob/.ssh", "bob", "bob", 0o700),
    ("/home/bob/.ssh/id_ed25519", "bob", "bob", 0o600),
    ("/home/carol", "carol", "carol", 0o711),
    ("/home/carol/payroll.xlsx", "carol", "ops", 0o640),
    ("/home/carol/public_html", "carol", "carol", 0o755),
    ("/home/carol/public_html/index.html", "carol", "carol", 0o644),
    ("/srv", "root", "root", 0o755),
    ("/srv/shared", "root", "lab", 0o770),
    ("/srv/shared/results.csv", "bob", "lab", 0o664),
    ("/var", "root", "root", 0o755),
    ("/var/www", "www", "www", 0o755),
    ("/var/www/config.php", "www", "www", 0o640),
]

# Then an administrator makes Alice's dataset available to the lab with a bind mount, the equivalent of
# `mount --bind /home/alice/datasets /srv/shared/datasets`. The mount point is a second name for the same directory.

mounts = ["/srv/shared/datasets" => "/home/alice/datasets"]

# Each directory and file is a vertex, and each name in a directory is an arc from the directory to it. The arc's
# label is the set of users allowed to look the name up: those with execute permission on the directory. Whether a
# user has a permission depends on which class they fall in, owner, group or other, exactly as the kernel decides.

function allowed(entry, perm)                      # perm: 4 read, 1 execute
    _, owner, group, mode = entry
    bits = UInt32(0)
    for u in users
        shift = u == owner ? 6 : u in members[group] ? 3 : 0
        (mode >> shift) & perm != 0 && (bits |= bit(u))
    end
    return bits
end

function filesystem(fs, mounts)
    id = Dict(e[1] => k for (k, e) in enumerate(fs))
    src = Int[]; dst = Int[]; label = UInt32[]
    for (path, _...) in fs
        path == "/" && continue
        p = dirname(path)
        push!(src, id[p]); push!(dst, id[path]); push!(label, allowed(fs[id[p]], 1))
    end
    for (point, target) in mounts                  # the mount point's directory leads to the mounted directory
        p = dirname(point)
        push!(src, id[p]); push!(dst, id[target]); push!(label, allowed(fs[id[p]], 1))
    end
    n = length(fs)
    return sparse(src, dst, label, n, n, |), id
end

A, id = filesystem(fs, mounts)
size(A, 1), nnz(A)    # entries, and names in directories (the mount adds one)

# ## Every Route at Once
#
# One call computes the star: for every pair of entries, the users who can get from one to the other.

G = mstar(OrAnd(), A);

# A user can read a file when they can reach it from `/` and have read permission on it.

readers(path; from = "/") = who(G[id[from], id[path]] & allowed(fs[id[path]], 4))
readers("/home/alice/thesis.pdf")

#-

readers("/home/alice/datasets/patients.csv")

# The thesis is Alice's alone, but the patient data is readable by Bob and Dave as well. Their route is through the
# lab's shared folder and the bind mount. It never passes through `/home/alice`, so the `700` on Alice's home
# directory, which she relied on, never applies.
#
# ## The Audit
#
# Write down who should be able to read each sensitive file, and compare with what the permissions allow.

intended = [
    "/etc/shadow" => String[],
    "/home/alice/thesis.pdf" => ["alice"],
    "/home/alice/datasets/patients.csv" => ["alice"],
    "/home/bob/.ssh/id_ed25519" => ["bob"],
    "/home/carol/payroll.xlsx" => ["carol", "dave"],
    "/srv/shared/results.csv" => ["alice", "bob", "dave"],
    "/var/www/config.php" => ["www"],
]
[path => setdiff(readers(path), ok) for (path, ok) in intended if !isempty(setdiff(readers(path), ok))]

#-

files = first.(intended)  #hide
can_read = [u in readers(f) for f in files, u in users]  #hide
leak = [u in readers(f) && !(u in ok) for (f, ok) in intended, u in users]  #hide
figure(panel("who can read each file", checkgrid(files, users, can_read; hi = leak)),  #hide
    caption = "Each row is one row of the star (from /), masked by the file's read permission. Red: access the policy did not intend.")  #hide

# The other files are as intended. Dave reads the payroll as a member of `ops`, and `www` reaches Carol's web page
# through her home directory: its `711` lets anyone pass through without listing it.

readers("/home/carol/public_html/index.html")

# ## Fixing It
#
# Two fixes close the leak: remove the mount, or make the dataset private where it lives. Take the second, since the
# lab may need the mount for other files, and audit again.

fixed = [e[1] == "/home/alice/datasets" ? (e[1], e[2], e[3], 0o700) : e for e in fs]
A2, _ = filesystem(fixed, mounts)
G2 = mstar(OrAnd(), A2)
who(G2[id["/"], id["/home/alice/datasets/patients.csv"]] & allowed(fixed[id["/home/alice/datasets/patients.csv"]], 4))

# Only Alice now. Every other answer is unchanged:

others = setdiff(1:length(fs), [id["/home/alice/datasets"], id["/home/alice/datasets/patients.csv"]])
G2[others, others] == G[others, others]

# ## From Anywhere, Not Only From the Root
#
# Every row of the star is a starting point. A process that is already inside a directory, its working directory,
# looks names up from there, so what it can reach depends on where it starts. The web server runs in `/var/www`:

[path for (path, _...) in fs if G[id["/var/www"], id[path]] & bit("www") != 0]

# The full matrix holds this for every directory and every user at once: which users can get from any directory
# to any other.
#
# ## Checking the Answer
#
# Each user can be checked on their own: user `u` can get from `i` to `j` exactly when `j` is reachable from `i`
# through the arcs whose labels contain `u`. Plain reachability, here as finite shortest-path distances, one
# computation per user:

all(users) do u
    I, J, L = findnz(A)
    keep = (L .& bit(u)) .!= 0
    Du = mstar(MinPlus(), sparse(I[keep], J[keep], ones(count(keep)), size(A)...))
    ((G .& bit(u)) .!= 0) == isfinite.(Du)
end

# One `OrAnd` star answers for all users at once.
#
# ## Larger Systems
#
# A real file system has millions of entries, and all ``n^2`` pairs would not fit in memory. An audit rarely needs
# them all: it needs the rows of the places users start from, such as `/`, each home directory, and the working
# directories of services. Factorize once and solve for those rows (see [Factorize Once, Query Many
# Times](queries.md)):

F = mlu(OrAnd(), A)
B = fill(UInt32(0), 1, size(A, 1)); B[id["/"]] = typemax(UInt32)
rmul!(B, F)
vec(B) == G[id["/"], :]

# A label is a bit, so `UInt32` holds 32 users and `UInt64` 64. For more, give the bits to groups instead of users,
# or compute one star per block of 64 users.
#
# ## The Same Pattern Elsewhere
#
# The model, permissions on the steps and access along whole routes, is the one behind several kinds of security
# analysis:
#
# - **Directory services.** In Active Directory, users, groups and computers form a graph whose arcs mean "can log
#   in to" or "can change the password of"; attack-path tools such as BloodHound ask which accounts can reach the
#   domain administrators.
# - **Cloud permissions.** Identities that can assume roles that can assume other roles: which of them can reach a
#   sensitive storage bucket?
# - **Networks.** Which hosts can reach which services through the firewall rules between subnets?
#
# Other semirings ask other questions of the same graph. With a cost on each step, `MinPlus` gives the cheapest route
# in; with the probability that each step succeeds, `MaxProd` gives the most likely chain. The use is defensive: to
# find the routes no one intended, and close them.
