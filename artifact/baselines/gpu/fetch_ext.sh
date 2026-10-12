#!/bin/bash
# Fetch the third-party code the GPU baselines are built from, at the exact commits of the reference runs, into EXT
# (default: ext/ next to this script). Usage: fetch_ext.sh [EXT]   (safe to rerun; nothing here is modified in place:
# the build scripts copy and patch what they need)
set -eo pipefail
EXT=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ext}
mkdir -p "$EXT"
pin() {   # pin NAME URL COMMIT
  local d=$EXT/$1
  [ -d "$d/.git" ] || git clone -q "$2" "$d"
  git -C "$d" cat-file -e "$3^{commit}" 2>/dev/null || git -C "$d" fetch -q origin "$3"
  git -C "$d" checkout -q --detach "$3"
  printf "  %-14s %s %s\n" "$1" "$(git -C "$d" rev-parse --short=12 HEAD)" "$2"
}
echo "third-party code -> $EXT"
pin FastAPSP      https://github.com/Liu-xiandong/FastAPSP.git eec2af2b8939e6cbdc10ae6bbfabfa6d1231a0d3   # Fast-APSP (Yang et al., ICS'23)
pin adds          https://github.com/UCLA-VAST/adds.git        9ef8166e87d3f13cb458ce2f128bc862e841200b   # ADDS (PPoPP'21), with its CUB 1.8.0
pin gunrock       https://github.com/gunrock/gunrock.git       748f79e66cc26762cd3f99422b028e742a737bde   # Gunrock essentials (v2.2.0-13)
pin cxxopts       https://github.com/jarro2783/cxxopts.git     c74846a891b3cc3bfa992d588b1295f528d43039   # v3.0.0 (Gunrock's)
pin nlohmann_json https://github.com/nlohmann/json.git         9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03   # v3.11.3 (Gunrock's)
pin cuASR         https://github.com/hpcgarage/cuASR.git       45b34c7631d1324bf15a6fd907063f2334f8ed88   # cuASR (fwbench's semiring GEMM)
git -C "$EXT/cuASR" submodule update -q --init cutlass                                               # CUTLASS v2.3.0 (c2b80ad4), cuASR's pin
printf "  %-14s %s (cuASR's submodule)\n" cutlass "$(git -C "$EXT/cuASR/cutlass" rev-parse --short=12 HEAD)"
# ECL-APSP v1.0 has no repository: the authors' page, checked by SHA-1
E=$EXT/ECL-APSP; mkdir -p "$E"
U=https://userweb.cs.txstate.edu/~burtscher/research/ECL-APSP
for f in ECL-APSP_10.cu ECLgraph.h; do [ -s "$E/$f" ] || curl -fsSL --retry 3 -o "$E/$f" "$U/$f"; done
( cd "$E" && sha1sum -c --quiet <<SUMS
2263a1c30b4281621d8c366e9388793d8aee44e8  ECL-APSP_10.cu
30edbf534cb778982fdadf848d88c0d8bd1c68a0  ECLgraph.h
SUMS
) && printf "  %-14s %s\n" ECL-APSP "v1.0 ($U), SHA-1 checked"
