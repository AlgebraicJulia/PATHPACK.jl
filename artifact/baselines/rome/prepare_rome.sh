#!/bin/bash
# Make the ROME source trees and libraries the reference runs' ROME binaries were built from (hpc/setup_gpu.sh, step rome, then
# copies them, runs rome_patch.py on each copy, sets the Scotch leaf size, and builds the binaries).
#   DEST/upstream          github.com/LyleLuo/ROME at 25beef3 with its two submodules:
#                          METIS (github.com/scivision/metis 7f8f9b0, METIS 5.1.0.3) and Scotch (gitlab.inria.fr, v7.0.10)
#   DEST/src-fair          ROME's CMakeLists.txt, include/ and src/ with fair.patch               -> ROME_SRC
#   DEST/src-instr         the same with instr.patch on top (phase timers and counters only)      -> ROME_SRC_INSTR
#   DEST/modules/metis     METIS, built as ROME's README says                                     -> ROME_MODULES
#   DEST/scotch-build-O3   Scotch built with CMAKE_BUILD_TYPE=Release (-O3), otherwise with the
#                          README's options (the README's own build sets no build type: -O0)      -> ROME_SCOTCH
# Usage: prepare_rome.sh DEST [sources]   ("sources": only the two source trees; no build)
# Needs git, GNU patch; for the libraries also cmake >= 3.12, gcc, flex and bison (no GPU, no CUDA).
set -eo pipefail
D=${1:?usage: prepare_rome.sh DEST [sources]}; mkdir -p "$D"; D=$(cd "$D" && pwd)
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
J=${J:-8}
[ -d "$D/upstream/.git" ] || git clone -q https://github.com/LyleLuo/ROME "$D/upstream"
git -C "$D/upstream" checkout -q --detach 25beef3260a52725803405208dd64f76dd385f0b
git -C "$D/upstream" submodule update -q --init modules/metis modules/scotch
[ "$(git -C "$D/upstream/modules/metis" rev-parse HEAD)" = 7f8f9b0d765c15efe3eefe3e9db0a9faf9a5f480 ] || { echo "METIS submodule not at 7f8f9b0"; exit 1; }
[ "$(git -C "$D/upstream/modules/scotch" rev-parse HEAD)" = 7fd71f5eee1911b3b1c000dad1f880c0a0a41c2c ] || { echo "Scotch submodule not at v7.0.10"; exit 1; }
for t in fair instr; do
  rm -rf "$D/src-$t"; mkdir -p "$D/src-$t"
  cp -a "$D/upstream/CMakeLists.txt" "$D/upstream/include" "$D/upstream/src" "$D/src-$t/"
  ( cd "$D/src-$t" && patch -s -p1 --no-backup-if-mismatch < "$H/fair.patch" \
      && { [ $t = fair ] || patch -s -p1 --no-backup-if-mismatch < "$H/instr.patch"; } )
  echo "src-$t: ROME 25beef3 + fair.patch$( [ $t = instr ] && echo ' + instr.patch')"
done
if [ "${2:-}" != sources ]; then
  ( cd "$D/upstream/modules/metis" && mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_THREAD=Y -Dintsize=64 .. > cmake.log 2>&1 && cmake --build . -j$J > make.log 2>&1 ) \
    || { echo "METIS build failed: see $D/upstream/modules/metis/build/*.log"; exit 1; }
  mkdir -p "$D/modules" && ln -sfn "$D/upstream/modules/metis" "$D/modules/metis"
  rm -rf "$D/scotch-build-O3" && mkdir -p "$D/scotch-build-O3"
  ( cd "$D/scotch-build-O3" && cmake "$D/upstream/modules/scotch" -DCMAKE_BUILD_TYPE=Release -DBUILD_FORTRAN=OFF \
      -DBUILD_PTSCOTCH=OFF -DINTSIZE=64 -DSCOTCH_DETERMINISTIC=FULL > cmake.log 2>&1 && make -j$J scotch scotcherr > make.log 2>&1 ) \
    || { echo "Scotch build failed: see $D/scotch-build-O3/*.log"; exit 1; }
  ls "$D/modules/metis/build/libmetis.a" "$D/scotch-build-O3/lib/libscotch.a" "$D/scotch-build-O3/lib/libscotcherr.a" > /dev/null
fi
echo "for hpc/config.local.sh:"
echo "  ROME_SRC=$D/src-fair"
echo "  ROME_SRC_INSTR=$D/src-instr"
echo "  ROME_MODULES=$D/modules"
echo "  ROME_SCOTCH=$D/scotch-build-O3"
