#!/bin/bash
# Recreate the cuGraph build/run environment exactly (rapids.explicit.txt) at PREFIX. Needs micromamba or conda.
# Usage: make_rapids_env.sh PREFIX      (e.g. $B/envs/rapids, where src/sssp/cugraph/build_cugraph.sh looks for it)
set -eo pipefail
P=${1:?usage: make_rapids_env.sh PREFIX}
L=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rapids.explicit.txt
[ -d "$P/conda-meta" ] && { echo "environment already at $P"; exit 0; }
if command -v micromamba >/dev/null 2>&1; then micromamba create -y -p "$P" --file "$L"
elif command -v conda >/dev/null 2>&1; then conda create -y -p "$P" --file "$L"
else echo "need micromamba or conda on PATH"; exit 1; fi
ls "$P/lib/libcugraph.so" > /dev/null && echo "cuGraph environment at $P"
