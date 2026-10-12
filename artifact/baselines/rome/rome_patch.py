#!/usr/bin/env python3
"""Patch a ROME source tree (src/App/APSP.cu, src/APSP/rome.cu) for the benchmark runs. Usage: rome_patch.py SRC_DIR

The tree is ROME (github.com/LyleLuo/ROME, 25beef3) with fair.patch: weights read from the file instead of random
ones, ROME_LOWMEM (one n x n pinned host buffer), ROME_SKIP_CHECK, and a post-timer row hash (and, for the
instrumented build, instr.patch on top). prepare_rome.sh makes both trees.
This adds two things, neither inside any of ROME's timers (tree build, allocation and initialization, computing):

  ROME_DEVICE_ONLY=1   no n x n host buffer and no copy of the result to the host: the result stays in device memory,
                       as ours does (and the job then needs no n^2 host memory). ROME's "data transfer to host" timer
                       then times nothing. The CUDA context is created where the stock driver creates it (its first CUDA
                       call, the host buffer's allocation), before ROME's timers.
  ROME_DUMP_SAMPLE=OUT with ROME_DUMP_SAMPLE_ROWS=IN (vertex ids, 1-based, one per line): those rows of the result in
                       the original labels, read from device memory, one line per row: "v d_1 ... d_n" (%.9g, Inf),
                       for the check against the CPU's Float64 distances.
"""
import sys

src = sys.argv[1]

def patch(path, edits):
    s = open(path).read()
    for old, new in edits:
        assert s.count(old) == 1, f"{path}: anchor not found once: {old[:60]!r}"
        s = s.replace(old, new)
    open(path, "w").write(s)

patch(f"{src}/src/App/APSP.cu", [
    ('const bool lowmem = std::getenv("ROME_LOWMEM") != nullptr;',
     '// [artifact] ROME_DEVICE_ONLY=1: no host copy of the result at all (implies the low-memory mode)\n'
     '    const bool devonly = std::getenv("ROME_DEVICE_ONLY") != nullptr;\n'
     '    const bool lowmem = devonly || std::getenv("ROME_LOWMEM") != nullptr;'),
    ('DebugAssert(__FILE__, __LINE__, "cudaMallocHost(buffer) failed",',
     '// [artifact] without the host buffer, the CUDA context is created here, where the stock driver creates it (its\n'
     '    // first CUDA call, cudaMallocHost), before ROME\'s timers start\n'
     '    if (devonly) cudaFree(0);\n'
     '    else DebugAssert(__FILE__, __LINE__, "cudaMallocHost(buffer) failed",'),
])

DUMP = r'''    // [artifact] ROME_DUMP_SAMPLE=<out> with ROME_DUMP_SAMPLE_ROWS=<in: vertex ids, 1-based>: those rows of the result
    // in the ORIGINAL labels, read from device memory (also without a host copy: ROME_DEVICE_ONLY). After all timers.
    if (const char* dsp = std::getenv("ROME_DUMP_SAMPLE")) {
        const char* dsr = std::getenv("ROME_DUMP_SAMPLE_ROWS");
        const dataSize_t* sip = eTree.IPerm();
        std::vector<long> srows;
        if (FILE* fr = dsr ? std::fopen(dsr, "r") : nullptr) { long v; while (std::fscanf(fr, "%ld", &v) == 1) srows.push_back(v); std::fclose(fr); }
        std::vector<weight_t> sbuf(size);
        FILE* sf = std::fopen(dsp, "w");
        for (long v : srows) {
            if (v < 1 || v > (long)size) continue;
            CHK_CUDA_ERR(cudaMemcpy(sbuf.data(), (const char*)d_permDistance + (size_t)sip[v - 1] * pitch, size * sizeof(weight_t), cudaMemcpyDeviceToHost));
            std::fprintf(sf, "%ld", v);
            for (dataSize_t c = 0; c < size; ++c) { const weight_t x = sbuf[sip[c]]; if (std::isinf(x)) std::fprintf(sf, " Inf"); else std::fprintf(sf, " %.9g", (double)x); }
            std::fprintf(sf, "\n");
        }
        if (sf) std::fclose(sf);
        printf("[artifact] sample dump: %zu rows (original labels, from device memory) to %s\n", srows.size(), dsp);
    }
'''

rome = f"{src}/src/APSP/rome.cu"
s = open(rome).read()
edits = [
    ("CHK_CUDA_ERR(cudaMemcpy2D(", "if (permDistance != nullptr) CHK_CUDA_ERR(cudaMemcpy2D("),       # no host buffer: no copy
    ('if (const char* hp = std::getenv("ROME_ROWHASH")) {', 'if (const char* hp = permDistance ? std::getenv("ROME_ROWHASH") : nullptr) {'),
    ("    CHK_CUDA_ERR(cudaFree(d_permDistance));\n", DUMP + "    CHK_CUDA_ERR(cudaFree(d_permDistance));\n"),
]
if 'if (const char* dp = std::getenv("ROME_DUMP_ROWS_PERM")) {' in s:                                # (instrumented tree)
    edits.append(('if (const char* dp = std::getenv("ROME_DUMP_ROWS_PERM")) {', 'if (const char* dp = permDistance ? std::getenv("ROME_DUMP_ROWS_PERM") : nullptr) {'))
patch(rome, edits)
print(f"patched {src}")
