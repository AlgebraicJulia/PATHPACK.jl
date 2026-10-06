# ===== graph capture =====
#
# A CUDA graph replays raw device pointers, so every scratch buffer that a
# recorded graph uses must live as long as the graph, even when the stream
# that owned it (and with it its entry in TRSM_WS) is gone. While capturing,
# trsm_workspace adds its buffers to CAPTURED[], and capture_graph ties them
# to the executable graph.

const CAPTURED = ScopedValue{Union{Nothing, Base.IdSet{Any}}}(nothing)

const GRAPH_SCRATCH = WeakKeyDict{CuGraphExec, Base.IdSet{Any}}()

# record f() on the current stream and instantiate it. Thread-local capture: in the default (global)
# mode, a synchronizing call from any other thread invalidates the capture (as the plans built ahead
# on another thread, on their own stream, did at random); the other threads' work is on their own streams.
function capture_graph(f)
    keep = Base.IdSet{Any}()
    graph = with(() -> CUDA.capture(f; flags = CUDA.STREAM_CAPTURE_MODE_THREAD_LOCAL), CAPTURED => keep)
    exec = CUDA.instantiate(graph)
    lock(() -> (GRAPH_SCRATCH[exec] = keep), GRAPH_SCRATCH)
    return exec
end
