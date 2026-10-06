using Aqua, PATHPACK 

# FileWatching and PrecompileTools are loaded only by the GPU extension (ext/PATHPACKCUDAExt.jl)
Aqua.test_all(PATHPACK, ambiguities=false, stale_deps=(ignore=[:FileWatching, :PrecompileTools],))
