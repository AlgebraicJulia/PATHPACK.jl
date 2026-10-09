# Loads PATHPACK with its GPU backend for the GPU tests: Ext is the extension module (PATHPACKCUDAExt),
# CPU is PATHPACK.CPU, the reference.
using PATHPACK, CUDA
using PATHPACK: CPU

const Ext = Base.get_extension(PATHPACK, :PATHPACKCUDAExt)
