# What the GPU benchmark scripts share: PATHPACK with its GPU backend, and the graphs (Matrix Market files
# in $PATHPACK_MTX, default benchmark/gpu/data/mtx; undirected, Float32 weights).
using PATHPACK, CUDA, SparseArrays

const SemiringGPU = Base.get_extension(PATHPACK, :PATHPACKCUDAExt)
const T = Float32
const MTX = get(ENV, "PATHPACK_MTX", joinpath(@__DIR__, "data", "mtx"))

function read_mtx(path)
    I = Int[]; J = Int[]; V = T[]; n = 0; header = true
    for line in eachline(path)
        startswith(line, '%') && continue
        a = split(line)
        if header
            n = parse(Int, a[1]); header = false
        else
            push!(I, parse(Int, a[1])); push!(J, parse(Int, a[2])); push!(V, parse(T, a[3]))
        end
    end
    return sparse(I, J, V, n, n, min)
end
