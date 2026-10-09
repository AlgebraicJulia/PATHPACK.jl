# what the CALLs and local-memory accesses in a few kernels are (context lines around them)
include(joinpath(@__DIR__, "setup.jl"))
using .Ext
using .CPU: MinPlus
using CUDA
s = MinPlus(); A = CUDA.ones(Float32, 512, 512)
for (name, f) in (("v7 128x64", () -> Ext.launch7!(s, A, A, A, Val(128), Val(64), Val(8), Val(false); tn = 8)),
                  ("rows 64x128 offset", () -> Ext.rows_gemm!(s, view(A, :, 1:128), view(A, 1:256, 129:200), view(A, 1:72, 1:128), 256, 3, nothing)))
    io = IOBuffer(); CUDA.@device_code_sass io = io f(); L = split(String(take!(io)), '\n')
    println("=== ", name, ": ", length(L), " lines; registers in header: ", filter(l -> occursin("REG", uppercase(l)) && occursin("//", l), L)[1:min(end, 2)])
    for (i, l) in enumerate(L)
        if occursin(r"\b(CALL|STL|LDL|RET)\b", l)
            println("  ", strip(l))
        end
    end
    println("  labels/functions: ", unique(filter(l -> occursin(r"^\s*\.?[A-Za-z_$][\w$.]*:\s*$", l) || occursin(".text.", l), L))[1:min(end, 12)])
end
