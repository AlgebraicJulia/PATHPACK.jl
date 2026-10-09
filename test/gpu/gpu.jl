# The GPU tests from the CPU test suite: run when an NVIDIA GPU is present (nvidia-smi lists one), or
# always with PATHPACK_TEST_GPU=true; PATHPACK_TEST_GPU=false skips them. They run in their own
# environment (test/gpu, with CUDA.jl), each file in a fresh process (test/gpu/runtests.jl).
using Test

function gpu_present()
    flag = get(ENV, "PATHPACK_TEST_GPU", "auto")
    flag == "auto" || return flag == "true"
    nvsmi = Sys.which("nvidia-smi")
    return !isnothing(nvsmi) && success(pipeline(`$nvsmi -L`; stdout = devnull, stderr = devnull))
end

if gpu_present()
    project = @__DIR__
    julia = Base.julia_cmd()
    run(`$julia --project=$project --startup-file=no -e 'using Pkg; Pkg.instantiate()'`)
    p = run(ignorestatus(`$julia --project=$project --startup-file=no $(joinpath(project, "runtests.jl")) $(get(ENV, "PATHPACK_TEST_GPU_SUITE", "quick"))`))

    if p.exitcode == 2
        @test_skip "no functional CUDA GPU"
    else
        @test success(p)
    end
else
    @test_skip "no NVIDIA GPU"
end
