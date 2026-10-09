using Test

@testset "Code Quality (Aqua.jl)" begin
  include("aqua.jl")
end

@testset "Core" begin
  include("core.jl")
end

@testset "GPU" begin
  include("gpu/gpu.jl")
end
