# Dense semiring kernel rates (sgemx!) on one core, against OpenBLAS dgemm on the same core; one result file per measurement.
include("lib.jl")
function bench(name, s, T, n)
    A = rand(T, n, n); B = rand(T, n, n); C = rand(T, n, n); f() = P.sgemx!(s, Val(:N), Val(:N), C, A, B; nt = 1); f()
    reps = max(1, round(Int, 2e9 / n^3)); t = tmin(() -> (for _ in 1:reps; f(); end)) / reps
    save("kernels", "$(name)-$(T)-$(n)", Dict("semiring" => name, "T" => string(T), "n" => n, "time" => t, "gmuladd" => n^3 / t / 1e9))
    @printf "%-10s %-8s n=%5d %.2f G multiply-adds/s\n" name T n n^3 / t / 1e9
end
for n in (64, 256, 1024, 2048)
    bench("MinPlus", MinPlus(), Float64, n); bench("MinPlus", MinPlus(), Float32, n); bench("MaxMin", MaxMin(), Float64, n)
    A = rand(n, n); B = rand(n, n); C = rand(n, n); mul!(C, A, B, 1.0, 1.0); reps = max(1, round(Int, 2e9 / n^3))
    t = tmin(() -> (for _ in 1:reps; mul!(C, A, B, 1.0, 1.0); end)) / reps
    A32 = rand(Float32, n, n); B32 = rand(Float32, n, n); C32 = rand(Float32, n, n); mul!(C32, A32, B32, 1f0, 1f0)
    t32 = tmin(() -> (for _ in 1:reps; mul!(C32, A32, B32, 1f0, 1f0); end)) / reps
    save("kernels", "sgemm-Float32-$(n)", Dict("semiring" => "sgemm", "T" => "Float32", "n" => n, "time" => t32, "gmuladd" => n^3 / t32 / 1e9))
    @printf "%-10s %-8s n=%5d %.2f G multiply-adds/s\n" "sgemm" Float32 n n^3 / t32 / 1e9
    save("kernels", "dgemm-Float64-$(n)", Dict("semiring" => "dgemm", "T" => "Float64", "n" => n, "time" => t, "gmuladd" => n^3 / t / 1e9))
    @printf "%-10s %-8s n=%5d %.2f G multiply-adds/s\n" "dgemm" Float64 n n^3 / t / 1e9
end
