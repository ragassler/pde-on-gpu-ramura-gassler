
using MPI
using MAT
using Printf

# 1) Initialise MPI ONCE for the whole program
MPI.Init()

# 2) Include the diffusion code (it must define diffusion_2D as modified above)
include("Diffusion_2D_perf_multipxu_benchmark.jl")

# 3) Problem sizes
nx_list = 16 .* 2 .^ (1:10)
ny_list = nx_list

T_eff = Float64[]

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("=== Strong scaling benchmark (single GPU) ===")
end

for (i, n) in pairs(nx_list)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @printf("\n[%d/%d] Running diffusion_2D with nx = ny = %d\n",
                i, length(nx_list), n)
    end

    # IMPORTANT: MPI is already initialised, so:
    #   init_MPI = false, finalize_MPI = false
    T_eff_i = diffusion_2D(n, n;
                           do_visu=false,
                           do_save=false,
                           init_MPI=false,
                           finalize_MPI=false)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @printf(" -> T_eff(%d) = %1.3f GB/s\n", n, T_eff_i)
    end
    push!(T_eff, T_eff_i)
end

# 4) Save results only on rank 0
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    file = matopen(joinpath(@__DIR__, "benchmark_strong_single_gpu.mat"), "w")
    write(file, "nx", nx_list)
    write(file, "ny", ny_list)
    write(file, "T_eff", T_eff)
    close(file)
    println("\nBenchmark data saved to benchmark_strong_single_gpu.mat")
end

# 5) Finalise MPI once at the very end
MPI.Finalize()