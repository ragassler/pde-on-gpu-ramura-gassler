
using MPI
using MAT
using Printf


# Include the diffusion code (it must define diffusion_2D as modified above)
include("Diffusion_2D_perf_multipxu_benchmark.jl")

# 2) Problem size from strong scaling
nx = 16384
ny = 16384

# 3) Run with different numbers of processes
# np = 1,4,16,25,64.
np = 25 # Modify this value as needed for different numbers of processes

@printf("\nRunning diffusion_2D with np = %d\n", np)
T_eff = diffusion_2D(nx, ny;
                           do_visu=false,
                           do_save=false,
                           init_MPI=true,
                           finalize_MPI=true)

@printf(" -> T_eff(%d) = %1.3f GB/s\n", np, T_eff)




# 4) Save results only on rank 0
file = matopen(joinpath(@__DIR__, "benchmark_weak_$(np)_gpu.mat"), "w")
write(file, "T_eff", T_eff)
close(file)
println("\nBenchmark data saved to benchmark_weak_$(np)_gpu.mat")

# 5) Finalise MPI once at the very end