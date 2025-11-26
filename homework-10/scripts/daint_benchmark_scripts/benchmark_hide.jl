
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
np = 64
hide_pair = (16, 4) # Modify this value as needed for different numbers of processes


@printf("\nRunning diffusion_2D with hide_pair = %s\n", string(hide_pair))
Time = diffusion_2D(nx, ny;
                           do_visu=false,
                           do_save=false,
                           init_MPI=true,
                           finalize_MPI=true)

@printf(" -> Time(%s) = %1.3f GB/s\n", string(hide_pair), Time)




# 4) Save results only on rank 0
file = matopen(joinpath(@__DIR__, "benchmark_hide_$(hide_pair)_gpu.mat"), "w")
write(file, "Time", Time)
close(file)
println("\nBenchmark data saved to benchmark_hide_$(hide_pair)_gpu.mat")

# 5) Finalise MPI once at the very end