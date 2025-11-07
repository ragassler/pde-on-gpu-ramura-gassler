using JLD2

include("Pf_diffusion_2D_loop_gpu_teff.jl")
include("memcopy_triad.jl")


function benchmark_all()
    println("Starting benchmarks...")


    # Benchmark parameters

    nx = ny = 32 .* 2 .^ (0:10) .- 1

    results_memcopy_triad = Float64[]
    results_Pf_diffusion_2D_loop_gpu = Float64[]

    for N in nx
        println("Benchmarking for N = $N")
        t_memcopy_triad = memcopy_triad_gpu(N, N)
        push!(results_memcopy_triad, t_memcopy_triad)
        println("  memcopy_triad: $t_memcopy_triad GB/s")
        t_Pf_diffusion_2D_loop_gpu = Pf_diffusion_2D_loop_gpu(N, N)
        push!(results_Pf_diffusion_2D_loop_gpu, t_Pf_diffusion_2D_loop_gpu)
        println("  Pf_diffusion_2D_loop_gpu: $t_Pf_diffusion_2D_loop_gpu GB/s")
    end

    @save "benchmark_results.jld2" nx results_memcopy_triad results_Pf_diffusion_2D_loop_gpu
end

benchmark_all()
