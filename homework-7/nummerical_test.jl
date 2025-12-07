# Numerical Test for Pressure Field
using JLD2
using Test
using Printf


function load_Pf_from_file(path::String)
    @load path Pf_host nx ny iter
    @printf("Loaded Pf from %s (nx=%d, ny=%d, iter=%d)\n", path, nx, ny, iter)
    return Pf_host
end

## Test using Base.Test

function test_Pf_diffusion_2D_loop_cpu_gpu()

    # Load saved results
    Pf_cpu_loaded = load_Pf_from_file("test_data/Pf_cpu_60.jld2")
    Pf_gpu_loaded = load_Pf_from_file("test_data/Pf_gpu_60.jld2")

    # Compare results
    @testset "Pressure field test" begin
        Res = Pf_cpu_loaded .- Pf_gpu_loaded
        max_diff = maximum(abs.(Res))
        @printf("Maximum difference between CPU and GPU results: %1.3e\n",
                max_diff)
        @test max_diff < 1e-6

    end


    @printf("Test passed: CPU and GPU results are approximately equal.\n")
end

# Run the test
test_Pf_diffusion_2D_loop_cpu_gpu()