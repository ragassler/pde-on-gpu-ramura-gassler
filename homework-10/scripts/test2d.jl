using Test
using MAT


## fast test to check the output file
@testset "Diffusion 2D XPU performance output file test" begin
    # load the output from GPU run
    file_path = joinpath(@__DIR__, "GPU2D_out_C.mat")
    C_gpu = matread(file_path)["C"]

    # load the output from CPU run
    C_cpu = matread(joinpath(@__DIR__, "CPU2D_out_C.mat"))["C"]
    @test size(C_gpu) == size(C_cpu)
    
    # test 20 random points
    nx, ny = size(C_cpu)
    for _ in 1:20
        ix = rand(1:nx)
        iy = rand(1:ny)
        @test isapprox(C_gpu[ix, iy], C_cpu[ix, iy]; atol=1e-10)
    end
end