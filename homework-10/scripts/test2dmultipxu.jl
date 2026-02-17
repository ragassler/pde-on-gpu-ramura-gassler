using Test
using MAT


## fast test to check the output file
@testset "Diffusion 2D single XPU and MXPU test" begin
    # load the output from GPU run
    file_path = joinpath(@__DIR__, "GPU2D_out_C.mat")
    C_gpu = matread(file_path)["C"]

    # load the output from CPU run
    C_v = matread(joinpath(@__DIR__, "MXPU2D_out_C.mat"))["C"]
    println("Sizes: GPU run C size = ", size(C_gpu), ", MXPU run C size = ", size(C_v))
    @test size(C_gpu[2:end-1, 2:end-1]) == size(C_v)
    
    # test 20 random points
    nx, ny = size(C_v)
    for _ in 1:20
        ix = rand(1:nx)
        iy = rand(1:ny)
        @test isapprox(C_gpu[ix+1, iy+1], C_v[ix, iy]; atol=1e-10)
    end
end