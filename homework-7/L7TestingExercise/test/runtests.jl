# unit test for Diff function
using Test

include("../scripts/l2_diffusion_1D_test.jl")


# unit test for Diff function
@testset "Diff function tests" begin
    A = [1.0, 2.0, 4.0, 7.0]
    expected = [1.0, 2.0, 3.0]
    @test Diff(A) == expected

    B = [10.0, 5.0, 0.0]
    expected_B = [-5.0, -5.0]
    @test Diff(B) == expected_B
end



#### DUMMY REFERENCE TEST - REPLACE WITH ACTUAL REFERENCE SOLUTION ####


# get a reference solution by running the diffusion_1D function once and saving the output

C_ref, qx_ref = diffusion_1D()

# reference test on 20 random indices
using Random
Random.seed!(1234)
test_indices = rand(2:length(C_ref)-1, 20)

@testset "Diffusion 1D tests" begin
    C_test, qx_test = diffusion_1D()
    for idx in test_indices
        @test isapprox(C_test[idx], C_ref[idx]; atol=1e-8)
    end
end


