const USE_GPU = false

using Printf # for formatted printing
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil

using Test
using Random

include("../scripts/PorousConvection_2D.jl")
include("../scripts/PorousConvection_2D_xpu_test.jl")
print("imported modules\n")


## unit testing for porous_convection_2D_xpu.jl ##

@testset "verifying kernel functions" begin

    # genarete Random input data
    nx, ny = 63, 31
    P     = randn(Float64, nx, ny)
    Pf_ref = copy(P)
    qDxx  = randn(Float64, nx+1, ny)
    qDyy  = randn(Float64, nx, ny+1)
    dxx   = 0.1f0
    dyy   = 0.1f0
    β_dτ_Dt = 1.0f0

    _dxx = 1.0f0/dxx
    _dyy = 1.0f0/dyy
    _β_dτ_Dt = 1.0f0/β_dτ_Dt

    # call reference kernel function
    Pf_ref = unit_update_test(Pf_ref, qDxx, qDyy, dxx, dyy, β_dτ_Dt)

    # call xpu kernel function
    Pf_xpu = unit_kernel_test(P, qDxx, qDyy, _dxx, _dyy, _β_dτ_Dt)


    # # test for 10 random indixes
     for _ in 1:10
         ix = rand(1:nx)
         iy = rand(1:ny)
         @test isapprox(Pf_xpu[ix, iy], Pf_ref[ix, iy]; rtol=1e-5, atol=1e-8)
     end


end



@testset "verifying porous_convection_xpu" begin


    T_new = porous_convection_implicit_2D_xpu(;do_viz=false, do_check=true)
    T_ref = porous_convection_implicit_2D(;do_viz=true)

    @test isapprox(T_new, T_ref; rtol=1e-5, atol=1e-8)


end

