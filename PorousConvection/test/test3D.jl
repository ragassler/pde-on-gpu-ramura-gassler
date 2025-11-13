using Random
using Test

const USE_GPU = false
using ParallelStencil
import ParallelStencil: @reset_parallel_stencil

using Printf

include("../scripts/PorousConvection_3D_xpu_test.jl")
include("../scripts/PorousConvection_2D.jl")


## unit testing for porous_convection_3D_xpu.jl ##

@testset "verifying kernel functions" begin

    # genarete Random input data
    nx, ny, nz = 63, 31, 31
    P     = randn(Float64, nx, ny, nz)
    Pf_ref = copy(P[:, :, 1])
    qDxx  = randn(Float64, nx+1, ny, nz)
    qDyy  = randn(Float64, nx, ny+1, nz)
    qDzz  = zeros(Float64, nx, ny, nz+1)
    dxx   = 0.1f0
    dyy   = 0.1f0
    dzz   = 0.1f0
    β_dτ_Dt = 1.0f0

    _dxx = 1.0f0/dxx
    _dyy = 1.0f0/dyy
    _dzz = 1.0f0/dzz
    _β_dτ_Dt = 1.0f0/β_dτ_Dt

    # call reference kernel function
    Pf_ref = unit_update_test(Pf_ref, qDxx[:, :, 1], qDyy[:, :, 1], dxx, dyy, β_dτ_Dt)

    # call xpu kernel function
    Pf_xpu = unit_kernel_test(P, qDxx, qDyy, qDzz, _dxx, _dyy, _dzz, _β_dτ_Dt)


    # # test for 10 random indixes
     for _ in 1:10
         ix = rand(1:nx)
         iy = rand(1:ny)
         @test isapprox(Pf_xpu[ix, iy, 1], Pf_ref[ix, iy]; rtol=1e-5, atol=1e-8)
     end


end

@testset "verifying porous_convection_3D_xpu" begin

    ## reference test by introducing Ra = 10 no convection should be pure diffusion ##
    T_new, dQz_new = porous_convection_implicit_3D_xpu(;do_viz=false, do_check=true)

    ## show that no convection meaning dQz_new should be close to zero ##
    @test maximum(abs.(dQz_new)) < 1e-5

end