using Plots, Plots.Measures, Printf
using LoopVectorization
using Base.Threads; @info "threads" nthreads()

default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

macro d_xa(A)  esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)  esc(:( $A[ix,iy+1]-$A[ix,iy] )) end

function compute_flux!(dQx, dQy, Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    nx,ny=size(Pf)

    @tturbo for iy=1:ny, ix=1:nx-1
        @inbounds dQx[ix+1, iy] -= (dQx[ix+1, iy] + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ

    end

    @tturbo for iy=1:ny-1, ix=1:nx
        @inbounds dQy[ix, iy+1] -= (dQy[ix, iy+1] + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ
    end

    return nothing
end

function update_Pf!(Pf, dQx, dQy, _dx, _dy, _β_dτ)
    nx, ny = size(Pf)

    @tturbo for iy=1:ny, ix=1:nx
        @inbounds Pf[ix, iy] -= (@d_xa(dQx) * _dx + @d_ya(dQy) * _dy) * _β_dτ
    end

    return nothing
end

function compute!(dQx, dQy, Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ, _dx, _dy, _β_dτ)
    compute_flux!(dQx, dQy, Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    update_Pf!(Pf, dQx, dQy, _dx, _dy, _β_dτ)
end

function Pf_diffusion_2D_loop_fun(nx, ny)

    # physics
    lx, ly = 20.0, 20.0
    k_ηf   = 1.0
    # numerics

    cfl     = 1.0 / sqrt(2.1)
    re      = 2π

    # derived numerics
    dx, dy  = lx / nx, ly / ny
    xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
    θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    _β_dτ = 1.0/β_dτ
    k_ηf_dx, k_ηf_dy = k_ηf/dx, k_ηf/dy
    _1_θ_dτ = 1.0./(1.0 + θ_dτ)
    _dx, _dy = 1.0/dx, 1.0/dy
    # array initialisation
    Pf      = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
    qDx     = zeros(Float64, nx + 1, ny)
    qDy     = zeros(Float64, nx, ny + 1)
    r_Pf    = zeros(nx, ny)


        
    # Btool benchmark

    t_it = @belapsed compute!($qDx, $qDy, $Pf, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ, $_dx, $_dy, $_β_dτ)

    A_eff = 6*nx*ny*8 / 1e9
    T_eff = A_eff / t_it
    @printf("Time = %.3e\n", t_it)
    @printf("A_eff=%1.3f GB, T_eff=%1.3f GB/s\n", A_eff, T_eff)

    return T_eff
end

