using Plots, Plots.Measures, Printf
using BenchmarkTools
using CUDA
using JLD2

GPU_ID = 0
device!(GPU_ID)


default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

macro d_xa(A)  esc(:( $A[ix+1,iy]-$A[ix,iy] )) end
macro d_ya(A)  esc(:( $A[ix,iy+1]-$A[ix,iy] )) end

function compute_flux!(dQx, dQy, Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    nx,ny=size(Pf)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if ix <= nx-1 && iy <= ny
        @inbounds dQx[ix+1, iy] -= (dQx[ix+1, iy] + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ
    
    end

    if ix <= nx && iy <= ny-1
        @inbounds dQy[ix, iy+1] -= (dQy[ix, iy+1] + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ
    end

    return nothing
end

function update_Pf!(Pf, dQx, dQy, _dx, _dy, _β_dτ)
    nx, ny = size(Pf)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if ix <= nx && iy <= ny
        @inbounds Pf[ix, iy] -= (@d_xa(dQx) * _dx + @d_ya(dQy) * _dy) * _β_dτ
    end
    return nothing
end

function compute!(dQx, dQy, Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ, _dx, _dy, _β_dτ)
    compute_flux!(dQx, dQy, Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
    update_Pf!(Pf, dQx, dQy, _dx, _dy, _β_dτ)
    return nothing
end

function compute_residual!(r_Pf, dQx, dQy, _dx, _dy)
    nx, ny = size(r_Pf)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if ix <= nx && iy <= ny
        @inbounds r_Pf[ix, iy] = (@d_xa(dQx) * _dx + @d_ya(dQy) * _dy)
    end
    return nothing
end


function Pf_diffusion_2D_loop_gpu(nx, ny; do_viz=false, save_Pf_at::Union{Nothing,Int}=nothing, save_path::Union{Nothing,String}=nothing)

    # physics
    lx, ly = 20.0, 20.0
    k_ηf   = 1.0
    # numerics

    ϵtol    = 1e-8
    maxiter = 500 
    ncheck  = ceil(Int, 0.25max(nx, ny))
    cfl     = 1.0 / sqrt(2.1)
    re      = 2π
    # GPU setup
    threads = (32, 4)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))



    # derived numerics
    dx, dy  = lx / nx, ly / ny
    xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)

    # move your 1D grids to the GPU (examples)
    xc_d = CuArray(xc)     # length Nx (column)
    yc_d = CuArray(yc)     # length Ny (column)
    # time step parameters
    θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    _β_dτ = 1.0/β_dτ
    k_ηf_dx, k_ηf_dy = k_ηf/dx, k_ηf/dy
    _1_θ_dτ = 1.0./(1.0 + θ_dτ)
    _dx, _dy = 1.0/dx, 1.0/dy
    # array initialisation
    Pf      = exp.(-((xc_d .- lx/2).^2) .- ((yc_d' .- ly/2).^2))
    qDx     = CUDA.zeros(Float64,nx + 1,ny)
    qDy     = CUDA.zeros(Float64, nx, ny + 1)
    r_Pf    = CUDA.zeros(Float64,nx,ny)

        # iteration loop
    iter = 1; err_Pf = 2ϵtol
    while err_Pf >= ϵtol && iter <= maxiter

        # --- save hook ---
        if save_Pf_at !== nothing && iter == save_Pf_at
            Pf_host = Array(Pf)  # bring to CPU
            path = isnothing(save_path) ? "Pf_gpu_$(save_Pf_at)_nx$(nx)_ny$(ny).jld2" : save_path
            @save path Pf_host nx ny iter
            @printf("Saved Pf at iter=%d to %s\n", iter, path)
        end
        # ------------------



        @cuda blocks=blocks threads=threads compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
        synchronize()   # grid-wide barrier

        @cuda blocks=blocks threads=threads update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
        synchronize()


        if do_viz 

            if iter % ncheck == 0
                @cuda blocks=blocks threads=threads compute_residual!(r_Pf, qDx, qDy, _dx, _dy)
                synchronize()
                err_Pf = maximum(abs.(Array(r_Pf)))
                
                @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
                display(heatmap(xc, yc, Array(Pf)'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
                
            end
        end
        iter += 1

        
    end

    t_it = @belapsed begin
        @cuda blocks=$blocks threads=$threads compute_flux!($qDx, $qDy, $Pf, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ)
        synchronize()
        @cuda blocks=$blocks threads=$threads update_Pf!($Pf, $qDx, $qDy, $_dx, $_dy, $_β_dτ)
        synchronize()
    end
    A_eff = 6*nx*ny*8 / 1e9
    T_eff = A_eff / t_it
    @printf("Time = %.3e\n", t_it)
    @printf("A_eff=%1.3f GB, T_eff=%1.3f GB/s\n", A_eff, T_eff)

    return nothing
end

## first try

## optimal resolution

nx = 127
ny = 127

Pf_diffusion_2D_loop_gpu(nx, ny; do_viz=false, save_Pf_at=60, save_path="Pf_gpu_60.jld2")


