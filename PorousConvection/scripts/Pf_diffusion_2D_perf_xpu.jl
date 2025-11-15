const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
import ParallelStencil: @reset_parallel_stencil
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
    @info "threads" Threads.nthreads()
end
using Plots, Plots.Measures, Printf




default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)



@parallel_indices (ix, iy) function compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
    nx, ny = size(Pf)
    if (ix <= nx - 1 && iy <= ny) qDx[ix+1, iy] -= (qDx[ix+1, iy] + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ end
    if (ix <= nx && iy <= ny - 1) qDy[ix, iy+1] -= (qDy[ix, iy+1] + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ end
    return nothing
end

@parallel_indices (ix, iy) function update_Pf!(Pf, dQx, dQy, _dx, _dy, _β_dτ)
    nx, ny = size(Pf)
    if (ix <= nx && iy <= ny)
        Pf[ix, iy] -= _β_dτ * ( @d_xa(dQx) * _dx + @d_ya(dQy) * _dy )
    end
    return nothing
end


@parallel_indices (ix, iy) function compute_residual!(r_Pf, dQx, dQy, _dx, _dy)
    nx, ny = size(r_Pf)
    if (ix <= nx && iy <= ny)
        r_Pf[ix, iy] += ( @d_xa(dQx) * _dx + @d_ya(dQy) * _dy )
    end
    return nothing
end


function Pf_diffusion_2D_perf_xpu(;do_viz=false)

    # physics
    lx, ly = 20.0, 20.0
    k_ηf   = 1.0
    # numerics

    nx, ny  = 16*32, 16*32

    ϵtol    = 1e-8
    maxiter = 500 
    ncheck  = ceil(Int, 0.25max(nx, ny))
    cfl     = 1.0 / sqrt(2.1)
    re      = 2π


    # derived numerics
    dx, dy  = lx / nx, ly / ny
    xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)

    # time step parameters
    θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    _β_dτ = 1.0/β_dτ
    k_ηf_dx, k_ηf_dy = k_ηf/dx, k_ηf/dy
    _1_θ_dτ = 1.0./(1.0 + θ_dτ)
    _dx, _dy = 1.0/dx, 1.0/dy
    # array initialisation
    Pf      = Data.Array(@. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2))
    dQx     = @zeros(nx + 1, ny    )
    dQy     = @zeros(nx    , ny + 1)
    r_Pf    = @zeros(nx    , ny    )

        # iteration loop
    iter = 1; err_Pf = 2ϵtol
    t_tic = 0.0; niter = 0
    while err_Pf >= ϵtol && iter <= maxiter

        if (iter==11) t_tic = Base.time(); niter = 0 end
        @parallel compute_flux!(dQx, dQy, Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ)
        @parallel update_Pf!(Pf, dQx, dQy, _dx, _dy, _β_dτ)

        if do_viz 

            if iter % ncheck == 0
                @parallel compute_residual!(r_Pf, dQx, dQy, _dx, _dy)
                err_Pf = maximum(abs.(Array(r_Pf)))
                
                @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
                display(heatmap(xc, yc, Array(Pf)'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
                
            end
        end
        iter += 1
        niter += 1

        
    end

    t_it = (Base.time() - t_tic) / niter

    A_eff = 9*nx*ny*8 / 1e9
    T_eff = A_eff / t_it
    @printf("Time = %.3e\n", t_it)
    @printf("A_eff=%1.3f GB, T_eff=%1.3f GB/s\n", A_eff, T_eff)

    return nothing
end

if isinteractive()
    Pf_diffusion_2D_perf_xpu(;do_viz=false)
end