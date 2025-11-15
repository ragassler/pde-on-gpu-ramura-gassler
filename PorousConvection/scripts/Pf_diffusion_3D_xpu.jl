const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
import ParallelStencil: @reset_parallel_stencil
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=false)
    @info "threads" Threads.nthreads()
end
using Plots, Plots.Measures, Printf




default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)



@parallel function compute_flux!(dQx, dQy, dQz, Pf,k_ηf_dx,k_ηf_dy, k_ηf_dz,_1_θ_dτ)
    @inn_x(dQx) = @inn_x(dQx) - (@inn_x(dQx) + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ
    @inn_y(dQy) = @inn_y(dQy) - (@inn_y(dQy) + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ
    @inn_z(dQz) = @inn_z(dQz) - (@inn_z(dQz) + k_ηf_dz * @d_za(Pf)) * _1_θ_dτ
    return nothing
end

@parallel function update_Pf!(Pf, dQx, dQy, dQz, _dx, _dy, _dz, _β_dτ)
    @inn(Pf) = @inn(Pf) - _β_dτ * ( @d_xa(dQx) * _dx + @d_ya(dQy) * _dy + @d_za(dQz) * _dz )
    return nothing
end


@parallel function compute_residual!(r_Pf, dQx, dQy, dQz, _dx, _dy, _dz)
    @inn(r_Pf) = @inn(r_Pf) + ( @d_xa(dQx) * _dx + @d_ya(dQy) * _dy + @d_za(dQz) * _dz )
    return nothing
end


function Pf_diffusion_3D_xpu(;do_viz=false)

    # physics
    lx, ly, lz = 20.0, 20.0, 20.0
    k_ηf   = 1.0
    # numerics

    nx, ny, nz  = 32, 32, 32
    ϵtol    = 1e-8
    maxiter = 500 
    ncheck  = ceil(Int, 0.25max(nx, ny, nz))
    cfl     = 1.0 / sqrt(3.1)
    re      = 2π


    # derived numerics
    dx, dy, dz  = lx / nx, ly / ny, lz / nz
    xc, yc, zc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny), LinRange(dz / 2, lz - dz / 2, nz)

    # time step parameters
    θ_dτ    = max(lx, ly, lz) / re / cfl / min(dx, dy, dz)
    β_dτ    = (re * k_ηf) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
    _β_dτ = 1.0/β_dτ
    k_ηf_dx, k_ηf_dy, k_ηf_dz = k_ηf/dx, k_ηf/dy, k_ηf/dz
    _1_θ_dτ = 1.0./(1.0 + θ_dτ)
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz
    # array initialisation
    Pf = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx, iy = 1:ny, iz = 1:nz])
    dQx     = @zeros(nx + 1, ny    , nz    )
    dQy     = @zeros(nx    , ny + 1, nz    )
    dQz     = @zeros(nx    , ny    , nz + 1)
    r_Pf    = @zeros(nx    , ny    , nz    )

        # iteration loop
    iter = 1; err_Pf = 2ϵtol
    t_tic = 0.0; niter = 0
    while err_Pf >= ϵtol && iter <= maxiter

        if (iter==11) t_tic = Base.time(); niter = 0 end
        @parallel compute_flux!(dQx, dQy, dQz, Pf,k_ηf_dx,k_ηf_dy, k_ηf_dz,_1_θ_dτ)
        @parallel update_Pf!(Pf, dQx, dQy, dQz, _dx, _dy, _dz, _β_dτ)

        if do_viz 

            if iter % ncheck == 0
                @parallel compute_residual!(r_Pf, dQx, dQy, dQz, _dx, _dy, _dz)
                err_Pf = maximum(abs.(Array(r_Pf)))
                
                @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
                display(heatmap(xc, yc, Array(Pf)'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
                
            end
        end
        iter += 1
        niter += 1

        
    end

    t_it = (Base.time() - t_tic) / niter

    A_eff = 12*nx*ny*nz * 8 / 1e9
    T_eff = A_eff / t_it
    @printf("Time = %.3e\n", t_it)
    @printf("A_eff=%1.3f GB, T_eff=%1.3f GB/s\n", A_eff, T_eff)

    return nothing
end

if isinteractive()
    Pf_diffusion_3D_xpu(;do_viz=false)
end