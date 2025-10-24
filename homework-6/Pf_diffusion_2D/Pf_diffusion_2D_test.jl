using Plots, Plots.Measures, Printf, Test
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

function Pf_diffusion_2D(nx, ny; do_viz=false)
    # benchmark 
    t_tock = 0.0
    t_start = 0.0


    # physics
    lx, ly = 20.0, 20.0
    k_ηf   = 1.0
    # numerics

    ϵtol    = 1e-8
    maxiter = 500 
    ncheck  = ceil(Int, 0.25max(nx, ny))
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

    xtest = [5, Int(cld(0.6*lx, dx)), nx-10]
    ytest = Int(cld(0.5*ly, dy))

    # iteration loop
    iter = 1; err_Pf = 2ϵtol
    while err_Pf >= ϵtol && iter <= maxiter
        if iter == 11
            t_start = time()
        end


        compute!(qDx, qDy, Pf,k_ηf_dx,k_ηf_dy,_1_θ_dτ, _dx, _dy, _β_dτ)

        if do_viz 

            if iter % ncheck == 0
                for iy=1:ny
                    for ix=1:nx
                        r_Pf[ix, iy] = (qDx[ix+1, iy]-qDx[ix, iy]) * _dx + (qDy[ix, iy+1]-qDy[ix, iy]) * _dy
                    end
                end
                err_Pf = maximum(abs.(r_Pf))
                
                @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
                display(heatmap(xc, yc, Pf'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
                
            end
        end
        iter += 1

        
    end
    t_tock = time() - t_start
    niter = iter -11
    t_it = t_tock/niter
    A_eff = 6*nx*ny*8 / 1e9
    T_eff = A_eff / t_it
    @printf("Time = %.3e, niter = %d\n", t_tock, niter)
    @printf("A_eff=%1.3f GB, T_eff=%1.3f GB/s\n", A_eff, T_eff)

    return Pf[xtest, ytest]
end


### Testing nummerical correctness  with test Set ####

nx = ny = 16 * 2 .^ (2:5) .- 1

# Results from lecture
Pf_n_test = zeros(4,3)

Pf_n_test[1, :] = [0.00785398056115133 0.007853980637555755 0.007853978592411982]
Pf_n_test[2, :] = [0.00787296974549236 0.007849556884184108 0.007847181374079883]
Pf_n_test[3, :] = [0.00740912103848251 0.009143711648167267 0.007419533048751209]
Pf_n_test[4, :] = [0.00566813765849919 0.004348785338575644 0.005618691590498087]

############################----Task----#######################################


### Testing with a test set for the different resolutions ###

@testset "Pressure field test" begin

    for i=1:length(nx)
        @test Pf_diffusion_2D(nx[i], ny[i])[1] ≈ Pf_n_test[i, 1]
        @test Pf_diffusion_2D(nx[i], ny[i])[2] ≈ Pf_n_test[i, 2]
        @test Pf_diffusion_2D(nx[i], ny[i])[3] ≈ Pf_n_test[i, 3] 
    end
end


