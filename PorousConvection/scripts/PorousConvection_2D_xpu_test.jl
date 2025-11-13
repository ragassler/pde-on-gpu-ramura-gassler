
using ParallelStencil
using ParallelStencil.FiniteDifferences2D


@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=true)
    CUDA.allowscalar(false)  # catches accidental CPU-style indexing on CuArrays
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=true)
    @info "threads" Threads.nthreads()
end


@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])


#------------------------------------------------------------------------------#
#-------NOTE: Same as _3D_xpp.jl but low resolution for testing----------------#
#------------------------------------------------------------------------------#




@parallel_indices (ix, iy) function compute_flux!(dQx, dQy, Pf, k_ηf, _1_θ_dτ, αρgx, αρgy, T, _dx, _dy)
    nx, ny = size(Pf)
    if (ix <= nx - 1 && iy <= ny) dQx[ix+1, iy] -= (dQx[ix+1, iy] + k_ηf * (_dx * (Pf[ix+1, iy] - Pf[ix, iy]) - αρgx * 0.5*(T[ix+1, iy] + T[ix, iy]))) * _1_θ_dτ  end
    if (ix <= nx && iy <= ny - 1) dQy[ix, iy+1] -= (dQy[ix, iy+1] + k_ηf * (_dy * (Pf[ix, iy+1] - Pf[ix, iy]) - αρgy * 0.5*(T[ix, iy+1] + T[ix, iy]))) * _1_θ_dτ end
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

@parallel_indices (ix, iy) function compute_temp_flux!(dQx, dQy, T, λ_ρCp, _1_θ_dτ, _dx, _dy)
    nxx,nyx = size(dQx)

    if (ix <= nxx && iy <= nyx)
        dQx[ix, iy] -= (dQx[ix, iy] + λ_ρCp * _dx * (T[ix+1, iy+1] - T[ix, iy+1])) * _1_θ_dτ 
    end

    nxy,nyy = size(dQy)
    if (ix <= nxy && iy <= nyy)
        dQy[ix, iy] -= (dQy[ix, iy] + λ_ρCp * _dy * (T[ix+1, iy+1] - T[ix+1, iy])) * _1_θ_dτ 
    end
    return nothing
end

@parallel_indices (ix, iy) function compute_dTdt!(dTdt, T, T_old, _dt, qDx, qDy, _ϕ, _dx, _dy)
    nx, ny = size(dTdt)
    if (ix <= nx && iy <= ny)
        dTdt[ix, iy] = (T[ix+1, iy+1] - T_old[ix+1, iy+1]) * _dt +
                       (max(qDx[ix+1, iy+1], 0.0) * (T[ix+1, iy+1] - T[ix, iy+1])* _dx +
                        min(qDx[ix+2, iy+1], 0.0) * (T[ix+2, iy+1] - T[ix+1, iy+1]) * _dx +
                        max(qDy[ix+1, iy+1], 0.0) * (T[ix+1, iy+1] - T[ix+1, iy])* _dy +
                        min(qDy[ix+1, iy+2], 0.0) * (T[ix+1, iy+2] - T[ix+1, iy+1]) * _dy) * _ϕ
    end
    return nothing
end

@parallel_indices (ix, iy) function update_T!(T, dTdt, dQx, dQy, _dx, _dy, _temp)
    nx, ny = size(dTdt)
    if (ix <= nx && iy <= ny)
        T[ix+1, iy+1] -= (dTdt[ix, iy] + ( @d_xa(dQx) * _dx ) + ( @d_ya(dQy) * _dy )) * _temp
    end
    return nothing
end

@parallel_indices (ix, iy) function compute_residual_T!(r_T, dTdt, dQx, dQy, _dx, _dy)
    nx, ny = size(r_T)
    if (ix <= nx && iy <= ny)
        r_T[ix, iy] += dTdt[ix, iy] + ( @d_xa(dQx) * _dx ) + ( @d_ya(dQy) * _dy )
    end
    return nothing
end

@parallel_indices (j) function bc_T_sides!(T)
    ny = size(T,2)
    if j<=ny
        T[1,  j] = T[2,     j]
        T[end,j] = T[end-1, j]
    end
    return nothing
end



@views function porous_convection_implicit_2D_xpu(;do_viz=false, do_check=false)
    # physics
    lx, ly     = 40.0, 20.0
    k_ηf       = 1.0
    αρgx, αρgy = 0.0, 1.0
    αρg        = sqrt(αρgx^2 + αρgy^2)
    ΔT         = 200.0
    ϕ          = 0.1
    _ϕ         = 1.0 / ϕ
    Ra         = 1000
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ

    # numerics

    ## nummerics for quick testing on CPU ##
    ny         = 31
    nx         = 2 * (ny + 1) - 1
    nt         = 50
    ## ---------------------------------- ##

    re_D       = 4π
    cfl        = 1.0 / sqrt(2.1)
    maxiter    = 10max(nx, ny)
    ϵtol       = 1e-6
    nvis       = 50
    ncheck     = ceil(2max(nx, ny))
        
    # derived numerics
    dx      = lx / nx
    dy      = ly / ny
    xc      = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc      = LinRange(-ly + dy / 2, - dy / 2, ny)
    cfl     = 1.0 / sqrt(2.1)
    dtd     = min(dx, dy)^2 / λ_ρCp / 4.1

    # pressure PT
    re_D    = 4π
    θ_dτ_D  = max(lx,ly) / re_D / (cfl * min(dx, dy))
    β_dτ_D  = k_ηf * re_D / (cfl * min(dx, dy) * max(lx, ly))

    # divison operators
    _β_dτ_D = 1.0 / β_dτ_D
    _1_θ_dτ_D = 1.0./(1.0 + θ_dτ_D)
    _dx, _dy = 1.0/dx, 1.0/dy

    # array initialisation
    # temperature
    T        = Data.Array(@. ΔT * exp(-xc^2 - (yc' + ly / 2)^2))
    T[:, 1] .= ΔT / 2; T[:, end] .= -ΔT / 2
    T_old    = Data.Array(copy(Array(T)))
    dTdt     = @zeros(nx - 2, ny - 2)
    r_T      = @zeros(nx - 2, ny - 2)
    qTx      = @zeros(nx - 1, ny - 2)
    qTy      = @zeros(nx - 2, ny - 1)

    # pressure
    Pf      = @zeros(nx, ny)
    r_Pf    = @zeros(nx, ny)
    qDx     = @zeros(nx + 1, ny)
    qDy     = @zeros(nx, ny + 1)
    qDx_c   = @zeros(nx, ny)
    qDy_c   = @zeros(nx, ny)



    t_it_avg = 0.0


        # time loop
        for it in 1:nt


            T_old .= T

            dt = if it == 1
                0.1 * min(dx, dy) / (αρg * ΔT * k_ηf)
            else
                min(5.0 * min(dx, dy) / (αρg * ΔT * k_ηf), ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1)
            end
            _dt = 1.0/dt
            re_T    = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
            θ_dτ_T  = max(lx, ly) / re_T / cfl / min(dx, dy)
            _1_θ_dτ_T = 1.0 / (1.0 + θ_dτ_T)
            β_dτ_T  = (re_T * λ_ρCp) / (cfl * min(dx, dy) * max(lx, ly))
            _tmp    = 1.0/(_dt + β_dτ_T)

            # iteration loop
            iter = 1; err_D = 2ϵtol; err_T = 2ϵtol
            t_tic = 0.0; niter = 0
            while max(err_D, err_T) >= ϵtol && iter <= maxiter


                if (iter==11) t_tic = Base.time(); niter = 0 end
                # pressure
                @parallel compute_flux!(qDx, qDy, Pf, k_ηf, _1_θ_dτ_D, αρgx, αρgy, T, _dx, _dy)
                @parallel update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ_D)

                
                # temperature

                @parallel compute_temp_flux!(qTx, qTy, T, λ_ρCp, _1_θ_dτ_T, _dx, _dy)
                @parallel compute_dTdt!(dTdt, T, T_old, _dt, qDx, qDy, _ϕ, _dx, _dy)   
                @parallel update_T!(T, dTdt, qTx, qTy, _dx, _dy, _tmp)

                # boundary conditions adiabatic
                @parallel bc_T_sides!(T)

                if do_check
                    if iter % ncheck == 0
                        fill!(r_T, 0.0)
                        fill!(r_Pf, 0.0)
                        @parallel compute_residual_T!(r_T, dTdt, qTx, qTy, _dx, _dy)
                        @parallel compute_residual!(r_Pf, qDx, qDy, _dx, _dy)

                        err_T = maximum(abs.(r_T))
                        err_D = maximum(abs.(r_Pf))
                        # if do_viz
                        #     @printf("it = %d,  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", it, iter / nx, err_D, err_T)
                        # end

                    end
                end

                iter += 1
                niter += 1
            end

            t_it = (Base.time() - t_tic) / niter
            t_it_avg += t_it

        end
    

    A_eff = 28*nx*ny*8 / 1e9
    T_eff = A_eff / t_it_avg
    @printf("Time = %.3e\n", t_it_avg)
    @printf("A_eff=%1.3f GB, T_eff=%1.3f GB/s\n", A_eff, T_eff)

    ## -------- return nothing if Run GPU -------##

    return T

    end

## for unit testing ##

@views function unit_kernel_test(P, dQxx, dQyy, _dxx, _dyy, _β_dτt)

    ## from CPU array to Data.Array
    P_d = Data.Array(P)
    dQx_d = Data.Array(dQxx)
    dQy_d = Data.Array(dQyy)

    # test update_Pf! kernel
    @parallel update_Pf!(P_d, dQx_d, dQy_d, _dxx, _dyy, _β_dτt)

    return Array(P_d)
end
