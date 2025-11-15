const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
    CUDA.allowscalar(false)  # catches accidental CPU-style indexing on CuArrays
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
    @info "threads" Threads.nthreads()
end

using Plots, Plots.Measures, Printf



@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])


#----------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------3D kernel functions: for parallelization ------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------------#

# compute fluxes with parallel indices

@parallel_indices (ix, iy, iz) function compute_flux!(dQx, dQy, dQz, Pf, k_ηf, _1_θ_dτ, g, T, _dx, _dy, _dz)
    nx, ny, nz = size(Pf)
    if (ix <= nx - 1 && iy <= ny && iz <= nz) dQx[ix+1, iy, iz] -= (dQx[ix+1, iy, iz] + k_ηf * (_dx * (Pf[ix+1, iy, iz] - Pf[ix, iy, iz]))) * _1_θ_dτ  end
    if (ix <= nx && iy <= ny - 1 && iz <= nz) dQy[ix, iy+1, iz] -= (dQy[ix, iy+1, iz] + k_ηf * (_dy * (Pf[ix, iy+1, iz] - Pf[ix, iy, iz]))) * _1_θ_dτ  end
    if (ix <= nx && iy <= ny && iz <= nz - 1) dQz[ix, iy, iz+1] -= (dQz[ix, iy, iz+1] + k_ηf * (_dz * (Pf[ix, iy, iz+1] - Pf[ix, iy, iz]) - g * 0.5*(T[ix, iy, iz+1] + T[ix, iy, iz]))) * _1_θ_dτ  end
    return nothing
end

# parallel update of Pf

@parallel function update_Pf!(Pf, dQx, dQy, dQz, _dx, _dy, _dz, _β_dτ)

    @all(Pf) = @all(Pf) - _β_dτ * ( @d_xa(dQx) * _dx + @d_ya(dQy) * _dy + @d_za(dQz) * _dz )
    return nothing
end

# parallel compute of residual

@parallel function compute_residual!(r_Pf, dQx, dQy, dQz, _dx, _dy, _dz)
    @all(r_Pf) = @all(r_Pf) + ( @d_xa(dQx) * _dx + @d_ya(dQy) * _dy + @d_za(dQz) * _dz )
    return nothing
end


# compute temperature fluxes with parallel indices

@parallel_indices (ix, iy, iz) function compute_temp_flux!(dQx, dQy, dQz, T, λ_ρCp, _1_θ_dτ, _dx, _dy, _dz)
    nxx, nyx, nzx = size(dQx)

    if (ix <= nxx && iy <= nyx && iz <= nzx)
        dQx[ix, iy, iz] -= (dQx[ix, iy, iz] + λ_ρCp * _dx * (T[ix+1, iy+1, iz+1] - T[ix, iy+1, iz+1])) * _1_θ_dτ 
    end

    nxy,nyy, nzy = size(dQy)
    if (ix <= nxy && iy <= nyy && iz <= nzy)
        dQy[ix, iy, iz] -= (dQy[ix, iy, iz] + λ_ρCp * _dy * (T[ix+1, iy+1, iz+1] - T[ix+1, iy, iz+1])) * _1_θ_dτ 
    end

    nxz, nyz, nzz = size(dQz)
    if (ix <= nxz && iy <= nyz && iz <= nzz)
        dQz[ix, iy, iz] -= (dQz[ix, iy, iz] + λ_ρCp * _dz * (T[ix+1, iy+1, iz+1] - T[ix+1, iy+1, iz])) * _1_θ_dτ 
    end
    return nothing
end


# compute dTdt with parallel indices

@parallel_indices (ix, iy, iz) function compute_dTdt!(dTdt, T, T_old, _dt, qDx, qDy, qDz, _ϕ, _dx, _dy, _dz)
    nx, ny, nz = size(dTdt)
    if (ix <= nx && iy <= ny && iz <= nz)
        dTdt[ix, iy, iz] = (T[ix+1, iy+1, iz+1] - T_old[ix+1, iy+1, iz+1]) * _dt +
                       (max(qDx[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix, iy+1, iz+1])* _dx +
                        min(qDx[ix+2, iy+1, iz+1], 0.0) * (T[ix+2, iy+1, iz+1] - T[ix+1, iy+1, iz+1]) * _dx +
                        max(qDy[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix+1, iy, iz+1])* _dy +
                        min(qDy[ix+1, iy+2, iz+1], 0.0) * (T[ix+1, iy+2, iz+1] - T[ix+1, iy+1, iz+1]) * _dy +
                        max(qDz[ix+1, iy+1, iz+1], 0.0) * (T[ix+1, iy+1, iz+1] - T[ix+1, iy+1, iz])* _dz +
                        min(qDz[ix+1, iy+1, iz+2], 0.0) * (T[ix+1, iy+1, iz+2] - T[ix+1, iy+1, iz+1]) * _dz
                        ) * _ϕ
    end
    return nothing
end


# compute temperature update with parallel indices

@parallel_indices (ix, iy, iz) function update_T!(T, dTdt, dQx, dQy, dQz, _dx, _dy, _dz, _temp)
    nx, ny, nz = size(dTdt)
    if (ix <= nx && iy <= ny && iz <= nz)
        T[ix+1, iy+1, iz+1] -= (dTdt[ix, iy, iz] + ( @d_xa(dQx) * _dx ) + ( @d_ya(dQy) * _dy ) + ( @d_za(dQz) * _dz )) * _temp
    end
    return nothing
end

# compute temperature residual with parallel indices would be possible also with @parallel

@parallel_indices (ix, iy, iz) function compute_residual_T!(r_T, dTdt, dQx, dQy, dQz, _dx, _dy, _dz)
    nx, ny, nz = size(r_T)
    if (ix <= nx && iy <= ny && iz <= nz)
        r_T[ix, iy, iz] += dTdt[ix, iy, iz] + ( @d_xa(dQx) * _dx ) + ( @d_ya(dQy) * _dy ) + ( @d_za(dQz) * _dz )
    end
    return nothing
end

# boundary conditions

@parallel_indices (iy, iz) function bc_x!(A)
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return nothing
end

@parallel_indices (ix, iz) function bc_y!(A)
    A[ix  , 1, iz] = A[ix, 2    , iz]
    A[ix, end, iz] = A[ix, end-1, iz]
    return
end

@parallel_indices (ix, iy) function bc_z_T_dirichlet!(T, Thot, Tcold)
    T[ix, iy, 1  ] = Thot
    T[ix, iy, end] = Tcold
    return nothing
end


##---------------------------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------- POROUS CONVECTION 3D IMPLICIT SOLVER FUNCTION --------------------------------------------##
##---------------------------------------------------------------------------------------------------------------------------------------------##

@views function porous_convection_implicit_3D_xpu(;do_viz=false, do_check=false)
    # physics
    lx, ly, lz = 40.0, 20.0, 20.0
    k_ηf       = 1.0
    αρg        = 1.0
    ΔT         = 200.0
    ϕ          = 0.1
    _ϕ         = 1.0 / ϕ
    Ra         = 1000
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * lz / ϕ) # Ra = αρg*k_ηf*ΔT*lz/λ_ρCp/ϕ

    # numerics

    ## -------------------------------------------------------------------------------------------##
    ## ------------------ SIMULATION PARAMETERS choose wisely nx, ny, nz and nt ------------------##
    nx, ny, nz = 255, 127, 127
    nt         = 2000
    re_D       = 4π
    cfl        = 1.0 / sqrt(3.1)
    maxiter    = 10max(nx, ny)
    ϵtol       = 1e-6
    nvis       = 50
    ncheck     = ceil(2max(nx, ny, nz))
    ## -------------------------------------------------------------------------------------------##

        
    # derived numerics
    dx      = lx / nx
    dy      = ly / ny
    dz      = lz / nz
    xc      = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc      = LinRange(-ly + dy / 2, - dy / 2, ny)
    zc      = LinRange(-lz + dz / 2, - dz / 2, nz)


    # pressure PT
    re_D    = 4π
    θ_dτ_D  = max(lx,ly,lz) / re_D / (cfl * min(dx, dy, dz))
    β_dτ_D  = k_ηf * re_D / (cfl * min(dx, dy, dz) * max(lx, ly, lz))

    # divison operators
    _β_dτ_D = 1.0 / β_dτ_D
    _1_θ_dτ_D = 1.0./(1.0 + θ_dτ_D)
    _dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz

    # array initialisation
    # temperature
    T        = [ΔT * exp(-xc[ix]^2 - yc[iy]^2 - (zc[iz] + lz / 2)^2) for ix = 1:nx, iy = 1:ny, iz = 1:nz]
    T        = Data.Array(T)
    @parallel (1:size(T, 1), 1:size(T, 2)) bc_z_T_dirichlet!(T, ΔT / 2, -ΔT / 2)
    @parallel (1:size(T, 2), 1:size(T, 3)) bc_x!(T)
    @parallel (1:size(T, 1), 1:size(T, 3)) bc_y!(T)
    T_old    = Data.Array(copy(Array(T)))
    dTdt     = @zeros(nx - 2, ny - 2, nz - 2)
    r_T      = @zeros(nx - 2, ny - 2, nz - 2)
    qTx      = @zeros(nx - 1, ny - 2, nz - 2)
    qTy      = @zeros(nx - 2, ny - 1, nz - 2)
    qTz      = @zeros(nx - 2, ny - 2, nz - 1)

    # pressure
    Pf      = @zeros(nx, ny, nz)
    r_Pf    = @zeros(nx, ny, nz)
    qDx     = @zeros(nx + 1, ny, nz)
    qDy     = @zeros(nx, ny + 1, nz)
    qDz     = @zeros(nx, ny, nz + 1)
    qDx_c   = @zeros(nx, ny, nz)
    qDy_c   = @zeros(nx, ny, nz)
    qDz_c   = @zeros(nx, ny, nz)


    iframe = 0


        # time loop
        for it in 1:nt

            

            # print progress
            if it % 50 == 0
                @printf("Time step %d / %d (%.1f%%)\n", it, nt, it / nt * 100)
            end


            T_old .= T

            dt = if it == 1
                0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)
            else
                min(5.0 * min(dx, dy, dz) / (αρg * ΔT * k_ηf), ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy)), dz / maximum(abs.(qDz))) / 3.1)
            end
            _dt = 1.0/dt
            re_T    = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
            θ_dτ_T  = max(lx, ly, lz) / re_T / cfl / min(dx, dy, dz)
            _1_θ_dτ_T = 1.0 / (1.0 + θ_dτ_T)
            β_dτ_T  = (re_T * λ_ρCp) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
            _tmp    = 1.0/(_dt + β_dτ_T)

            # iteration loop
            iter = 1; err_D = 2ϵtol; err_T = 2ϵtol
            while max(err_D, err_T) >= ϵtol && iter <= maxiter

                # pressure
                @parallel compute_flux!(qDx, qDy, qDz, Pf, k_ηf, _1_θ_dτ_D, αρg, T, _dx, _dy, _dz)
                @parallel update_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)

                
                # temperature

                @parallel compute_temp_flux!(qTx, qTy, qTz, T, λ_ρCp, _1_θ_dτ_T, _dx, _dy, _dz)
                @parallel compute_dTdt!(dTdt, T, T_old, _dt, qDx, qDy, qDz, _ϕ, _dx, _dy, _dz)   
                @parallel update_T!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _tmp)

                # boundary conditions adiabatic
                @parallel (1:size(T, 2), 1:size(T, 3)) bc_x!(T)
                @parallel (1:size(T, 1), 1:size(T, 3)) bc_y!(T)

                if do_check
                    if iter % ncheck == 0
                        fill!(r_T, 0.0)
                        fill!(r_Pf, 0.0)
                        @parallel compute_residual_T!(r_T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz)
                        @parallel compute_residual!(r_Pf, qDx, qDy, qDz, _dx, _dy, _dz)

                        err_T = maximum(abs.(r_T))
                        err_D = maximum(abs.(r_Pf))
                        if do_viz && USE_GPU==false
                            @printf("it = %d,  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", it, iter / nx, err_D, err_T)
                        end

                    end
                end

                iter += 1
            end

            
            # visualisation
            if do_viz && (it % nvis == 0)
                p1 = heatmap(xc, zc, Array(T)[:, ceil(Int, ny / 2), :]'; xlims=(xc[1], xc[end]), ylims=(zc[1], zc[end]), aspect_ratio=1, c=:turbo)
                png(p1, @sprintf("viz3D_out/%04d.png", iframe += 1))
            end
        end

    return nothing

    end



