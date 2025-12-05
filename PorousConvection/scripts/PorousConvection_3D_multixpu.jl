"""
Julia script to solve 3D porous convection using implicit time integration with multi-XPU support (CPU/GPU) via ParallelStencil.jl and ImplicitGlobalGrid.jl.

It can be run on multiple devices (CPUs/GPUs) to distribute the computational load.
Adjust USE_GPU constant to switch between CPU and GPU execution. (For CPU execution ensure low resolution)

On Daint supercomputer use sbatch script provided in the repository to run on multiple GPUs.

"""

##---------------------------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------- GPU/CPU SELECTION ------------------------------------------------------------------------##
const USE_GPU = false

##---------------------------------------------------------------------------------------------------------------------------------------------##


##-------------------------------------------------- IMPORTS AND INITIALIZATIONS --------------------------------------------------------------##

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using ImplicitGlobalGrid
import MPI
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
    CUDA.allowscalar(false)  # catches accidental CPU-style indexing on CuArrays
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
    @info "threads" Threads.nthreads()
end

using Plots, Plots.Measures, Printf

##---------------------------------------------------------------------------------------------------------------------------------------------##

# utility functions

max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end



#----------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------3D kernel functions: for parallelization ------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------------------#


"""
Compute Darcy fluxes `dQx`, `dQy`, `dQz` from pressure `Pf` and temperature `T`
using the Boussinesq approximation on a staggered grid.

Arguments
- dQx, dQy, dQz : Darcy fluxes in x/y/z (updated in place)
- Pf            : pressure field
- k_ηf          : permeability / viscosity
- _1_θ_dτ       : 1 / (1 + θ * dτ)
- g             : gravitational acceleration
- T             : temperature field
- _dx,_dy,_dz   : inverse grid spacings

Notes
- Uses `@parallel_indices` and FiniteDifferences3D operators.
- Handles staggered locations and avoids out-of-bounds access.
- Returns nothing.
"""
@parallel_indices (ix, iy, iz) function compute_flux!(dQx, dQy, dQz, Pf, k_ηf, _1_θ_dτ, g, T, _dx, _dy, _dz)
    nx, ny, nz = size(Pf)
    if (ix <= nx - 1 && iy <= ny && iz <= nz) dQx[ix+1, iy, iz] -= (dQx[ix+1, iy, iz] + k_ηf * (_dx * (Pf[ix+1, iy, iz] - Pf[ix, iy, iz]))) * _1_θ_dτ  end
    if (ix <= nx && iy <= ny - 1 && iz <= nz) dQy[ix, iy+1, iz] -= (dQy[ix, iy+1, iz] + k_ηf * (_dy * (Pf[ix, iy+1, iz] - Pf[ix, iy, iz]))) * _1_θ_dτ  end
    if (ix <= nx && iy <= ny && iz <= nz - 1) dQz[ix, iy, iz+1] -= (dQz[ix, iy, iz+1] + k_ηf * (_dz * (Pf[ix, iy, iz+1] - Pf[ix, iy, iz]) - g * 0.5*(T[ix, iy, iz+1] + T[ix, iy, iz]))) * _1_θ_dτ  end
    return nothing
end


"""
Implicit pressure update for `Pf` using flux divergence.

Arguments
- Pf            : pressure field (updated in place)
- dQx,dQy,dQz   : Darcy fluxes in x/y/z
- _dx,_dy,_dz   : inverse grid spacings
- _β_dτ         : implicit time-integration factor

Returns
- nothing
"""
@parallel function update_Pf!(Pf, dQx, dQy, dQz, _dx, _dy, _dz, _β_dτ)

    @all(Pf) = @all(Pf) - _β_dτ * ( @d_xa(dQx) * _dx + @d_ya(dQy) * _dy + @d_za(dQz) * _dz )
    return nothing
end



"""
Compute residual of the pressure conservation equation.

Arguments
- r_Pf         : residual field (updated in place)
- dQx,dQy,dQz  : Darcy fluxes in x/y/z
- _dx,_dy,_dz  : inverse grid spacings

Typical usage
- After call, use `max_g(abs.(r_Pf))` as convergence measure.

Returns
- nothing
"""
@parallel function compute_residual!(r_Pf, dQx, dQy, dQz, _dx, _dy, _dz)
    @all(r_Pf) = @all(r_Pf) + ( @d_xa(dQx) * _dx + @d_ya(dQy) * _dy + @d_za(dQz) * _dz )
    return nothing
end


"""
Compute conductive temperature fluxes on staggered grids.

Arguments
- dQx,dQy,dQz  : temperature fluxes in x/y/z (updated in place)
                 size(dQx) = (nx-2, ny-1, nz-1)
                 size(dQy) = (nx-1, ny-2, nz-1)
                 size(dQz) = (nx-1, ny-1, nz-2)
- T            : temperature field
- λ_ρCp        : thermal diffusivity
- _1_θ_dτ      : 1 / (1 + θ * dτ)
- _dx,_dy,_dz  : inverse grid spacings

Returns
- nothing
"""
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


"""
Compute temperature time derivative `dTdt` including advection.

Arguments
- dTdt         : temperature time derivative (updated in place, interior cells)
- T            : temperature field at current step
- T_old        : temperature field at previous step
- _dt          : inverse time step (1/Δt)
- qDx,qDy,qDz  : Darcy fluxes in x/y/z
- _ϕ           : inverse porosity (1/φ)
- _dx,_dy,_dz  : inverse grid spacings

Notes
- Uses upwind advection based on the sign of the fluxes.
- Indices refer to interior cells of `T`.

Returns
- nothing
"""
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


"""
Final implicit update of temperature field `T`.

Arguments
- T            : temperature field (updated in place)
- dTdt         : temperature time derivative
- dQx,dQy,dQz  : temperature fluxes in x/y/z
- _dx,_dy,_dz  : inverse grid spacings
- _temp        : scalar factor from time integration

Notes
- Uses divergence of temperature fluxes plus `dTdt`.
- Operates on interior cells (shifted indices).

Returns
- nothing
"""
@parallel_indices (ix, iy, iz) function update_T!(T, dTdt, dQx, dQy, dQz, _dx, _dy, _dz, _temp)
    nx, ny, nz = size(dTdt)
    if (ix <= nx && iy <= ny && iz <= nz)
        T[ix+1, iy+1, iz+1] -= (dTdt[ix, iy, iz] + ( @d_xa(dQx) * _dx ) + ( @d_ya(dQy) * _dy ) + ( @d_za(dQz) * _dz )) * _temp
    end
    return nothing
end

"""
Compute residual of the temperature equation for convergence checks.

Arguments
- r_T          : residual field (updated in place)
- dTdt         : temperature time derivative
- dQx,dQy,dQz  : temperature fluxes in x/y/z
- _dx,_dy,_dz  : inverse grid spacings

Typical usage
- After call, use `max_g(abs.(r_T))` as convergence measure.

Returns
- nothing
"""
@parallel_indices (ix, iy, iz) function compute_residual_T!(r_T, dTdt, dQx, dQy, dQz, _dx, _dy, _dz)
    nx, ny, nz = size(r_T)
    if (ix <= nx && iy <= ny && iz <= nz)
        r_T[ix, iy, iz] += dTdt[ix, iy, iz] + ( @d_xa(dQx) * _dx ) + ( @d_ya(dQy) * _dy ) + ( @d_za(dQz) * _dz )
    end
    return nothing
end

"""
Boundary conditions for adiabatic side walls and Dirichlet top/bottom for T.

- `bc_x!`: Adiabatic in x-direction (Neumann: ∂A/∂x = 0).
- `bc_y!`: Adiabatic in y-direction (Neumann: ∂A/∂y = 0).
- `bc_z_T_dirichlet!`: Dirichlet in z-direction for temperature.

Arguments
- A            : generic 3D array (for adiabatic BCs)
- T            : temperature field (for Dirichlet BCs)
- Thot, Tcold  : imposed temperatures at bottom/top

All functions update the arrays in place and return nothing.
"""
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

"""
Main function to run the 3D porous convection simulation solving thermal porous convection using the pseudo-transient method and multi-XPU support.
do_viz   : boolean flag to enable/disable visualization and data saving disable for performance measurements
do_check : boolean flag to enable/disable convergence checking

It initalizes the physical and numerical parameters, sets up the computational grid, and runs the time-stepping loop.
Modify the simulation parameters (nx, ny, nz, nt, etc.) in the function as needed.

The computational grid is distributed across multiple devices (CPUs/GPUs) using ParallelStencil.jl and ImplicitGlobalGrid.jl.
With the kernel functions defined above, it computes the Darcy fluxes, updates the pressure and temperature fields, and optionally performs convergence checks.

If visualization is enabled, it saves temperature field snapshots at specified intervals for later visualization.

Example:
    porous_convection_implicit_3D_multixpu(do_viz=true, do_check=true)

Output:
    progress printed to console
    viz3Dmpi_out/ : directory containing saved temperature field data for visualization


returns nothing
"""

@views function porous_convection_implicit_3D_multixpu(;do_viz=false, do_check=false)
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

   
    nz          = 63
    nx,ny       = 2 * (nz + 1) - 1, nz
    me, dims    = init_global_grid(nx, ny, nz, select_device=false)  # init global grid and more
    b_width     = (8, 8, 4)      
    
    nt         = 500
    re_D       = 4π
    cfl        = 1.0 / sqrt(3.1)
    maxiter    = 10max(nx_g(), ny_g())
    ϵtol       = 1e-6
    nvis       = 50
    ncheck     = ceil(2max(nx_g(), ny_g(), nz_g()))
    ## -------------------------------------------------------------------------------------------##

        
    # derived numerics
    dx      = lx / nx_g()
    dy      = ly / ny_g()
    dz      = lz / nz_g()
    xc      = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx_g())
    yc      = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny_g())
    zc      = LinRange(-lz + dz / 2, - dz / 2, nz_g())


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
    T  = @zeros(nx, ny, nz)
    T .= Data.Array([ΔT * exp(-(x_g(ix, dx, T) + dx / 2 - lx / 2)^2
                            -(y_g(iy, dy, T) + dy / 2 - ly / 2)^2
                            -(z_g(iz, dz, T) + dz / 2 - lz / 2)^2) for ix = 1:size(T, 1), iy = 1:size(T, 2), iz = 1:size(T, 3)])
    T[:, :, 1  ] .=  ΔT / 2
    T[:, :, end] .= -ΔT / 2
    update_halo!(T)
    T_old = copy(T)
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

    # visualisation setup on rank 0
    if do_viz
        ENV["GKSwstype"]="nul"
        if (me==0) if isdir("viz3Dmpi_out")==false mkdir("viz3Dmpi_out") end; loadpath="viz3Dmpi_out/"; anim=Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2], (nz - 2) * dims[3]
        (nx_v * ny_v * nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory()) && error("Not enough memory for visualization.")
        T_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
        T_inn = zeros(nx - 2, ny - 2, nz - 2) # no halo local array for visu
        xi_g, zi_g = LinRange(-lx / 2 + dx + dx / 2, lx / 2 - dx - dx / 2, nx_v), LinRange(-lz + dz + dz / 2, -dz - dz / 2, nz_v) # inner points only
        iframe = 0
    end

    # time loop
    for it in 1:nt


        # print progress only on rank 0
        if (it % 50 == 0) && (me==0)
            @printf("Time step %d / %d (%.1f%%)\n", it, nt, it / nt * 100)
        end


        T_old .= T

        dt = if it == 1
            0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)
        else
            # use max_g
            min(5.0 * min(dx, dy, dz) / (αρg * ΔT * k_ηf), ϕ * min(dx / max_g(abs.(qDx)), dy / max_g(abs.(qDy)), dz / max_g(abs.(qDz))) / 3.1)
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
            @hide_communication b_width begin
                @parallel compute_flux!(qDx, qDy, qDz, Pf, k_ηf, _1_θ_dτ_D, αρg, T, _dx, _dy, _dz)
                update_halo!(qDx, qDy, qDz)
            end

            @parallel update_Pf!(Pf, qDx, qDy, qDz, _dx, _dy, _dz, _β_dτ_D)

            # temperature

            @parallel compute_temp_flux!(qTx, qTy, qTz, T, λ_ρCp, _1_θ_dτ_T, _dx, _dy, _dz)
            @parallel compute_dTdt!(dTdt, T, T_old, _dt, qDx, qDy, qDz, _ϕ, _dx, _dy, _dz)
            
            @hide_communication b_width begin
                @parallel update_T!(T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz, _tmp)

                # boundary conditions adiabatic
                @parallel (1:size(T, 2), 1:size(T, 3)) bc_x!(T)
                @parallel (1:size(T, 1), 1:size(T, 3)) bc_y!(T)
                update_halo!(T)
            end


            if do_check
                if iter % ncheck == 0
                    fill!(r_T, 0.0)
                    fill!(r_Pf, 0.0)
                    @parallel compute_residual_T!(r_T, dTdt, qTx, qTy, qTz, _dx, _dy, _dz)
                    @parallel compute_residual!(r_Pf, qDx, qDy, qDz, _dx, _dy, _dz)

                    err_T = max_g(abs.(r_T))
                    err_D = max_g(abs.(r_Pf))
                    if do_viz && USE_GPU==false
                        @printf("it = %d,  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", it, iter / nx, err_D, err_T)
                    end

                end
            end

            iter += 1
        end

        
        # visualisation
        if do_viz && (it % nvis == 0)
            T_inn .= Array(T)[2:end-1, 2:end-1, 2:end-1]; gather!(T_inn, T_v)
            if me == 0
                p1 = heatmap(xi_g, zi_g, T_v[:, ceil(Int, ny_g() / 2), :]'; xlims=(xi_g[1], xi_g[end]), ylims=(zi_g[1], zi_g[end]), aspect_ratio=1, c=:turbo)
                # display(p1)
                png(p1, @sprintf("viz3Dmpi_out/%04d.png", iframe += 1))
                save_array(@sprintf("viz3Dmpi_out/out_T_%04d", iframe), convert.(Float32, T_v))
            end
        end
    end

    finalize_global_grid()
    return nothing

end

##---------------------------------------------------------------------------------------------------------------------------------------------##

##-------------------------------------------------- Run Simulation ---------------------------------------------------------------------------##
# porous_convection_implicit_3D_multixpu(do_viz=true, do_check=true)



