# # 2D linear diffusion Julia MPI solver in GPU
# run: ~/.julia/bin/mpiexecjl -n 4 julia --project diffusion_2D_mpi.jl
using Plots, Printf, MAT
import MPI
using CUDA


# enable plotting by default
if !@isdefined do_save; do_save = true end

# MPI functions using crytos.jl for GPU arrays
@views function update_halo!(A, neighbors_x, neighbors_y, comm)
    # Send to / receive from neighbor 1 in dimension x ("left neighbor")
    if neighbors_x[1] != MPI.PROC_NULL
        h_send = Vector{T}(undef, size(A[2,:]))
        h_recv = Vector{T}(undef, size(A[1,:]))

        CUDA.crypto!(h_send, @view(A[2,:]))
        MPI.Send(h_send,  neighbors_x[1], 0, comm)
        MPI.Recv!(h_recv, neighbors_x[1], 1, comm)
        
        d_recv = CuArray{T}(undef, ny)
        CUDA.copyto!(d_recv, h_recv)
        A[1, :] = d_recv
    end
    # Send to / receive from neighbor 2 in dimension x ("right neighbor")
    if neighbors_x[2] != MPI.PROC_NULL
        h_send = Vector{T}(undef, size(A[end-1,:]))
        h_recv = Vector{T}(undef, size(A[end,:]))

        CUDA.crypto!(h_send, @view(A[end-1,:]))
        MPI.Send(h_send,  neighbors_x[2], 0, comm)
        MPI.Recv!(h_recv, neighbors_x[2], 1, comm)

        d_recv = CuArray{T}(undef, ny)
        CUDA.copyto!(d_recv, h_recv)
        A[end, :] = d_recv
    end
    # Send to / receive from neighbor 1 in dimension y ("bottom neighbor")
    if neighbors_y[1] != MPI.PROC_NULL
        h_send = Vector{T}(undef, size(A[:,2]))
        h_recv = Vector{T}(undef, size(A[:,1]))
        CUDA.crypto!(h_send, @view(A[:,2]))
        MPI.Send(h_send,  neighbors_y[1], 2, comm)
        MPI.Recv!(h_recv, neighbors_y[1], 3, comm)
        d_recv = CuArray{T}(undef, nx)
        CUDA.copyto!(d_recv, h_recv)
        A[:,1] = d_recv
    end
    # Send to / receive from neighbor 2 in dimension y ("top neighbor")
    if neighbors_y[2] != MPI.PROC_NULL
        h_send = Vector{T}(undef, size(A[:,end-1]))
        h_recv = Vector{T}(undef, size(A[:,end]))
        CUDA.crypto!(h_send, @view(A[:,end-1]))
        MPI.Recv!(h_recv, neighbors_y[2], 2, comm)
        MPI.Send(h_send,  neighbors_y[2], 3, comm)
        d_recv = CuArray{T}(undef, nx)
        CUDA.copyto!(d_recv, h_recv)
        A[:,end] = d_recv
    end
    return
end

@views function diffusion_2D_mpi(; do_save=false)
    # MPI
    MPI.Init()
    dims        = [0,0]
    comm        = MPI.COMM_WORLD
    nprocs      = MPI.Comm_size(comm)
    MPI.Dims_create!(nprocs, dims)
    comm_cart   = MPI.Cart_create(comm, dims, [0,0], 1)
    me          = MPI.Comm_rank(comm_cart)
    coords      = MPI.Cart_coords(comm_cart)
    neighbors_x = MPI.Cart_shift(comm_cart, 0, 1)
    neighbors_y = MPI.Cart_shift(comm_cart, 1, 1)
    if (me==0) println("nprocs=$(nprocs), dims[1]=$(dims[1]), dims[2]=$(dims[2])") end
    # Physics
    lx, ly     = 10.0, 10.0
    D          = 1.0
    nt         = 100
    # Numerics
    nx, ny     = 32, 32                             # local number of grid points
    nx_g, ny_g = dims[1]*(nx-2)+2, dims[2]*(ny-2)+2 # global number of grid points
    # Derived numerics
    dx, dy     = lx/nx_g, ly/ny_g                   # global
    dt         = min(dx,dy)^2/D/4.1
    # Array allocation
    qx         = CUDA.zeros(nx-1,ny-2)
    qy         = CUDA.zeros(nx-2,ny-1)
    # Initial condition
    x0, y0     = coords[1]*(nx-2)*dx, coords[2]*(ny-2)*dy
    xc         = [x0 + ix*dx - dx/2 - 0.5*lx  for ix=1:nx]
    yc         = [y0 + iy*dy - dy/2 - 0.5*ly  for iy=1:ny]

    xc_d       = CuArray(xc)
    yc_d       = CuArray(yc)

    C          = CUDA.zeros(nx, ny)
    C          = exp.(.-xc_d.^2 .-yc_d'.^2)

    t_tic = 0.0
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time() end
        qx  .= .-D*diff(C[:,2:end-1], dims=1)/dx
        qy  .= .-D*diff(C[2:end-1,:], dims=2)/dy
        C[2:end-1,2:end-1] .= C[2:end-1,2:end-1] .- dt*(diff(qx, dims=1)/dx .+ diff(qy, dims=2)/dy)
        update_halo!(C, neighbors_x, neighbors_y, comm_cart)
        if do_save file = matopen("$(@__DIR__)/../vis/mpi2D_out_C_$(me)_$(it).mat", "w"); write(file, "C", Array(C)); close(file) end
    end
    t_toc = (Base.time()-t_tic)
    if (me==0) @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2/1e9*nx*ny*sizeof(lx))/(t_toc/(nt-10)), sigdigits=2)) end
    # Save to visualise
    if do_save file = matopen("$(@__DIR__)/mpi2D_out_C_$(me).mat", "w"); write(file, "C", Array(C)); close(file) end
    MPI.Finalize()
    return
end

diffusion_2D_mpi(; do_save=do_save)