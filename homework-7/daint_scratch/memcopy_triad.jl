function memcopy_triad_KP!(A, B, C, s)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    @inbounds A[ix,iy] = B[ix,iy] + s*C[ix,iy]
    return nothing
end

function memcopy_triad_gpu(nx, ny)
    # Allocate arrays on GPU
    A = CUDA.zeros(Float64, nx, ny)
    B = CUDA.rand(Float64, nx, ny)
    C = CUDA.rand(Float64, nx, ny)


    s = rand()
    threads = (32, 4)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    t_it = @belapsed begin @cuda blocks=$blocks threads=$threads memcopy_triad_KP!($A, $B, $C, $s); synchronize() end
    T_tot = 3*1/1e9*nx*ny*sizeof(Float64)/t_it

    return T_tot
end
