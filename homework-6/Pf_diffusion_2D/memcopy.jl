using Plots, Plots.Measures, Printf, BenchmarkTools
using LoopVectorization
using Base.Threads; @info "threads" nthreads()

default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)



# array broadcasting version of memory copy operation
function compute_ap!(C, C2, A)


    C2 .= C .+ A

    return nothing
end

# loop vectorized version of memory copy operation
function compute_kp!(C, C2, A, ny, nx)


    @tturbo for iy=1:ny, ix=1:nx
        @inbounds C2[ix, iy] = C[ix, iy] + A[ix, iy]
    end

    return nothing
end

function memcopy(nx, ny; bench=:btool)

    # iterations
    nt      = 2e2
    # array initialisation
    C       = rand(Float64, nx, ny)
    C2      = copy(C)
    A       = copy(C)

    # A_eff = 2 Du + Dk here all arrays are read and written once Dk= 3*nx*ny*8 / 1e9  # GB and Du = 0
    A_eff = 3*nx*ny*8 / 1e9  # GB

    # switch to monitor performance using BenchmarkTools or iteration loop
    if bench == :loop
        # iteration loop
        t_tic_ap = 0.0
        t_tic_kp = 0.0
        for iter=1:nt

            if iter == 110
                t_tic_ap = time()

            end
            compute_ap!(C, C2, A)
        end

        t_toc_ap = Base.time() - t_tic_ap

        for iter=1:nt

            if iter == 110
                t_tic_kp = time()

            end
            compute_kp!(C, C2, A, ny, nx)
        end

        t_toc_kp = Base.time() - t_tic_kp

        niter = nt -110
        t_it_kp = t_toc_kp/niter
        t_it_ap = t_toc_ap/niter
        T_eff_ap = A_eff / t_it_ap
        T_eff_kp = A_eff / t_it_kp
        @printf("Time_ap = %.3e, niter = %d\n", t_toc_ap, niter)
        @printf("Time_kp = %.3e, niter = %d\n", t_toc_kp, niter)
        @printf("A_eff=%1.3f GB, T_eff_ap=%1.3f GB/s\n", A_eff, T_eff_ap)
        @printf("A_eff=%1.3f GB, T_eff_kp=%1.3f GB/s\n", A_eff, T_eff_kp)
        return T_eff_ap, T_eff_kp


    elseif bench == :btool

        t_kp = @belapsed compute_kp!($C, $C2, $A, $ny, $nx)  # seconds per call (minimum over many trials)
        t_ap = @belapsed compute_ap!($C, $C2, $A)      # seconds per call (minimum over many trials)

        niter = nt
        T_eff_ap = A_eff / t_ap
        T_eff_kp = A_eff / t_kp
        @printf("Time_ap = %.3e, niter = %d\n", t_ap*niter, niter)
        @printf("Time_kp = %.3e, niter = %d\n", t_kp*niter, niter)
        @printf("A_eff=%1.3f GB, T_eff_ap=%1.3f GB/s\n", A_eff, T_eff_ap)
        @printf("A_eff=%1.3f GB, T_eff_kp=%1.3f GB/s\n", A_eff, T_eff_kp)
        return T_eff_ap, T_eff_kp
    end

end


### BENCHMARKING T_eff FOR DIFFERENT NX, NY VALUES ####

nx = ny = 16 * 2 .^ (1:8)

T_eff_ap = zeros(length(nx))
T_eff_kp = zeros(length(nx))

### With BenchmarkTools ###

for i=1:length(nx)
    T_eff_ap[i], T_eff_kp[i] = memcopy(nx[i], ny[i]; bench=:btool)
end

### Plotting ###
plot(nx, T_eff_ap; xscale=:log2, label="Array Broadcasting Btool", xlabel="nx=ny", ylabel="T_eff [GB/s]", legend=:topright, markershape=:circle, markersize=10, linesize=1.5)
plot!(nx, T_eff_kp; label="Loop Vectorization Btool", markershape=:diamond, markersize=10, linesize=1.5)


### Without BenchmarkTools ###
for i=1:length(nx)
    T_eff_ap[i], T_eff_kp[i] = memcopy(nx[i], ny[i]; bench=:loop)
end
### Plotting ###
plot!(nx, T_eff_ap; label="Array Broadcasting Loop", markershape=:circle, markersize=10, linesize=1.5)
plot!(nx, T_eff_kp; label="Loop Vectorization Loop", markershape=:diamond, markersize=10, linesize=1.5)
savefig("memcopy.pdf")