
using CairoMakie
@views function implicit_diffusion_advection_2D()
    # physics
    lx, ly  = 10.0, 10.0
    dc      = 1.0
    vx      = 10.0
    vy      = -10.0

    # numerics
    nt      = 100 
    nx      = 200
    ny      = 201
    dx      = lx / nx
    dy      = ly / ny

    # derived physics
    dt      = min(dx / abs(vx), dy / abs(vy)) / 2
    da      = lx^2 / dc / dt
    re      = π + sqrt(π^2 + da)
    ρ       = (lx / (dc * re))^2

    # convergence numerics
    ϵtol    = 1e-8
    maxiter = 10nx
    ncheck  = ceil(Int, 0.02nx)

    # derived numerics
    xc      = LinRange(dx / 2, lx - dx / 2, nx)
    yc      = LinRange(dy / 2, ly - dy / 2, ny)
    dτ      = min(dx, dy) / sqrt(1 / ρ) / sqrt(2)
    
    # array initialisation
    C       = @. exp(-(xc - lx / 4)^2 - (yc' - 3ly / 4)^2)
    C_old   = copy(C)
    qx      = zeros(Float64, nx - 1, ny)
    qy      = zeros(Float64, nx, ny - 1)
    

    # time loop
        # plots setup

    fig = Figure(size=(400, 650))
    ax1 = Axis(fig[1, 1]; 
                xlabel="lx", 
                ylabel="ly", 
                aspect=DataAspect(), 
                title="iter/nx=$(round(0.0,sigdigits=3))")

    ax2 = Axis(fig[2, 1]; 
                yscale=log10, 
                xlabel="iter/nx", 
                ylabel="err",
                limits = ((0.0, 0.2), (1e-10, 1e0)))
    
    hm  = heatmap!(ax1, xc, yc, C;
                colormap=:roma, 
                colorrange=(0,1))

    cb  = Colorbar(fig[1, 2], hm;
                label="C",
                width=15)

    # arrows plot
    # plot every 10th center (skip borders because we need left/right & bottom/top faces)
    xcar = xc[1:10:end-1]
    ycar = yc[1:10:end-1]

 


    ar  = arrows2d!(ax1, xcar, ycar, qx[1:10:end-1, 1:10:end-1], qy[1:10:end-1, 1:10:end-1]; color=:black)
    plt = scatterlines!(ax2, Float64[], Float64[])
    record(fig, "heatmap_arrows.mp4"; fps=20) do io
        for it = 1:nt
            C_old .= C
            iter = 1; err = 2ϵtol; iter_evo = Float64[]; err_evo = Float64[]

            # pseudo transient diffusion implicit loop    
            while err >= ϵtol && iter <= maxiter
                qx         .-= dτ ./ (ρ .+ dτ / dc) .* (qx ./ dc .+ diff(C, dims=1) ./ dx)
                qy         .-= dτ ./ (ρ .+ dτ / dc) .* (qy ./ dc .+ diff(C, dims=2) ./ dy)
                C[2:end-1, 2:end-1] .-= dτ ./ (1.0 .+ dτ / dt) .* ((C[2:end-1, 2:end-1] .- C_old[2:end-1, 2:end-1]) ./ dt .+ diff(qx[:,2:end-1], dims=1) ./ dx + diff(qy[2:end-1,:], dims=2) ./ dy)
                if iter % ncheck == 0
                    
                    # compute residual and error since in 2D we have divergence
                    divq = diff(qx[:, 2:end-1], dims=1)/dx .+ diff(qy[2:end-1, :], dims=2)/dy
                    res  = (C[2:end-1, 2:end-1] .- C_old[2:end-1, 2:end-1]) / dt .+ divq
                    err  = maximum(abs.(res))
                    push!(iter_evo, iter / nx); push!(err_evo, err)
                end
                iter += 1
            end

            # advection explicit upwind

            C[2:end-1, :] .-=   (max(vx,0)*dt).*diff(C[1:end-1, :], dims=1)./dx
            C[2:end-1, :] .-=   (min(vx,0)*dt).*diff(C[2:end, :], dims=1)./dx

            C[:, 2:end-1] .-=   (max(vy,0)*dt).*diff(C[:, 1:end-1], dims=2)./dy
            C[:, 2:end-1] .-=   (min(vy,0)*dt).*diff(C[:, 2:end], dims=2)./dy

            # update heatmap
            hm[3] = C
            ax1.title = "iter/nx=$(round(iter/nx,sigdigits=3))"

            # update arrows
            ar[3] = qx[1:10:end-1, 1:10:end-1]
            ar[4] = qy[1:10:end-1, 1:10:end-1]
            # update error plot
            plt[1] = (iter_evo)
            plt[2] = (err_evo)


            recordframe!(io)


        end
    end
end

implicit_diffusion_advection_2D()
