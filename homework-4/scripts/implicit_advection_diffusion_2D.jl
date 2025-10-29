using Plots, Plots.Measures, Printf
default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function implicit_diffusion_advection_2D()
    # physics
    lx, ly  = 10.0, 10.0
    dc      = 1.0
    vx      = 10.0
    vy      = -10.0

    # numerics
    nt      = 50 #for only diffusion use nt=100
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
    # iteration loop
    anim = @animate for it = 1:nt
        C_old .= C
        iter = 1; err = 2ϵtol; iter_evo = Float64[]; err_evo = Float64[]
        # pseudo transient diffusion implicit    
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

        #-------------------TASK 2: ADVECTION------------------- (comment out to see only diffusion)
        # advection explicit upwind (comment out to see only diffusion)

        C[2:end-1, :] .-=   (max(vx,0)*dt).*diff(C[1:end-1, :], dims=1)./dx
        C[2:end-1, :] .-=   (min(vx,0)*dt).*diff(C[2:end, :], dims=1)./dx

        C[:, 2:end-1] .-=   (max(vy,0)*dt).*diff(C[:, 1:end-1], dims=2)./dy
        C[:, 2:end-1] .-=   (min(vy,0)*dt).*diff(C[:, 2:end], dims=2)./dy

        #-------------------END ADVECTION-------------------

        # visualisation
        p1 = Plots.heatmap(xc, yc, C'; xlims=(0, lx), ylims=(0, ly), clims=(0, 1), aspect_ratio=1,
                    xlabel="lx", ylabel="ly", title="iter/nx=$(round(iter/nx,sigdigits=3))")
        p2 = Plots.plot(iter_evo, err_evo; xlabel="iter/nx", ylabel="err", yscale=:log10, grid=true, markershape=:circle, markersize=10)
        Plots.plot(p1, p2; layout=(2, 1))

    end
    gif(anim, "implicit_diffusion_advection_2D.gif", fps=10)
end

implicit_diffusion_advection_2D()
