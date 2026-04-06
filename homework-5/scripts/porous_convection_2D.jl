using CairoMakie # for visualisation
using Printf # for formatted printing

@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])

@views function double_diffusion_2D()
    # physics
    lx      = 40.0
    ly      = 20.0
    #λ       = 0.001
    k_ηf    = 1.0
    α       = 1.0
    αρgx, αρgy = 0.0, 1.0
    αρg        = sqrt(αρgx^2 + αρgy^2)
    ΔT         = 200.0
    ϕ          = 0.1
    Ra         = 1000.0
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    nx      = 127
    ny      = 64
    ϵtol    = 1e-8
    maxiter = 50nx
    ncheck  = ceil(Int, 0.25*max(nx, ny))
    nt      = 500 
    nvis    = 5
    
    # derived numerics
    dx      = lx / nx
    dy      = ly / ny
    xc      = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc      = LinRange(-ly + dy / 2, - dy / 2, ny)
    cfl     = 1.0 / sqrt(2.1)
    dtd     = min(dx, dy)^2 / λ_ρCp / 4.1
    # pressure PT
    re_D    = 2π
    θ_dτ_D  = max(lx,ly) / re_D / (cfl * min(dx, dy))
    β_dτ_D  = k_ηf * re_D / (cfl * min(dx, dy) * max(lx, ly))
    # array initialisation
    # temperature
    T         = @. ΔT * exp(-xc^2 - (yc' + ly / 2)^2)
    T[:, 1] .= ΔT / 2; T[:, end] .= -ΔT / 2
    T_i     = copy(T)
    # pressure
    Pf      = zeros(nx, ny)
    r_Pf    = similar(Pf)
    qDx     = zeros(Float64, nx + 1, ny)
    qDy     = zeros(Float64, nx, ny + 1)
    qDx_c   = zeros(Float64, nx, ny)
    qDy_c   = zeros(Float64, nx, ny)



    # vis
    fig, ax, hm = heatmap(xc, yc, T;
                          figure=(size=(600, 300),),
                          axis=(aspect=DataAspect(), title="Temperature"),
                          colormap=:turbo,
                          colorrange=(-0.25ΔT, 0.25ΔT))

    st          = ceil(Int, nx / 25)
    
    ar          = arrows2d!(ax, xc[1:st:end], yc[1:st:end], qDx_c[1:st:end,1:st:end], qDy_c[1:st:end,1:st:end]; 
                            normalize=true,
                            shaftwidth=0.5,
                            tipwidth=7,
                            tiplength=7)


    Colorbar(fig[:, end+1], hm)
    limits!(ax, -lx / 2, lx / 2, -ly, 0)
    
    record(fig, "heatmap_arrows.mp4"; fps=20) do io
        # time loop
        for it in 1:nt

            # iteration loop
            iter = 1; err_Pf = 2ϵtol
            while err_Pf >= ϵtol && iter <= maxiter
                # pressure
                qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ k_ηf .* (diff(Pf, dims=1) ./ dx .- αρgx .* avx(T))) ./ (θ_dτ_D + 1.0)
                qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ k_ηf .* (diff(Pf, dims=2) ./ dy .- αρgy .* avy(T))) ./ (θ_dτ_D + 1.0)
                Pf           .-= (diff(qDx, dims=1) ./ dx) ./ β_dτ_D .+ (diff(qDy, dims=2) ./ dy) ./ β_dτ_D
                if iter % ncheck == 0
                    r_Pf  .= diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy
                    err_Pf = maximum(abs.(r_Pf))
                    @printf("it = %d, iter/nx=%.1f, err_Pf=%1.3e\n", it, iter / nx, err_Pf)
                end
                iter += 1
            end
            dta = ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1
            dt  = min(dtd, dta)
            # temperature

            T[2:end-1, 2:end-1] .+= dt .* λ_ρCp .* diff(diff(T[:,2:end-1], dims=1) ./ dx, dims=1) ./ dx .+
                                    dt .* λ_ρCp .* diff(diff(T[2:end-1, :], dims=2) ./ dy, dims=2) ./ dy

            T[2:end-1, 2:end-1] .-= dt ./ ϕ .* (max.(qDx[2:end-2, 2:end-1], 0.0) .* diff(T[1:end-1, 2:end-1], dims=1) ./ dx .+
                                        min.(qDx[3:end-1, 2:end-1], 0.0) .* diff(T[2:end, 2:end-1], dims=1) ./ dx .+
                                        max.(qDy[2:end-1, 2:end-2], 0.0) .* diff(T[2:end-1, 1:end-1], dims=2) ./ dy .+
                                        min.(qDy[2:end-1, 3:end-1], 0.0) .* diff(T[2:end-1, 2:end], dims=2) ./ dy)

            # boundary conditions adiabatic
            T[[1, end], :] .= T[[2, end-1], :]
            
            
            # visualisation
            if it % nvis == 0
                qDx_c .= avx(qDx)
                qDy_c .= avy(qDy)
                ar[3] = qDx_c[1:st:end, 1:st:end]
                ar[4] = qDy_c[1:st:end, 1:st:end]
                hm[3] = T
                #display(fig)
                recordframe!(io)
            end
        end
    end
end

double_diffusion_2D()