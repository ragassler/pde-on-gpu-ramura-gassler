using Plots, Plots.Measures, Printf
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

function Pf_diffusion_2D_Teff(nx, ny)



    # physics
    lx, ly = 20.0, 20.0
    k_ηf   = 1.0
    # numerics

    cfl     = 1.0 / sqrt(2.1)
    re      = 2π
    # derived numerics
    dx, dy  = lx / nx, ly / ny
    xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
    θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    # array initialisation
    Pf      = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
    qDx     = zeros(Float64, nx + 1, ny)
    qDy     = zeros(Float64, nx, ny + 1)


    # benchmark Btool

    t_it = @belapsed begin

        $qDx[2:end-1, :] .-= ($qDx[2:end-1, :] .+ $k_ηf .* (diff($Pf, dims=1) ./ $dx)) ./ (1.0 + $θ_dτ)
        $qDy[:, 2:end-1] .-= ($qDy[:, 2:end-1] .+ $k_ηf .* (diff($Pf, dims=2) ./ $dy)) ./ (1.0 + $θ_dτ)
        $Pf              .-= (diff($qDx, dims=1) ./ $dx .+ diff($qDy, dims=2) ./ $dy) ./ $β_dτ

    end

    A_eff = 6*nx*ny*8 / 1e9
    T_eff = A_eff / t_it
    @printf("Time = %.3e\n", t_it)
    @printf("A_eff=%1.3f GB, T_eff=%1.3f GB/s\n", A_eff, T_eff)

    return T_eff
end
