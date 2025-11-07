# 1D Diffusion equation solver for testing purposes

@views Diff(A) = A[2:end].-A[1:end-1]



function diffusion_1D()
    # physics
    lx   = 20.0
    dc   = 1.0

    # numerics
    nx   = 200

    # derived numerics
    dx   = lx / nx
    dt   = dx^2 / dc / 2
    nt   = nx^2 ÷ 100
    xc   = LinRange(dx / 2, lx - dx / 2, nx)

    # array initialisation
    C    = @. 0.5cos(9π * xc / lx) + 0.5
    qx   = zeros(Float64, nx - 1)

    # time loop
    for it = 1:nt
        qx          .= .-dc .* Diff(C) ./ dx
        C[2:end-1] .-= dt .* Diff(qx) ./ dx

    end
    return C, qx
end




