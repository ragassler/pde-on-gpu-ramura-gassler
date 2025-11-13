using GLMakie

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid = open(fname, "r"); read!(fid, A); close(fid)
end

function visualise()
    lx, ly, lz = 40.0, 20.0, 20.0
    nx = 255
    ny = nz = 127

    T = zeros(Float32, nx, ny, nz)
    load_array("out_T", T)

    fig = Figure(size=(800, 500))
    ax  = Axis3(fig[1, 1];
        aspect=(1, 1, 0.5),
        title="Temperature",
        xlabel="lx", ylabel="ly", zlabel="lz"
    )

    # For VolumeLike data (contour!/volume/isosurface), sides must be endpoints:
    xside = 0.0 .. lx
    yside = 0.0 .. ly
    zside = 0.0 .. lz

    contour!(ax, xside, yside, zside, T; alpha=0.05, colormap=:turbo)
    save("T_3D.png", fig)
    return fig
end

visualise()