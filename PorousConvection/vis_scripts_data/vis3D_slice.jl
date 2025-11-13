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
    load_array("out_Ta", T)

    # Mid-plane index in y (1-based, nearest to ly/2)
    iy = (ny + 1) ÷ 2
    ymid = ly / 2

    # Coordinates for 2D slice (x–z)
    xvec = LinRange(0, lx, nx)
    zvec = LinRange(0, lz, nz)
    Tslice = @view T[:, iy, :]  # size (nx, nz)

    fig = Figure(size=(1100, 500))

    # --- 3D isocontours (VolumeLike => use endpoints) ---
    ax3d = Axis3(fig[1, 1];
        aspect=(1, 1, 0.5),
        title="Temperature (3D contours)",
        xlabel="lx", ylabel="ly", zlabel="lz"
    )
    xside = 0.0 .. lx
    yside = 0.0 .. ly
    zside = 0.0 .. lz
    contour!(ax3d, xside, yside, zside, T; alpha=0.05, colormap=:turbo)

    # --- 2D slice at y = ly/2 ---
    ax2d = Axis(fig[1, 2];
        aspect=DataAspect(),
        title = "Slice at y = $(round(ymid; digits=3))",
        xlabel="x", ylabel="z"
    )
    hm = heatmap!(ax2d, xvec, zvec, Tslice; colormap=:turbo)
    Colorbar(fig[1, 3], hm, label="T")

    save("T_3D_with_slice.png", fig)
    return fig
end

visualise()