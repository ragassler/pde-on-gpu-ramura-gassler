using GLMakie
using Printf

function load_array!(Aname::AbstractString, A)
    fname = string(Aname, ".bin")  # e.g. "viz3Dmpi_out_large/out_T_0002" -> "…0002.bin"
    open(fname, "r") do fid
        read!(fid, A)
    end
end

function animate_orbit()
    lx, ly, lz = 40.0, 20.0, 20.0
    nz          = 250
    ny          = nz
    nx          = 506

    # base names WITHOUT .bin, to match load_array!
    files = [@sprintf("viz3Dmpi_out_large/out_T_%04d", i) for i in 2:60]

    println("Found $(length(files)) timesteps")
    println("First: ", files[1], ".bin")
    println("Last : ", files[end], ".bin")

    # single 3D array reused for all frames
    T = zeros(Float32, nx, ny, nz)
    println("Loading initial field...")
    load_array!(files[1], T)

    fig = Figure(size = (800, 500))  # use `size` not `resolution`
    ax  = Axis3(fig[1, 1];
        aspect = (1, 1, 0.5),
        title  = "Temperature",
        xlabel = "lx", ylabel = "ly", zlabel = "lz"
    )

    # fix camera view (no rotation in the loop)
    ax.azimuth[]   = - π/4
    ax.elevation[] = - π/6

    # VolumeLike sides MUST be endpoints, not ranges:
    xside = 0.0 .. lx
    yside = 0.0 .. ly
    zside = 0.0 .. lz

    # observable for streaming data
    T_obs = Observable(T)

    # --- choose your 3D visualization here ---

    # 1) Default: contour plot (isosurfaces via marching cubes)
    # contour!(ax, xside, yside, zside, T_obs; alpha = 0.05, colormap = :turbo)


    # 2) Alternative: volume rendering (uncomment to use)
    # volume!(ax, xside, yside, zside, T_obs;
    #     colormap     = :turbo,
    #     transparency = true,
    #     algorithm    = :absorption,  # or :mip, :iso, ...
    # )

    # 3) Alternative: thresholded (masked) volume (only T > Tcrit visible)
    #
    # Uncomment this whole block AND comment out A/B above if you want
    # only hot regions to show up as “clouds”.
    #
    Tmin, Tmax = extrema(T)
    # choose a threshold; e.g. top 30% of the temperature range:
    Tcrit = Tmax - 0.75f0 * (Tmax - Tmin)
    
    Tmask = similar(T)                   # same size/type as T
    Tmask_obs = Observable(Tmask)
    
    # initialize mask from first T
    @inbounds @simd for i in eachindex(T)
        v = T[i]
        Tmask[i] = v < Tcrit ? v : NaN32   # NaN32 is Float32-NaN
    end
    Tmask_obs[] = Tmask

    # volume!(ax, xside, yside, zside, Tmask_obs;
    #     colormap     = :turbo,
    #     transparency = true,
    #     nan_color    = :transparent,   # <- NaNs are invisible
    # )
    # 3 .b 
    # contour!(ax, xside, yside, zside, Tmask_obs; alpha = 0.05, colormap = :turbo)

    contour!(ax, xside, yside, zside, T_obs;
    levels = [Tcrit],         # only this level
    colormap = :cool,   # single red layer
    )
    # ------------------------------------------

    nframes = length(files)
    println("Creating animation...")

    record(fig, "../docs/PorousConvection_under_$(nx+2)x$(ny+2)x$(nz+2).gif",
           1:nframes; framerate = 20) do k
        load_array!(files[k], T)    # re-use same array
        T_obs[] = T                 # trigger redraw

        # If using thresholded volume (block C), also update the mask:
        # @inbounds @simd for i in eachindex(T)
        #     v = T[i]
        #     Tmask[i] = v > Tcrit ? v : NaN32
        # end
        # Tmask_obs[] = Tmask

        if k % 10 == 0 || k == 1 || k == nframes
            println("Frame $k / $nframes")
        end
    end

    return fig
end

animate_orbit()