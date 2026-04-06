using GLMakie
using Printf

function load_array!(fname::AbstractString, A)
    open(fname, "r") do fid
        read!(fid, A)
    end
end

function visualise()
    lx, ly, lz = 40.0, 20.0, 20.0
    # large simulation
    nz          = 250
    ny          = nz
    nx          = 506

    # --- construct list of filenames chronologically ---
    files = [@sprintf("viz3Dmpi_out_large/out_T_%04d.bin", i) for i in 2:60]
    println("Found $(length(files)) timesteps")
    println("First file: ", files[1])
    println("Last  file: ", files[end])

    # --- allocate a single 3D array and load the first timestep into it ---
    T = zeros(Float32, nx, ny, nz)
    println("Loading initial field...")
    load_array!(files[1], T)

    # --- Makie setup ---
    fig = Figure(size = (800, 500))  # use `size` instead of deprecated `resolution`
    ax  = Axis3(fig[1, 1];
        aspect = (1, 1, 0.5),
        title  = "Temperature",
        xlabel = "lx", ylabel = "ly", zlabel = "lz"
    )

    xside = 0.0 .. lx
    yside = 0.0 .. ly
    zside = 0.0 .. lz

    # Observable for the current field
    T_obs = Observable(T)

    # One contour! that will be updated via T_obs
    contour!(ax, xside, yside, zside, T_obs; alpha = 0.05, colormap = :turbo)

    println("Creating animation...")

    record(fig, "../docs/PorousConvection_3D_multixpu_$(nx+2)x$(ny+2)x$(nz+2).gif",
           1:length(files); framerate = 20) do k
        fname = files[k]

        # reuse the *same* array T for every frame
        load_array!(fname, T)
        T_obs[] = T  # trigger replot without allocating a new array

        if k % 10 == 0 || k == 1 || k == length(files)
            println("Frame $(k) / $(length(files))  (", fname, ")")
        end
    end

    return fig
end

visualise()