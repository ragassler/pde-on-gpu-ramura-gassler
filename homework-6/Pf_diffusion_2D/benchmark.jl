# benchmark.jl — minimal runner (expects each file to define f(nx, ny) -> Number)

using Plots, Plots.Measures, Printf, BenchmarkTools
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=1, labelfontsize=11, tickfontsize=11, titlefontsize=11)


# === EDIT: list your files here ===
FILES = [
    "Pf_diffusion_2D_perf.jl",
    "Pf_diffusion_2D_Teff.jl",
    "Pf_diffusion_2D_loop_fun.jl"
]

# size sweep (nx = ny)
sizes = 16 .* 2 .^(1:8)



function run_one(file::AbstractString, sizes)
    include(file)
    fname = splitext(basename(file))[1]
    fsym  = Symbol(fname)
    @assert isdefined(Main, fsym) "Expected function $(fname)(nx,ny) (or with bench=:btool) in $file"
    f = getfield(Main, fsym)



    T = Float64[]
    for n in sizes
        val = Base.invokelatest(f, n, n)
        @assert val isa Number "Function $fname must return a Number (T_eff_kp), got $(typeof(val))"
        push!(T, Float64(val))
    end

    plt = plot(sizes, T; xscale=:log2, xlabel="nx = ny", ylabel="T_eff [GB/s]",
               label="T_eff_kp", markershape=:diamond, markersize=5,
               linesize=0.5, title=fname)
    savefig(plt, "$(fname).png")
    @info "Saved $(fname).png"

    return fname, T
end

# run all and a combined comparison plot
combined = plot(xlabel="nx = ny", ylabel="T_eff [GB/s]", xscale=:log2,
                title="Pf_diffusion_2D kernels (btool)")

for file in FILES
    fname, T = run_one(file, sizes)
    plot!(combined, sizes, T; label=fname, markershape=:diamond, markersize=5, linesize=0.5)
end

savefig(combined, "Pf_diffusion_2D_all.png")
@info "Saved Pf_diffusion_2D_all.png"