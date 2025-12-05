using Literate

src    = joinpath(@__DIR__, "bin_io_script.jl")
md_dir = joinpath(@__DIR__, "md")

Literate.markdown(src, md_dir;
    name       = "bin_io_script",
    execute    = true,
    documenter = false,
    credit     = false,
)