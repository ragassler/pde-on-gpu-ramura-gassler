using Printf
using FFMPEG

srcdir = joinpath(@__DIR__, "frames")
dstdir = joinpath(@__DIR__, "frames_seq")
mkpath(dstdir)

steps = vcat([1], collect(50:50:4000))
for (k, it) in enumerate(steps)
    src = joinpath(srcdir, "heatmap_arrows_implicit_$(it).png")
    dst = joinpath(dstdir, @sprintf("frame_%05d.png", k))
    cp(src, dst; force=true)
end

# make the mp4
run(`ffmpeg -y -framerate 20 -i $(joinpath(dstdir, "frame_%05d.png")) -c:v libx264 -pix_fmt yuv420p $(joinpath(@__DIR__, "porous_convection.mp4"))`)
