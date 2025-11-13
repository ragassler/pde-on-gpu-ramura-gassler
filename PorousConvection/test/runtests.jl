# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package ParallelStencil.jl.
# push!(LOAD_PATH, "../src")
using PorousConvection

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    println("Using Julia executable: $exename")
    testdir = pwd()
    println("Test directory: $testdir")

    printstyled("Testing PorousConvection.jl\n"; bold=true, color=:white)

    run(`$exename --project -O3 --startup-file=no -t 8 $(joinpath(testdir, "test2D.jl"))`)
    run(`$exename --project -O3 --startup-file=no -t 8 $(joinpath(testdir, "test3D.jl"))`)

    return
end

runtests()