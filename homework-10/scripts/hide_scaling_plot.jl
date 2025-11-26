using Plots, Plots.Measures, Printf
using LinearAlgebra
using Statistics
using MAT
using Printf
using LaTeXStrings



default(size=(1200, 800), framestyle=:box, label=false, grid=true, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24,  legendfontsize=16)


function T_eff_benchmark()

    ## x data
    hide_pairs = [(0,0), (2,2), (8,2), (16,4), (16,16)]  # number of processes used

    ## loading times from .mat file where performance results are stored times_hide
    Times = zeros(5)

    ref_time = 1.958  # time for 1 process

    # hide pairs and corresponding times from output files of the benchmark runs

    Times[1] = 100.0 * ref_time / 2.206 
    Times[2] = 100.0 * ref_time / 2.066
    Times[3] = 100.0 * ref_time / 2.068
    Times[4] = 100.0 * ref_time / 2.046
    Times[5] = 100.0 * ref_time / 2.063


    
    @printf("Loaded benchmark times \n")

    x = [string("(",hide_pairs[i][1],",",hide_pairs[i][2],")") for i in 1:5]
    x[1] = "no com. hide"

    # Plotting for each hide pair the relative time as bar plot
    p = plot(x, Times; 
            xlabel=L"\textbf{Hide\ Pairs}\ (P_x,P_y)", ylabel="Rel. reziproc Time (%) to single GPU",
            title="hide communication benchmark on 64 GPUs",
            legend=false, lw=2, color=:black)

    display(p)
    savefig(p, "../docs/Times_hide_scale.png")
    @printf("Saved benchmark plot to docs/Times_hide_scale.png\n")
    return nothing
end

T_eff_benchmark()