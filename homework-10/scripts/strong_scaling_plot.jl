using Plots, Plots.Measures, Printf
using MAT
using Printf
using LaTeXStrings



default(size=(1200, 800), framestyle=:box, label=false, grid=true, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24,  legendfontsize=16)


function T_eff_benchmark()

    ## loading data from .mat file where performance results are stored nx, ny and T_eff
    data = matread("../matfiles/benchmark_strong_single_gpu.mat")
    nx = data["nx"]

    T_eff = data["T_eff"]  # T_eff[i,j] is the effective time
    @printf("Loaded benchmark times \n")

    # Plotting

    #Plot effective throughput of memcopy and diffusion kps vs nx
    # dashed line for T_peak which is the maximum of memcopy
    # gray dashed line for vendors claimed peak performance
    vendors_peak = 4000.0  # GB/s for GH2100

    p = plot(nx.*nx, T_eff; label= L"T_{\textbf{eff}}\ \textbf{ Diffusion}", xscale=:log2,
             xlabel=L"\textbf{Grid}\ \textbf{Size}\ N=N_x^2", ylabel=L"T_{\textbf{eff}}\ \textbf{ (GB/s)}",
             title="Strong Scaling T_eff vs Size on Singel GPU",
             legend=:bottomright, lw=2, color=:black)
    hline!([vendors_peak]; linestyle=:dash, color=:gray, label=L"T_{\textbf{peak}}\ \textbf{Vendor}\ \textbf{Claimed}", lw=2)
    scatter!(nx.*nx, T_eff; label="", color=:black)

    display(p)
    savefig(p, "../docs/T_eff_strong_scale.png")
    @printf("Saved benchmark plot to docs/T_eff_strong_scale.png\n")
    return nothing
end

T_eff_benchmark()