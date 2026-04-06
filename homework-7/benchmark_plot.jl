using Plots, Plots.Measures, Printf
using BenchmarkTools
using JLD2
using LaTeXStrings

default(size=(1200, 800), framestyle=:box, label=false, grid=true, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24,  legendfontsize=16)


function T_eff_benchmark()

    # Load saved results
    @load "test_data/benchmark_results.jld2" nx results_memcopy_triad results_Pf_diffusion_2D_loop_gpu
    @printf("Loaded benchmark times from test_data/benchmark_results.jld2\n")

    # Plotting

    #Plot effective throughput of memcopy and diffusion kps vs nx
    # dashed line for T_peak which is the maximum of memcopy
    # gray dashed line for vendors claimed peak performance
    T_peak = maximum(results_memcopy_triad)
    vendors_peak = 4000.0  # GB/s for GH2100

    p = plot(nx.*nx, results_memcopy_triad; label= L"T_{\textbf{eff}}\ \textbf{ Memcopy}", xscale=:log2,
             xlabel=L"\textbf{Grid}\ \textbf{Size}\ N=N_x^2", ylabel=L"T_{\textbf{eff}}\ \textbf{ (GB/s)}",
             title="Effective Throughput vs Grid Size",
             legend=:bottomright, lw=4)
    plot!(nx.*nx, results_Pf_diffusion_2D_loop_gpu; label=L"T_{\textbf{eff}}\ \textbf{Diffusion}\ \textbf{Kernel}", lw=4)
    hline!([T_peak]; linestyle=:dot, color=:black, label=L"T_{\textbf{peak}}\ \textbf{Memcopy}", lw=1)
    hline!([vendors_peak]; linestyle=:dash, color=:gray, label=L"T_{\textbf{peak}}\ \textbf{Vendor}\ \textbf{Claimed}", lw=1)

    # add to Plot the ratio of T_peak achieved by diffusion kernel at largest nx
    T_eff_largest = results_Pf_diffusion_2D_loop_gpu[end]
    pct_T_peak = (T_eff_largest / T_peak)
    pct_T_peak = round(pct_T_peak, digits=3)
    annotate!(nx[3]^2 * 0.6, T_eff_largest * 1.2,
              text("γ = $(pct_T_peak)", :black, 16))
    display(p)
    savefig(p, "Docs/T_eff_benchmark.png")
    @printf("Saved benchmark plot to Docs/T_eff_benchmark.png\n")
    return nothing
end

T_eff_benchmark()


