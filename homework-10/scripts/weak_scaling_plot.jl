using Plots, Plots.Measures, Printf
using LinearAlgebra
using Statistics
using MAT
using Printf
using LaTeXStrings



default(size=(1200, 800), framestyle=:box, label=false, grid=true, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24,  legendfontsize=16)


function T_eff_benchmark()

    ##
    nprocs = [1, 4, 16, 25, 64]  # number of processes used
    nx = [16384, 32766, 65530, 81912, 131058]  # corresponding grid sizes

    #time effective data : from output files of the benchmark runs 
    # nprocs = 1
    Time_n1 = 1.958

    Times_n4 = zeros(4)
    # nprocs = 4
    Times_n4[1] = 2.067  
    Times_n4[2] = 2.067 
    Times_n4[3] = 2.069 
    Times_n4[4] = 2.070
    # average time for nprocs = 4 


    Times_n16 = zeros(16)
    # nprocs = 16
    Times_n16[1] = 2.062 
    Times_n16[2] = 2.060 
    Times_n16[3] = 2.063 
    Times_n16[4] = 2.063 
    Times_n16[5] = 2.065 
    Times_n16[6] = 2.062 
    Times_n16[7] = 2.062 
    Times_n16[8] = 2.065 
    Times_n16[9] = 2.065 
    Times_n16[10] = 2.061 
    Times_n16[11] = 2.062 
    Times_n16[12] = 2.063 
    Times_n16[13] = 2.062 
    Times_n16[14] = 2.063 
    Times_n16[15] = 2.065 
    Times_n16[16] = 2.062 

    # nprocs = 25
    Times_n25 = zeros(25)
    Times_n25[1] = 2.026 
    Times_n25[2] = 2.025 
    Times_n25[3] = 2.028 
    Times_n25[4] = 2.028 
    Times_n25[5] = 2.030 
    Times_n25[6] = 2.029 
    Times_n25[7] = 2.030 
    Times_n25[8] = 2.027 
    Times_n25[9] = 2.031 
    Times_n25[10] = 2.030 
    Times_n25[11] = 2.033 
    Times_n25[12] = 2.031 
    Times_n25[13] = 2.030 
    Times_n25[14] = 2.026 
    Times_n25[15] = 2.028 
    Times_n25[16] = 2.027 
    Times_n25[17] = 2.033 
    Times_n25[18] = 2.030 
    Times_n25[19] = 2.031 
    Times_n25[20] = 2.031 
    Times_n25[21] = 2.031 
    Times_n25[22] = 2.028 
    Times_n25[23] = 2.034 
    Times_n25[24] = 2.026 
    Times_n25[25] = 2.024 

    
    # nprocs = 64
    Times_n64 = zeros(64)
    Times_n64[1] = 2.067 
    Times_n64[2] = 2.069 
    Times_n64[3] = 2.073 
    Times_n64[4] = 2.072 
    Times_n64[5] = 2.070 
    Times_n64[6] = 2.071 
    Times_n64[7] = 2.073 
    Times_n64[8] = 2.068 
    Times_n64[9] = 2.070 
    Times_n64[10] = 2.069 
    Times_n64[11] = 2.069 
    Times_n64[12] = 2.071 
    Times_n64[13] = 2.072 
    Times_n64[14] = 2.070 
    Times_n64[15] = 2.070 
    Times_n64[16] = 2.068 
    Times_n64[17] = 2.070 
    Times_n64[18] = 2.066 
    Times_n64[19] = 2.070 
    Times_n64[20] = 2.074 
    Times_n64[21] = 2.069 
    Times_n64[22] = 2.072 
    Times_n64[23] = 2.073 
    Times_n64[24] = 2.070 
    Times_n64[25] = 2.071 
    Times_n64[26] = 2.068 
    Times_n64[27] = 2.069 
    Times_n64[28] = 2.071 
    Times_n64[29] = 2.071 
    Times_n64[30] = 2.069 
    Times_n64[31] = 2.070 
    Times_n64[32] = 2.066 
    Times_n64[33] = 2.071 
    Times_n64[34] = 2.068 
    Times_n64[35] = 2.071 
    Times_n64[36] = 2.075
    Times_n64[37] = 2.070 
    Times_n64[38] = 2.070 
    Times_n64[39] = 2.072 
    Times_n64[40] = 2.069 
    Times_n64[41] = 2.071 
    Times_n64[42] = 2.069 
    Times_n64[43] = 2.072 
    Times_n64[44] = 2.071 
    Times_n64[45] = 2.069 
    Times_n64[46] = 2.070 
    Times_n64[47] = 2.067 
    Times_n64[48] = 2.071 
    Times_n64[49] = 2.068 
    Times_n64[50] = 2.070 
    Times_n64[51] = 2.072 
    Times_n64[52] = 2.070 
    Times_n64[53] = 2.070 
    Times_n64[54] = 2.072 
    Times_n64[55] = 2.069 
    Times_n64[56] = 2.070 
    Times_n64[57] = 2.068 
    Times_n64[58] = 2.070 
    Times_n64[59] = 2.071 
    Times_n64[60] = 2.072 
    Times_n64[61] = 2.070 
    Times_n64[62] = 2.070 
    Times_n64[63] = 2.071 
    Times_n64[64] = 2.071 

    # compute average times for each nprocs
    Times = [Time_n1; mean(Times_n4); mean(Times_n16); mean(Times_n25); mean(Times_n64)]

    # compute relative times to Time_n1
    for i in 1:length(Times)
        Times[i] = (Time_n1/Times[i])*100
    end

    # compute standerd deviations for error bars
    std_devs = [0; std(Times_n4)*100; std(Times_n16)*100; std(Times_n25)*100; std(Times_n64)*100]

    @printf("Loaded benchmark times \n")

    # Plotting

    #plot Times vs nprocs with error area
    pp= plot(nprocs, Times; ribbon=std_devs, label= "multi GPU (Gridsize: nx = 16384*nprocs)", xscale=:log2,
             xlabel="Number of Processes", ylabel="Rel. reziproc. Time in (%) to 1 Process",
             title="Weak scaling Relative Time vs Nprocs Diff_2D",
             legend=:bottomright, lw=2, ylims=(90, 101), color=:black, fillalpha=0.3)
        scatter!(nprocs, Times; yerror=std_devs, label="", color=:black)

    display(pp)
    savefig(pp, "../docs/T_eff_weak_scale.png")
    @printf("Saved benchmark plot to docs/T_eff_weak_scale.png\n")
    return nothing
end

T_eff_benchmark()