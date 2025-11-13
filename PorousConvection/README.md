# PorousConvection

[![Build Status](https://github.com/ragassler/pde-on-gpu-ramura-gassler/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ragassler/pde-on-gpu-ramura-gassler/actions/workflows/CI.yml?query=branch%3Amain)



## Setup

**Julia version:** ≥ 1.9  
**Packages:** `ParallelStencil`, `Plots`, `GLMakie`, `Printf`, `Test`, `CUDA`, `JLD2`

```bash
julia --project -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'

```

## 📁 Repository Structure
```bash
├─ scripts
│ ├─ Pf_diffusion_2D_perf_xpu.jl                      
│ ├─ Pf_diffusion_2D_xpu.jl                   
│ ├─ PorousConvection_2D_xpu_test.jl  
│ ├─ PorousConvection_2D_xpu.jl       
│ ├─ PorousConvection_3D_xpu_test.jl   
│ ├─ PorousConvection_3D_xpu.jl
│ ├─ PorousConvection_2D.jl
├─ docs
│ ├─ Results in png and mp4
├─ test                    
│ ├─ scripts/                           
│ ├─ test/                              
│ ├─ Manifest.toml   
│ ├─ Project.toml                       # activate . 
├─ vis_scripts_data                           #JLD2 files for daint->remote
├─ benchmark_plot.jl                    # plots the test_data benchmark
├─ nummerical_test.jl                   # reference test of gpu version
├─ lecture7_ex1_sub.ipynb               # solution Ex 1.
└─ README.md
```

**scripts:** contain the actual source code for this project.

```bash
├─ scripts
│ ├─ Pf_diffusion_2D_perf_xpu.jl   # task 1.2 parallel indices                    
│ ├─ Pf_diffusion_2D_xpu.jl        # task 1.1 parallel using parallelstencil and macros            
│ ├─ PorousConvection_2D_xpu_test.jl  # same as the next but lower resolution and additional unit testing function 
                                      # it has the same nummerical simulation which can be run on daint.
                                      # but for local testing and nummerical verification
│ ├─ PorousConvection_2D_xpu.jl      # the Porous Convection in 2D for daint run 
│ ├─ PorousConvection_3D_xpu_test.jl # "" 3D "" 
│ ├─ PorousConvection_3D_xpu.jl      # "" 3D ""
│ ├─ PorousConvection_2D.jl         # verified 2D nummerical code produced in submission 5 for reference testing
```

## 2D Porous Convection (XPU)

Parallel CPU/GPU implementation of 2D porous convection with an implicit pseudo–time-stepping scheme.  
Source: `scripts/PorousConvection_2D_xpu.jl`

### Run a simulation

- Ensure visualization is enabled in the script (Makie/Plots).
- **Local CPU (recommended for quick tests):**
  - Set `"USE_GPU"=false` (or configure inside the script).
  - Adjust resolution and number of time steps for your machine.
- **GPU on Daint:**
  - Set `"USE_GPU"=true`.
  - Reserve sufficient wall time; a production run may take ~1 hour depending on resolution.
- The simulation writes frame images (PNG) to disk.

![porous_conv2d](docs/porous_convection.gif)

### Make Plots

1. Download the generated frames/ directory.

2. Move it into vis_scripts_data/.

3. Run:
```bash
julia --project vis.jl
```


## 3D Porous Convection (XPU)

Parallel CPU/GPU implementation of 3D porous convection with an implicit pseudo–time-stepping scheme.  
Source: `scripts/PorousConvection_3D_xpu.jl`

### Run a simulation

- Ensure visualization output is enabled if desired.
- **Local CPU:** set `ENV["USE_GPU"]=false` and reduce resolution/time steps.
- **GPU on Daint:** set `ENV["USE_GPU"]=true` and request adequate wall time (production runs can take ~1 hour depending on resolution).
- The simulation writes the final temperature field to a `.bin` file.


![porous_3d_xpu](docs/T_3D_with_slice.png)

### Visualize 3D results

1. Download the produced .bin file.

2. Move it into vis_scripts_data/.

3. Run either:

```bash
julia --project vis_3D.jl
# or
julia --project vis_3D_slice.jl
```



## Unit and Reference Testing

Note that the _test.jl source codes are nummerical identical to the source code only difference are some parameters like grid size, number of timesteps.
and addtional unit_tesing function for unit testing.

For Nummerical Correctness we have for each 2D and 3D a unit and a reference test
The unit tests are the same where we check one kernel_function for its correctness 

For reference in 2D we can verify nummerical corectness by reference testing the whole simulation against the already verified 2D simulation we have from Homework 5.

For 3D the reference testing is tricky. At the moment we run simulation with Ra=10. In Theory with such low Ra we get only diffusion no convection.
Therefore the reference test checks if that holds for the nummerical simulation.

To run the test use Pkg mode test suite.
