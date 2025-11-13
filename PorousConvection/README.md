# PorousConvection

[![Build Status](https://github.com/ragassler/pde-on-gpu-ramura-gassler/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ragassler/pde-on-gpu-ramura-gassler/actions/workflows/CI.yml?query=branch%3Amain)


A collection of 2D/3D porous convection solvers written in Julia, with CPU/GPU backends via [ParallelStencil.jl]. Includes scripts for running simulations, visualization helpers, and a test suite with unit and reference tests.

---

## Setup

**Julia:** ≥ 1.9  
**Packages:** `ParallelStencil`, `Plots`, `GLMakie`, `Printf`, `Test`, `CUDA`, `JLD2`

in folders : `scripts` and `vis_scripts_data` have their own environments activate them before running

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
│ ├─ Results in png and mp4             # results of the simulation
├─ test                    
│ ├─ runtests.jl                          
│ ├─ test2D.jl                              
│ ├─ test3D.jl                         
├─ vis_scripts_data                     # JLD2 files for daint->remote its own project with GLMakie
└─ README.md                            # report
```

**scripts:** contain the actual source code for this project.

```bash
├─ scripts                             # its own project with Plots, CairoMakie and so on.
│  ├─ Pf_diffusion_2D_perf_xpu.jl      # Task 1.2: parallel indices
│  ├─ Pf_diffusion_2D_xpu.jl           # Task 1.1: ParallelStencil
│  ├─ PorousConvection_2D_xpu_test.jl  # Low-res 2D + unit tests 
│  ├─ PorousConvection_2D_xpu.jl       # 2D porous convection (CPU/GPU)
│  ├─ PorousConvection_3D_xpu_test.jl  # Low-res 3D + unit tests 
│  ├─ PorousConvection_3D_xpu.jl       # 3D porous convection (CPU/GPU)
│  ├─ PorousConvection_2D.jl           # Verified 2D reference (from HW5)
```
Note: The `_test.jl` variants are numerically identical to their counterparts; they differ only in parameters (grid size, time steps) and include extra helpers for unit testing.

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

We provide unit tests (kernel-level checks) and reference tests (end-to-end) for both 2D and 3D:

**Unit tests:** Validate individual kernels for correctness.

**2D reference test:** Compares the full 2D XPU simulation against the verified 2D reference solver (PorousConvection_2D.jl from HW5).

**3D reference test:** For low Rayleigh number (e.g., Ra = 10), theory predicts diffusion-dominated behavior (no convection). The test verifies that the numerical solution exhibits this regime.

To run the test go into Pkg mode run command `test`
