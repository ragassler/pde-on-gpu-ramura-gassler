# pde-on-gpu-ramura-gassler

> Private student repository for the ETH Zürich course  
> **Solving Partial Differential Equations in Parallel on GPUs**.  
> Language: **Julia** · Focus: **GPU-accelerated, parallel PDE solvers** and reproducible experiments.

---

## 🎯 Goals

- Implement and benchmark PDE solvers for the weekly homeworks (diffusion, advection–diffusion, Burgers’, Poisson, Navier–Stokes, … as assigned).  
- Use **Julia** with **GPU acceleration** and parallelism (threads, CUDA, MPI).  
- Practice clean, testable scientific software and reproducible runs.

---

## 🧰 Tech stack

- Julia: version `1.10` or newer recommended  
- Packages used across the repository (depending on the folder):  
  `ParallelStencil`, `CUDA`, `ImplicitGlobalGrid`, `Plots`, `GLMakie`, `Printf`, `Test`, `JLD2`, plus standard libraries.

Each main folder (`Homework-X`, `PorousConvection`, `Literate`, …) has its **own project environment** (`Project.toml`).  
Always activate the corresponding environment before running code there.

---

## 🚀 Getting started

1. Clone the repository  
   ```bash
     git clone https://github.com/ragassler/pde-on-gpu-ramura-gassler.git  
     cd pde-on-gpu-ramura-gassler
   ```

2. Pick a folder (homework or final project)  
   Example: Homework 3  
   ```bash
     cd Homework-3  
     julia --project
   ```

3. Instantiate the environment (first time only)
   ```julia
      using Pkg 
      Pkg.instantiate()
   ```

5. Run the code  
   Each folder has its own `README.md` explaining:
   - which script to run (for example `diffusion_2D_xpu.jl`, `burgers_1D.jl`, …),  
   - how to switch between CPU and GPU,  
   - how to reproduce the plots.

---


## ✅ Running tests

Many folders (especially `PorousConvection/` and the later homeworks) define tests.

Example for the final project:

1. In a terminal:  
   - `cd PorousConvection`  
   - `julia --project`

2. Inside Julia:  
   - `using Pkg`  
   - `Pkg.test()`

Continuous integration is configured via GitHub Actions (see the CI badge on GitHub) and runs the test suites automatically.

---

If you want to jump straight to the final project, open `PorousConvection/README.md` for the full description, governing equations, numerical method, and animations.
