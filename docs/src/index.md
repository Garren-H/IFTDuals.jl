# IFTDuals.jl

IFTDuals.jl is a lightweight Julia package for computing higher-order derivatives of functions implicitly defined through the Implicit Function Theorem (IFT) using dual numbers. The package enables automatic differentiation for implicit relationships where ``x=g(\theta)`` is defined implicitly through ``f(g(\theta), \theta) = 0``.

## Overview
 
The IFT provides a way to compute derivatives of implicitly defined functions. IFTDuals.jl leverages ForwardDiff's dual number system to efficiently compute higher-order derivatives by recursively applying the IFT formulation. Generically, the IFT gives the Kᵗʰ order derivative of ``x`` as:

```math
\frac{\partial^K x}{\partial \theta^K} = -\left[\frac{\partial f}{\partial x}\right]^{-1} B_{K}
```

Where ``B_K``, depends on all order of derivatives of `y` up to order `K-1`. All previous derivatives hence needs to be computed and stored to compute the next order derivative. IFTDuals.jl obtains the ``B_K`` terms recursively, solves for the derivatives and converts the results back into dual numbers. This process is repeated recursively to obtain derivatives of arbitrary order.

## Installation

You can install IFTDuals.jl via Julia's package manager. In the Julia REPL, enter the package manager by pressing `]` and then run:

```julia
pkg> add IFTDuals
```

Alternatively, you can install directly from the REPL:

```julia
using Pkg
Pkg.add("IFTDuals")
```

## Quick Start

Here is a minimal example demonstrating how to use IFTDuals.jl:

```julia
using IFTDuals
using DifferentiationInterface
import ForwardDiff

# Define the implicit function f(x, θ) = 0
f(x, θ) = ...

# Define a function that solves for y and computes derivatives
function solve_x(θ)
    θ_primal = nested_pvalue(θ) # Extract primal value, stripping all dual parts
    x = root_solver(f, θ_primal)  # Use any root solver to find x such that f(x, θ_primal) = 0
    return ift(x, f, θ, θ_primal) # Compute derivatives if θ contains duals, otherwise return x
end

θ = 2.0  # Input parameter

value_derivative_and_second_derivative(solve_x, AutoForwardDiff(), θ) # computes derivatives using existing frameworks
```

## Contributing
Contributions to IFTDuals.jl are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Garren-H/IFTDuals.jl/issues). If you'd like to contribute code, feel free to fork the repository and submit a pull request.
