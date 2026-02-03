# IFTDuals.jl

IFTDuals.jl is a lightweight Julia package for computing higher order derivatives of a function, $x=g(\theta)$, implicitly defined through the implicit function theorem (IFT) using dual numbers. IFT gives the relationship between $g$ and $\theta$ implicitly through the solution of the equation $f\left(g(\theta\right), \theta) = 0$. This package currently only supports derivatives obtained through [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)

## Installation
You can install IFTDuals.jl via Julia's package manager. In the Julia REPL, enter the package manager by pressing `]` and then run:

```julia
pkg> add IFTDuals
```

## Usage
Here is a simple example demonstrating how to use IFTDuals.jl to compute higher order derivatives using the implicit function theorem. For more detailed discussions, refer to the [documentation](https://garren-h.github.io/IFTDuals.jl).

```julia
using IFTDuals
using DifferentiationInterface
import ForwardDiff

# Define the implicit function f(x, θ) = 0
f(x, θ) = ...

function get_x(θ)
    θ_primal = nested_pvalue(θ)
    x = root_solver(f, θ_primal)  # Any root solver that finds the primal value of x such that f(x, θ_primal) = 0
    return ift(x, f, θ, θ_primal) # compute derivatives if duals are present
end

grad = second_derivative(get_x, AutoForwardDiff(), θ) # use existing interfaces to compute derivatives 
```

## Alternatives
[ImplicitDifferentiation.jl](https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl) is a different Julia package which provides the functionality to compute **first order** derivatives implicitly using any AD backend. 

## Contributing
Contributions to IFTDuals.jl are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Garren-H/IFTDuals.jl/issues). If you'd like to contribute code, feel free to fork the repository and submit a pull request.
