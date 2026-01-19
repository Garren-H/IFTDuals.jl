# IFTDuals.jl

IFTDuals.jl is a lightweight Julia package for computing higher order derivatives for a function, $x=g(\theta)$ implicitly defined through the implicit function theorem (IFT) using dual numbers. IFT gives the relationship between $g$ and $\theta$ implicitly through the equation $f(g(\theta), \theta) = 0$. 

**Note**: We currently do not support mixed-mode AD, such as closures or (nested) Duals with different tags. As a workaround, one can concatenate all parameters into a single vector and use a single Dual tag.

## Installation
You can install IFTDuals.jl via Julia's package manager. In the Julia REPL, enter the package manager by pressing `]` and then run:

```julia
pkg> add IFTDuals
```

## Usage
Using IFTDuals.jl is fairly straightforward. We export 4 functions, `ift`, `pvalue`, `nested_pvalue` and `promote_common_dual_type`.
- `ift(y::Union{V,<:AbstractVector{V}},f::Function,tups) where {V<:Real}`: Computes the higher order derivatives of `y` wrt the parameters defined in `tups`, with the relationship defined implicitly through `f(y, tups) = 0`. Here `tups` can be any data structure (e.g., scalar, vector, tuple, struct, etc.) containing `Dual` numbers. If `tups` does not contain any `Dual` numbers, we return `y` as is.

- `pvalue(x::T) where T`: Extracts the value fields from a data structure `x` containing `Dual` numbers. 

- `nested_pvalue(x::T) where T`: Similar to `pvalue`, but recursively extracts the value fields to obtain the primal value. 

- `promote_common_dual_type(tups)`: Promotes all `Dual` numbers in the data structure `tups` to have the same type, which is the common supertype of all `Dual` numbers in `tups`. This is useful when you have multiple `Dual` types in `tups` and want to ensure they are all of the same type for compatibility. I.e. `tups` may contain for instance first order, second order and third order (nested) `Dual` numbers (with the same tag). This function promotes all these order to a common order (the maximum order found in `tups`).

Providing custom implementations for your custom structs for the methods, `pvalue`, `nested_pvalue` and `promote_common_dual_types` is highly advised. We do provide generic implementations for this, but may not be performant, and may fail.

### Example
Here is a simple example demonstrating how to use IFTDuals.jl to compute higher order derivatives using the implicit function theorem.

```julia
using IFTDuals
using ForwardDiff

# Define the implicit function f(x, θ) = 0
f(x, θ) = x^3 + θ*x - 1

function get_x(θ)
    θ_primal = nested_pvalue(θ)
    x = root_solver(f, θ_primal)  # Any root solver that finds the primal value of x such that f(x, θ_primal) = 0
    return ift(x, f, θ) # compute derivatives if duals are present
end

grad = ForwardDiff.gradient(get_x, θ) # or derivative, jacobian and/or hessien should work.
```

## Contributing
Contributions to IFTDuals.jl are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Garren-H/IFTDuals.jl/issues). If you'd like to contribute code, feel free to fork the repository and submit a pull request.
