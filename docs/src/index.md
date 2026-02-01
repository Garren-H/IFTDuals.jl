# IFTDuals.jl

IFTDuals.jl is a lightweight Julia package for computing higher-order derivatives of functions implicitly defined through the Implicit Function Theorem (IFT) using dual numbers. The package enables automatic differentiation for implicit relationships where ``x=g(\theta)`` is defined implicitly through ``f(g(\theta), \theta) = 0``.

## Overview

The Implicit Function Theorem provides a way to compute derivatives of implicitly defined functions. IFTDuals.jl leverages ForwardDiff's dual number system to efficiently compute higher-order derivatives by recursively applying the IFT formulation.

## Limitations

### Mixed tags
The package supports two differentiation modes:
- **Single tag (symmetric)**: All variables use the same tag. This assumes `partials.value == value.partials` (symmetric derivative structures). This is the default and most efficient mode, suitable for computing derivatives like d²g/dx₁² and d²g/dx₂².
- **Mixed tags**: Different variables each have their own tag. This is useful when you need cross-derivatives such as d²g/dx₁dx₂ and not d²g/dx₁² nor d²g/dx₂². While mixed tags work correctly, they incur performance penalties when computing multiple directional derivatives due to the additional bookkeeping required to track different tags through the recursive process.

For most applications, using a single tag (concatenating all parameters into a single vector) will provide the best performance.

### Mutation of arguments
Variables being differentiated (i.e., `args` passed to `ift`) should **not be mutated** within your function `f` or any intermediate computations. The `args` structure is reused throughout the recursive process when computing higher-order derivatives. While mutation may work correctly for first-order Duals, it will lead to incorrect results for higher-order derivatives (second-order and above).

**Recommendation**: If your function requires mutating the input arguments, create an intermediate function that copies `args` before passing it to your computation:

```julia
function f_safe(y, args)
    args_copy = deepcopy(args)  # Create a copy to avoid mutation issues
    # Now safe to mutate args_copy
    return f_original(y, args_copy)
end

# Use f_safe instead of f_original in ift
y_dual = ift(y, f_safe, args)
```

Note that this approach may incur a performance penalty due to the copying overhead.

### Function signature
The `ift` function only supports the two-argument function signature `f(x, args)`. If your implicit function requires multiple separate parameter arguments, you should define `args` as a tuple and use splatting within your function:

```julia
# If you have a function with multiple parameters
function my_function(x, α, β, γ)
    return ... # some operation
end

# Wrap it to use the f(y, theta) signature
f(x, args) = my_function(x, args...)

# Define theta as a tuple of parameters
args = (α, β, γ)

# Now you can use ift
x_dual = ift(x, f, args)
```

### Automatic differentiation backend
This package was developed specifically to work with ForwardDiff.jl and its dual number implementation. It is not designed to work with other automatic differentiation packages (such as ReverseDiff.jl, Zygote.jl, or Enzyme.jl). All differentiation operations should be performed using ForwardDiff.jl.

### Array type annotations
If `args` (and/or `x`) contain arrays with dual numbers, any functions or methods you define should use `AbstractArray` type annotations rather than concrete `Array` types. This is because IFTDuals.jl uses intermediate `AbstractArray` subtypes (such as `PValueArray`, `NestedPValueArray`, `SeedDualArray`, and `PartialsArray`) for efficient non-allocating operations on dual number arrays.

**Recommendation**: Use `AbstractArray`, `AbstractVector`, or `AbstractMatrix` in your type signatures:

```julia
# Good - works with IFTDuals wrapper types
function my_function(y::AbstractVector, args::AbstractVector)
    # ... your code
end

# Bad - may fail with IFTDuals wrapper types
function my_function(y::Vector, args::Vector)
    # ... your code
end
```

This ensures compatibility with the wrapper types used internally by the package for performance optimization.

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

# Define the implicit function f(y, θ) = 0
f(y, θ) = y^3 + θ*y - 1

# Define a function that solves for y and computes derivatives
function solve_y(θ)
    θ_primal = nested_pvalue(θ) # Extract primal value, stripping all dual parts
    y = root_solver(f, θ_primal)  # Use any root solver to find y such that f(y, θ_primal) = 0
    return ift(y, f, θ) # Compute derivatives if θ contains duals, otherwise return y
end

θ = 2.0  # Input parameter
dy_dθ = ForwardDiff.derivative(solve_y, θ) # First derivative
d2y_dθ2 = ForwardDiff.derivative(θ -> ForwardDiff.derivative(solve_y, θ), θ) # Second derivative
```

## Key Concepts

### Implicit Function Theorem

For a system of equations ``f(y, \theta) = 0`` where ``y`` depends implicitly on ``\theta``, the IFT gives us:

```math
\frac{\partial y}{\partial \theta} = -\left[\frac{\partial f}{\partial y}\right]^{-1} \frac{\partial f}{\partial \theta}
```

IFTDuals.jl recursively applies this relationship to compute higher-order derivatives efficiently using nested dual numbers.

### Dual Numbers

The package relies on ForwardDiff's dual number representation to propagate derivatives through the implicit relationship. When you pass dual numbers as arguments, the `ift` function automatically computes the appropriate derivatives and reconstructs the dual number structure.

## Exported Functions

IFTDuals.jl exports five main functions:

- **`ift`**: Computes higher-order derivatives using the Implicit Function Theorem
- **`pvalue`**: Extracts the primal value from dual numbers
- **`nested_pvalue`**: Recursively extracts the innermost primal value from nested duals
- **`promote_common_dual_type`**: Promotes all dual numbers to a common type
- **`promote_my_type`**: Extracts and promotes the underlying numeric type from data structures

See the [API Reference](@ref) for detailed documentation of each function.

## Basic Usage Examples

### Scalar Case

```julia
using IFTDuals
import ForwardDiff

# Implicit equation: y^2 + θ*y - 1 = 0
f(y, θ) = y^2 + θ*y - 1

function get_y(θ)
    θ_p = nested_pvalue(θ)
    # Solve for y (using any root finding method)
    y = (-θ_p + sqrt(θ_p^2 + 4)) / 2
    return ift(y, f, θ)
end

# Compute first derivative
dy_dθ = ForwardDiff.derivative(get_y, 1.0)

# Compute second derivative
d2y_dθ2 = ForwardDiff.derivative(θ -> ForwardDiff.derivative(get_y, θ), 1.0)
```

### Vector Case

```julia
using IFTDuals
import ForwardDiff

# System of equations: f(y, θ) = 0
function f(y, θ)
    return [
        y[1]^2 + y[2] - θ[1],
        y[1] + y[2]^2 - θ[2]
    ]
end

function solve_system(θ)
    θ_p = nested_pvalue(θ)
    # Solve the system for y (using any solver)
    y = ... # your solver here
    return ift(y, f, θ)
end

θ = [1.0, 2.0]
J = ForwardDiff.jacobian(solve_system, θ)  # Jacobian matrix
```

## Working with Custom Structs

For custom data structures passed as arguments to `ift`, it's highly recommended to provide custom implementations of `pvalue`, `nested_pvalue`, `promote_common_dual_type`, and `promote_my_type`:

```julia
struct MyParams{T<:Real}
    a::Vector{T}
    b::T
    c::String  # Non-differentiable field
end

# Extract primal value
pvalue(p::MyParams{T}) where T = MyParams{pvalue(T)}(pvalue(p.a), pvalue(p.b), p.c)

# Extract nested primal value
nested_pvalue(p::MyParams{T}) where T = MyParams{nested_pvalue(T)}(
    nested_pvalue(p.a), nested_pvalue(p.b), p.c
)

# Promote to common dual type
promote_common_dual_type(p::MyParams{T}, DT::Type{<:Dual}) where T = 
    MyParams{DT}(promote_common_dual_type(p.a, DT), promote_common_dual_type(p.b, DT), p.c)

# Already correct type (optimization)
promote_common_dual_type(p::MyParams{T}, ::Type{T}) where T<:Dual = p

# Extract numeric type (similar to eltype)
promote_my_type(p::MyParams{T}) where T = T
promote_my_type(::Type{MyParams{T}}) where T = T
```

## Performance Considerations

- The package uses non-allocating wrapper types (`PValueArray`, `NestedPValueArray`, `SeedDualArray`, `PartialsArray`) to efficiently extract and manipulate dual number components without unnecessary allocations.
- For vector-valued functions, the Jacobian matrix is computed once and factored using LU decomposition for efficient repeated solves during derivative computation.
- Custom implementations of `pvalue`, `nested_pvalue`, and `promote_my_type` for your data structures can significantly improve performance.

## Contributing

Contributions to IFTDuals.jl are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Garren-H/IFTDuals.jl/issues). If you'd like to contribute code, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
