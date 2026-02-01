# Examples and Tutorials

This page provides detailed examples demonstrating various use cases of IFTDuals.jl.

## Example 1: Simple Scalar Implicit Function

Consider the cubic equation:

```math
f(y, \theta) = y^3 + \theta y - 1 = 0
```

We want to compute derivatives of the solution `y` with respect to `θ`.

```julia
using IFTDuals
import ForwardDiff

# Define the implicit equation
f(y, θ) = y^3 + θ*y - 1

# Solve for y given θ (using Newton's method as an example)
function solve_for_y(θ)
    y = 1.0  # Initial guess
    for _ in 1:10
        y = y - f(y, θ) / (3y^2 + θ)
    end
    return y
end

# Wrapper function that computes derivatives via IFT
function solve_y_with_derivatives(θ)
    θ_primal = nested_pvalue(θ)
    y = solve_for_y(θ_primal)
    return ift(y, f, θ)
end

# Compute first derivative
θ = 2.0
dy_dθ = ForwardDiff.derivative(solve_y_with_derivatives, θ)

# Compute second derivative
d2y_dθ2 = ForwardDiff.derivative(θ -> ForwardDiff.derivative(solve_y_with_derivatives, θ), θ)

println("dy/dθ = ", dy_dθ)
println("d²y/dθ² = ", d2y_dθ2)
```

## Example 2: System of Nonlinear Equations

Consider a system of two equations with two unknowns:

```math
\begin{align}
f_1(y_1, y_2, \theta_1, \theta_2) &= y_1^2 + y_2 - \theta_1 = 0 \\
f_2(y_1, y_2, \theta_1, \theta_2) &= y_1 + y_2^2 - \theta_2 = 0
\end{align}
```

```julia
using IFTDuals
import ForwardDiff
using LinearAlgebra

# Define the system of equations
function f(y, θ)
    return [
        y[1]^2 + y[2] - θ[1],
        y[1] + y[2]^2 - θ[2]
    ]
end

# Solve the system using Newton's method
function solve_system(θ)
    y = [1.0, 1.0]  # Initial guess
    
    for _ in 1:20
        J = [2*y[1] 1.0; 1.0 2*y[2]]  # Jacobian
        residual = f(y, θ)
        y = y - J \ residual
        
        if norm(residual) < 1e-10
            break
        end
    end
    
    return y
end

# Wrapper for IFT
function solve_with_derivatives(θ)
    θ_primal = nested_pvalue(θ)
    y = solve_system(θ_primal)
    return ift(y, f, θ)
end

# Compute Jacobian matrix dy/dθ
θ = [1.0, 2.0]
J = ForwardDiff.jacobian(solve_with_derivatives, θ)

println("Jacobian dy/dθ:")
println(J)
```

## Example 3: Optimization Problem via KKT Conditions

Consider a constrained optimization problem where the solution satisfies the KKT conditions. We can use IFT to compute derivatives of the optimal solution with respect to problem parameters.

```julia
using IFTDuals
import ForwardDiff

# Problem: minimize (x - θ)² subject to x ≥ 0
# KKT condition: f(x, λ, θ) = [2(x - θ) + λ; λ*x] = 0 with λ ≥ 0, x ≥ 0

function kkt_conditions(z, θ)
    x, λ = z[1], z[2]
    return [
        2*(x - θ) + λ,
        λ*x
    ]
end

function solve_kkt(θ)
    # For this simple problem, we know the solution analytically
    if θ ≥ 0
        return [θ, 0.0]  # x = θ, λ = 0 (interior solution)
    else
        return [0.0, -2*θ]  # x = 0, λ = -2θ (boundary solution)
    end
end

function optimal_x_with_derivatives(θ)
    θ_primal = nested_pvalue(θ)
    z = solve_kkt(θ_primal)
    z_dual = ift(z, kkt_conditions, θ)
    return z_dual[1]  # Return only x (not λ)
end

# Compute derivative of optimal x with respect to θ
θ = 2.0
dx_dθ = ForwardDiff.derivative(optimal_x_with_derivatives, θ)
println("dx*/dθ = ", dx_dθ)  # Should be 1.0 for θ > 0
```

## Example 4: Equilibrium in Economics

Consider a simple supply-demand equilibrium model where equilibrium price `p` and quantity `q` depend on parameters `α` (demand shifter) and `β` (supply shifter):

```math
\begin{align}
\text{Demand:} \quad q &= \alpha - p \\
\text{Supply:} \quad q &= \beta + p \\
\text{Equilibrium:} \quad \alpha - p &= \beta + p
\end{align}
```

```julia
using IFTDuals
import ForwardDiff

# Define equilibrium conditions as implicit equations
function equilibrium(vars, params)
    q, p = vars
    α, β = params
    return [
        q - (α - p),  # Demand equation
        q - (β + p)   # Supply equation
    ]
end

function solve_equilibrium(params)
    α, β = nested_pvalue(params)
    
    # Analytical solution for this simple case
    p = (α - β) / 2
    q = α - p
    
    return ift([q, p], equilibrium, params)
end

# Compute how equilibrium responds to parameter changes
params = [10.0, 4.0]  # α = 10, β = 4
J = ForwardDiff.jacobian(solve_equilibrium, params)

println("Equilibrium quantity and price:")
qp = solve_equilibrium(params)
println("  q = ", nested_pvalue(qp[1]))
println("  p = ", nested_pvalue(qp[2]))

println("\nSensitivity matrix [dq/dα dq/dβ; dp/dα dp/dβ]:")
println(J)
```

## Example 5: Working with Custom Parameter Structures

When working with complex models, it's often convenient to organize parameters in custom structs:

```julia
using IFTDuals
import ForwardDiff

# Define a custom parameter struct
struct ModelParams{T<:Real}
    α::T
    β::Vector{T}
    γ::T
    name::String  # Non-numeric field
end

# Implement required methods for IFTDuals
import IFTDuals: pvalue, nested_pvalue, promote_common_dual_type, promote_my_type

pvalue(p::ModelParams{T}) where T = 
    ModelParams{pvalue(T)}(pvalue(p.α), pvalue(p.β), pvalue(p.γ), p.name)

nested_pvalue(p::ModelParams{T}) where T = 
    ModelParams{nested_pvalue(T)}(nested_pvalue(p.α), nested_pvalue(p.β), nested_pvalue(p.γ), p.name)

promote_common_dual_type(p::ModelParams{T}, DT::Type{<:Dual}) where T = 
    ModelParams{DT}(
        promote_common_dual_type(p.α, DT),
        promote_common_dual_type(p.β, DT),
        promote_common_dual_type(p.γ, DT),
        p.name
    )

promote_common_dual_type(p::ModelParams{T}, ::Type{T}) where T<:Dual = p

promote_my_type(p::ModelParams{T}) where T = T
promote_my_type(::Type{ModelParams{T}}) where T = T

# Now use it in an implicit function
function model_equation(y, params::ModelParams)
    return y^2 + params.α * y + sum(params.β) - params.γ
end

function solve_model(params::ModelParams)
    p_primal = nested_pvalue(params)
    
    # Solve quadratic equation
    a, b, c = 1.0, p_primal.α, sum(p_primal.β) - p_primal.γ
    y = (-b + sqrt(b^2 - 4*a*c)) / (2*a)
    
    return ift(y, model_equation, params)
end

# Create parameters and compute derivatives
params = ModelParams(2.0, [1.0, 2.0], 5.0, "test_model")

# Derivative with respect to α
dy_dα = ForwardDiff.derivative(α -> solve_model(ModelParams(α, params.β, params.γ, params.name)), params.α)
println("dy/dα = ", dy_dα)
```

## Example 6: Higher-Order Derivatives

IFTDuals.jl supports computing higher-order derivatives through nested dual numbers:

```julia
using IFTDuals
import ForwardDiff

f(y, θ) = y^3 - θ*y + 1

function solve_y(θ)
    θ_p = nested_pvalue(θ)
    # Simple fixed-point iteration for this example
    y = 1.0
    for _ in 1:20
        y = cbrt(θ_p * y - 1)
    end
    return ift(y, f, θ)
end

θ = 2.0

# First derivative
dy_dθ = ForwardDiff.derivative(solve_y, θ)

# Second derivative
d2y_dθ2 = ForwardDiff.derivative(θ -> ForwardDiff.derivative(solve_y, θ), θ)

# Third derivative
d3y_dθ3 = ForwardDiff.derivative(
    θ -> ForwardDiff.derivative(
        θ -> ForwardDiff.derivative(solve_y, θ), 
        θ
    ), 
    θ
)

println("First derivative:  dy/dθ = ", dy_dθ)
println("Second derivative: d²y/dθ² = ", d2y_dθ2)
println("Third derivative:  d³y/dθ³ = ", d3y_dθ3)
```

## Example 7: Multi-Dimensional Hessian

Computing the Hessian matrix for a vector of parameters:

```julia
using IFTDuals
import ForwardDiff

function f(y, θ)
    return [
        y[1]^2 + y[2] - θ[1] - θ[2],
        y[1] + y[2]^2 - θ[1]*θ[2]
    ]
end

function solve_for_y(θ)
    θ_p = nested_pvalue(θ)
    # Simplified solver for demonstration
    y = [sqrt(θ_p[1] + θ_p[2]), sqrt(θ_p[1]*θ_p[2])]
    return ift(y, f, θ)
end

θ = [2.0, 3.0]

# Jacobian
J = ForwardDiff.jacobian(solve_for_y, θ)
println("Jacobian:")
println(J)

# Hessian of first component y[1] with respect to θ
H1 = ForwardDiff.hessian(θ -> solve_for_y(θ)[1], θ)
println("\nHessian of y[1]:")
println(H1)

# Hessian of second component y[2] with respect to θ
H2 = ForwardDiff.hessian(θ -> solve_for_y(θ)[2], θ)
println("\nHessian of y[2]:")
println(H2)
```

## Tips and Best Practices

1. **Always extract primal values before solving**: Use `nested_pvalue` to strip all dual number parts before passing to your numerical solver.

2. **Verify the implicit relationship**: Ensure that `f(y, args) ≈ 0` holds at the solution before calling `ift`.

3. **For vector systems**: Make sure your Jacobian matrix is square (number of equations equals number of unknowns).

4. **Custom structs**: Implement the four key methods (`pvalue`, `nested_pvalue`, `promote_common_dual_type`, `promote_my_type`) for better performance.

5. **Numerical stability**: If derivatives seem incorrect, check the conditioning of your Jacobian matrix and the accuracy of your primal solution.

6. **Higher-order derivatives**: The computational cost grows with the order of derivatives, but IFTDuals.jl minimizes allocations through efficient wrapper types.
