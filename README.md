# IFTDuals.jl

IFTDuals.jl is a lightweight Julia package for computing higher order derivatives of a function, $x=g(\theta)$, implicitly defined through the implicit function theorem (IFT) using dual numbers. IFT gives the relationship between $g$ and $\theta$ implicitly through the solution of the equation $f\left(g(\theta\right), \theta) = 0$. This package currently only supports derivatives obtained through [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)

## Installation
You can install IFTDuals.jl via Julia's package manager. In the Julia REPL, enter the package manager by pressing `]` and then run:

```julia
pkg> add IFTDuals
```

## Usage
Using IFTDuals.jl is fairly straightforward. We export 5 functions, `ift`, `pvalue`, `nested_pvalue`, `promote_common_dual_type` and `promote_my_type`.
- `ift(y::Union{V,<:AbstractVector{V}},f::Function,args) where {V<:Real}`: Computes the higher order derivatives of `y` wrt the parameters defined in `args`, with the relationship defined implicitly through `f(y, args) = 0`. Here `args` can be any data structure (e.g., scalar, vector, tuple, struct, etc.) containing `Dual` numbers. If `args` does not contain any `Dual` numbers, we return `y` as is.

- `pvalue(x::T) where T<:Real`: Extracts the value fields from a data structure `x` containing `Dual` numbers. If `T<:Dual`, it return `ForwardDiff.value(x)`.

- `nested_pvalue(x::T) where T<:Real`: Similar to `pvalue`, but recursively extracts the value fields to obtain the primal value. 

- `promote_common_dual_type(args,DT::Type{<:Dual})`: Promotes all `Dual` numbers in the data structure `args` to have the same type, which is the common supertype of all `Dual` numbers in `args`. This is useful when you have multiple `Dual` types in `args` and want to ensure they are all of the same type for compatibility. I.e. `args` may contain for instance first order, second order and third order (nested) `Dual` numbers (with the same tag). This function promotes all these variables to a common dual type, `DT`.

- `promote_my_type(x::T) where T`: This function signature acts similarly to `Base.eltype`. It should be used to extract the underlying numeric data type from a data structure `x`. For non-numeric data structures, it should return `Nothing`. Internally, when calling `promote_my_type(args)`, we extract the underlying numeric type `T` from all numeric types in `args`, promote them to a common (`Dual`) supertype and return that type. The returned type is then used in `promote_common_dual_type` to promote all `Dual` numbers in `args` to have the same type. This functions is hence a combination of `Base.eltype` and `Base.promote_type` but coded with the intention to be non-allocating.

We provide default implementations to handle custom data structures for the above methods, however these are not garanteed to work nor be performant. it is highly recommended to provide custom implementations (excluding `ift`). 

### Example
Here is a simple example demonstrating how to use IFTDuals.jl to compute higher order derivatives using the implicit function theorem.

```julia
using IFTDuals
using DifferentiationInterface
import ForwardDiff

# Define the implicit function f(x, θ) = 0
f(x, θ) = x^3 + θ*x - 1

function get_x(θ)
    θ_primal = nested_pvalue(θ)
    x = root_solver(f, θ_primal)  # Any root solver that finds the primal value of x such that f(x, θ_primal) = 0
    return ift(x, f, θ, θ_primal) # compute derivatives if duals are present
end

grad = second_derivative(get_x, AutoForwardDiff(), θ) # or derivative, jacobian and/or hessian should work.
```

#### Custom structs overloads
If you have custom structs passed as `args`, it is highly advised to provide custom implementations for the methods, `pvalue`, `nested_pvalue`, `promote_common_dual_types` and `promote_my_type`. We do attempt to provide generic implementations for these methods but no garantees are made that these will work for your custom structs, or be performant.

```julia
struct MyParams{T<:Real}
    a::AbstractVector{T}
    b::T
    c::String # variable that does not require differentiation
end

pvalue(p::MyParams{T}) where T = MyParams{pvalue(T)}(pvalue(p.a), pvalue(p.b), p.c)
nested_pvalue(p::MyParams{T}) where T = MyParams{nested_pvalue(T)}(nested_pvalue(p.a), nested_pvalue(p.b), p.c)
promote_common_dual_type(p::MyParams{T}, DT::Type{<:Dual}) where T = MyParams{DT}(promote_common_dual_type(p.a, DT), promote_common_dual_type(p.b, DT), p.c) # promote Dual to type DT
promote_common_dual_type(p::MyParams{T}, T::Type) where T = p # already target dual, overload for efficiency
promote_my_type(p::MyParams{T}) where T = T # similar to Base.eltype
promote_my_type(::Type{MyParams{T}}) where T = T # might be needed if args contains NTuple{N,MyParams{T}} types
```

## Alternatives
[ImplicitDifferentiation.jl](https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl) is a different Julia package which provides the functionality to compute first order derivatives implicitly using any AD backend. 

## Contributing
Contributions to IFTDuals.jl are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/Garren-H/IFTDuals.jl/issues). If you'd like to contribute code, feel free to fork the repository and submit a pull request.
