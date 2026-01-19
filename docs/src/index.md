# IFTDuals.jl

Welcome to the documentation for IFTDuals.jl, a Julia package for dual number computations in the context of Implicit Function Theorem applications.

## Installation

```julia
using Pkg
Pkg.add("IFTDuals")
```

## Quick Start

```julia
using IFTDuals
import ForwardDiff

f(y,x) = ... # system of equations to solve, f(y=g(x),x) = 0 -> solve y

function solve_y(x)
    xp = nested_pvalue(x) # strips all dual parts
    y = ... # solve for y from f(y,x) using any toolbox/method
    return ift(y,f,x) # construct dual of y if x contains duals else return x
end

x = ... # define input

dy = ForwardDiff.gradient(solve_y, x) # can be hessian, jacobian, derivative
```
