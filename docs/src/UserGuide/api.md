# API Reference
IFTDuals exports the following functions

### Main function 
```@docs
ift
```
The `ift` function is the core function for computing higher-order derivatives. Using this function will work for most use cases without needing to overload the utility functions. It is a requirement that all functions called on `y` and `args` use `AbstractArray` annotations and not concrete `Vector`, `Matrix` or `Array` types. Internally `ift` calls some functions which convert arrays of duals into `AbstractArray` types, to avoid allocating new arrays and hence improve performance.

The `ift` function has several optional keyword arguments to optimize performance in specific scenarios. These are described in detail in the [Advanced Usage](@ref) section of the user guide.

### Utility functions
```@docs
pvalue
nested_pvalue
promote_common_dual_type
promote_my_type
```

## Implicit function signature requirements
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
