# Advanced Usage
The `ift` function accepts several keyword arguments that can improve performance by providing additional information about your differentiation setup. By default, these parameters are auto-detected, but specifying them explicitly can avoid redundant computations.

## Keyword Arguments

```julia
ift(y, f, args; DT=nothing, tag_is_mixed=nothing, args_needs_promotion=true)
ift(y, f, args, args_primal; DT=nothing, tag_is_mixed=nothing, args_needs_promotion=true)
```

### `args_primal` - Primal value of args (positional argument)

The `ift` function can accept the primal value of `args` as a second positional argument. This is particularly useful because you typically need to compute `nested_pvalue(args)` to solve for the primal solution `y` anyway. By passing it to `ift`, you avoid recomputing it internally:

```julia
function solve_y(θ)
    θ_primal = nested_pvalue(θ)  # Compute once
    y = solve_for_y(θ_primal)     # Use to solve for y
    return ift(y, f, θ, θ_primal) # Pass to ift to avoid recomputation
end
```

**Performance benefit**: This eliminates the overhead of calling `nested_pvalue(args)` inside `ift`, which can be significant for complex nested structures.

### `DT::Union{Nothing,Type{<:Dual}}` - Target dual type

By default, `ift` automatically detects the dual type from `args`. If you already know the target dual type, you can specify it to skip the type detection step. This is useful in cases where several variables are passed to `args` for which only one is of Dual type and the type is know.

### `tag_is_mixed::Union{Nothing,Bool}` - Whether tags are mixed (default: `nothing`)
This is only applicable when computing derivatives of order 2 or higher.

Two different implementations for IFT exists depending on the Dual structures/tags used in `args`. If all of the variables share the same Dual Tag, i.e. computing Hessians, second or higher order derivatives for a single variable (or vector/array of variables), the dual structure is symmetric. That is `value.partials` contains the same information as `partials.value`. In this case one can specify `tag_is_mixed=false` to effectively copy `value.partials` into `partials.value`, and hence avoid redundant computations. Important to note is depending on how AD is called, different Dual tags can still be created internally, even when differentiating wrt a single variable. There is no way of identifying in this case whether the tags are truly mixed, hence it is left to the user to ensure this is set correctly. When specifying `tag_is_mixed=false`, an internal check is only conducted on the number of partials per Dual layer, but this is not sufficient to guarantee symmetry. This functionality should hence be used with caution.

If the variables in `args` have different Duals Tags, i.e. having multiple variables to differentiate with respect to, the dual structure is not symmetric. A simple use case here is computing cross-derivatives, e.g. having a function `g(x1, x2)` and wanting to compute `d²g/dx1dx2`. In this case the `value.partials` and `partials.value` fields are assymetric and both needs to be computed seperately. In most cases this will be the default due to the mixed tag creation mentioned above. 

### `args_needs_promotion::Bool` - Whether args need promotion to common dual type (default: `true`)
Internally, `pvalue` is used to extract the `value` field from dual numbers and hence build the Dual from the inner most Dual outwards. This requires that all dual numbers in `args` are of the same type. By default, `ift` will promote all Dual numbers (in all elements of an Array/tuple and/or fields and nested fields in structs) to type `DT`. If you have already ensured that all dual numbers in `args` are of the same type, you can set this parameter to `false` to skip the promotion step and improve performance. This functionality should be used with caution, as providing for instance dual numbers of different tags will to incorrect results or errors.
