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

**Performance benefit**: This eliminates the overhead of calling `nested_pvalue(args)` inside `ift`, which can be significant for complex nested structures or when `ift` is called repeatedly.

### `DT::Union{Nothing,Type{<:Dual}}` - Target dual type

By default, `ift` automatically detects the dual type from `args`. If you already know the target dual type, you can specify it to skip the type detection step:

```julia
using ForwardDiff

# Manually specify the dual type
DT = typeof(ForwardDiff.Dual(1.0, 1.0))
y_dual = ift(y, f, args; DT=DT)
```

**When to use**: When you're repeatedly calling `ift` with the same dual type structure and want to avoid the overhead of type detection.

### `tag_is_mixed::Union{Nothing,Bool}` - Whether mixed tags are present

By default, `ift` checks whether `args` contains mixed tags (different variables with different tags). If you know whether your setup uses mixed tags, you can specify this to skip the check:

```julia
# For single tag (symmetric) mode
y_dual = ift(y, f, args; tag_is_mixed=false)

# For mixed tags mode (cross-derivatives)
y_dual = ift(y, f, args; tag_is_mixed=true)
```

**Important details**:

1. **Performance with `DT`**: Even when you specify `tag_is_mixed`, the package still validates this against the dual types in `args`. For maximum performance, you should also specify `DT` to avoid the dual type detection step:
   ```julia
   # Most efficient approach
   y_dual = ift(y, f, args; DT=DT, tag_is_mixed=false)
   ```

2. **Internal validation**: If `DT` is not provided, it will be computed internally from `args` and then used to validate the `tag_is_mixed` parameter.

3. **Error conditions**: If mixed tags are detected in `DT` (different function signatures in tag types) but the partials fields have different numbers of entries, an error will be thrown. This helps catch configuration mistakes.

4. **Valid usage of `tag_is_mixed=false`**: Setting `tag_is_mixed=false` is only valid when all partials fields in the nested dual structure have the same number of partials. This ensures the symmetric derivative structure assumption holds throughout the recursion.

**When to use**:
- Set to `false` when working with single-tag (symmetric) derivatives for optimal performance. This is the most common case and uses the more efficient implementation.
- Set to `true` when computing cross-derivatives like d²g/dx₁dx₂ where you need derivatives with respect to different variables (each having its own tag).

### `args_needs_promotion::Bool` - Whether args need promotion to common dual type (default: `true`)

When computing higher-order derivatives, `ift` may need to promote `args` to a common dual type. If you've already ensured all duals in `args` have the same type, you can skip this promotion step:

```julia
# If args already has a common dual type
y_dual = ift(y, f, args; args_needs_promotion=false)
```

**When to use**: When you've already ensured type consistency in `args` and want to avoid the promotion overhead.

## Performance Examples

### Basic Optimization

```julia
using IFTDuals
import ForwardDiff

f(y, θ) = y^3 + θ*y - 1

function solve_y_optimized(θ)
    θ_primal = nested_pvalue(θ)
    y = solve_for_y(θ_primal)
    
    # Pass θ_primal to avoid recomputation and specify single tag mode
    return ift(y, f, θ, θ_primal; tag_is_mixed=false)
end
```

### Second-Order Derivatives with Full Optimization

```julia
function solve_y_second_order(θ)
    θ_primal = nested_pvalue(θ)
    y = solve_for_y(θ_primal)
    
    # Get the dual type from θ
    DT = typeof(θ)
    
    # Skip all auto-detection for maximum performance
    return ift(y, f, θ, θ_primal; DT=DT, tag_is_mixed=false, args_needs_promotion=false)
end

# Compute second derivative
θ = 2.0
d2y_dθ2 = ForwardDiff.derivative(θ -> ForwardDiff.derivative(solve_y_second_order, θ), θ)
```

### Vector System with Optimization

```julia
function f(y, θ)
    return [
        y[1]^2 + y[2] - θ[1],
        y[1] + y[2]^2 - θ[2]
    ]
end

function solve_system_optimized(θ)
    θ_primal = nested_pvalue(θ)
    y = numerical_solver(f, θ_primal)
    
    # For vector systems, optimization can provide significant speedups
    return ift(y, f, θ, θ_primal; tag_is_mixed=false)
end
```

## Benchmarking Example

Here's how to benchmark the difference between auto-detection and manual specification:

```julia
using BenchmarkTools
using IFTDuals
import ForwardDiff

f(y, θ) = y^3 + θ*y - 1

function solve_y_auto(θ)
    θ_primal = nested_pvalue(θ)
    y = θ_primal^(1/3)  # Simplified solver
    return ift(y, f, θ)  # Auto-detect everything
end

function solve_y_manual(θ)
    θ_primal = nested_pvalue(θ)
    y = θ_primal^(1/3)
    DT = typeof(θ)
    return ift(y, f, θ, θ_primal; DT=DT, tag_is_mixed=false, args_needs_promotion=false)
end

θ = 2.0

# Benchmark auto-detection
@benchmark ForwardDiff.derivative(solve_y_auto, $θ)

# Benchmark manual specification
@benchmark ForwardDiff.derivative(solve_y_manual, $θ)
```

## Recommendations

1. **Start simple**: Begin with the basic `ift(y, f, args)` call without keyword arguments. The auto-detection is fast and correct.

2. **Always pass `args_primal`**: Since you typically need to compute `nested_pvalue(args)` to solve for the primal solution anyway, passing it to `ift` is essentially free and avoids redundant computation:
   ```julia
   args_primal = nested_pvalue(args)
   y = solve_for_y(args_primal)
   return ift(y, f, args, args_primal)  # Recommended pattern
   ```

3. **Profile first**: Only add other optimization keywords after profiling shows that type detection is a bottleneck.

4. **Use `tag_is_mixed=false` liberally**: If you're using single-tag mode (most common case), setting this to `false` is generally safe and can provide noticeable speedups.

5. **Combine optimizations**: For maximum performance in tight loops, combine all optimizations when you have complete knowledge of your dual type structure:
   ```julia
   return ift(y, f, args, args_primal; DT=DT, tag_is_mixed=false, args_needs_promotion=false)
   ```

6. **Be cautious with `args_needs_promotion=false`**: Only use this when you're certain all duals in `args` already have the same type, or you may get incorrect results.

## When Auto-Detection is Preferred

In most cases, the auto-detection overhead is negligible compared to the computational cost of:
- Solving the implicit equation
- Computing Jacobians
- LU decomposition (for vector systems)

Use auto-detection (the default) when:
- You're computing derivatives only once or a few times
- Your implicit function solve is expensive (most common case)
- Code clarity is more important than micro-optimizations
- You're prototyping or developing new models

The keyword arguments are most beneficial when:
- Calling `ift` in tight loops with many iterations
- The implicit solve is very cheap
- You've profiled and identified type detection as a bottleneck
- You're implementing high-performance libraries
