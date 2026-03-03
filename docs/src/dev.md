# How IFTDuals work
As mentioned, the IFT gives a generic way in which one can compute higher-order derivatives of implicit functions.

```math
\frac{d^K y}{d \theta^K} = - \left( \frac{\partial f}{\partial y} \right)^{-1} B_K
```

The core idea of IFTDuals.jl is using dual numbers (specifically ForwardDiff.jl) and starting from the inner most Dual, compute ``B_K`` using AD, solve for the derivative, convert the result back into a dual and repeat the process until all dual levels have been processed.

To achieve this, all that is required is to seed `y` appropriately at each level and ``B_K`` will be recovered as the partials field. That is, for the inner most dual (1st order derivative), `y` is seeded with zero (or just using the primal value), resulting in ``B_1`` being given as
```math
B_1 = \frac{\partial f}{\partial \theta}
```

Now we reconstruct the dual number for the first order derivative
```julia
y_dual1 = DT1(y, PT1((dy_dtheta,))) # DT1 -> Dual Type for level 1, PT1 -> Partial Type for level 1
```

Now to obtain the second order derivative, we need to seed `y_dual1` appropriately. If the partials are symmetric, then we seed
```julia
y_dual2_star = DT2(y_dual1, PT2(ntuple(j -> DT1(y_dual.partials[j]),length(y)))) # DT2 -> Dual Type for level 2, PT2 -> Partial Type for level 2
```

The evaluation of `f` at `y_dual2_star` and the second order dual `args` will yield ``B_2`` as the `partials.partials` field (this can easily be proved by performing the Dual arithmetic). We can then solve for the second order derivative, reconstruct the dual and repeat the process for higher order derivatives. Note that for symmetry of higher partials, we need to seed all partials.value fields with the correct derivatives computed. I.e. for a third order dual, we would need to seed the `partials.value.value` field with the first order derivatives computed, the `partials.partials.value` field with the second order derivatives computed and finally compute ``B_3``, extract this as the `partials.partials.partials` field, solve the system and reconstruct the third order dual. This process is implemented recursively.

When we do not have symmetry of partials, we are still able to recover the correct ``B_K`` terms, but involves a bit more bookkeeping. If we have two parameters, `θ₁` and `θ₂`, then for the second order dual, where `θ₁` is the inner dual and `θ₂` the outer dual. The first step would be convert `θ₁` and `θ₂` into duals of the same type, i.e. both second order duals where the inner dual has zero partials for `θ₂` and the outer dual has zero partials for `θ₁`. This is easily achieved with `Base.promote`. The first order dual is then extracted and the above mentioned procedure is followed to obtain `dy/dθ₁`. To obtain `dy/dθ₂`, we only need to evaluate `f` at the second order dual (irrespective of whether `y` is seeded with `dy/dθ₁` or zero) and extract the `partials.value` field to obtain ``B_1`` for `θ₂`. Following the Dual arithmetic one will note that the `value.partials` and `partials.value` fields are independent of each other. Once both `dy/dθ₁` and `dy/dθ₂` have been computed, we can reconstruct the second order dual with the correct partials, with zeroed `partials.partials` field. 
```
y_dual2_star = DT2(DT1(y,PT1((dy_dtheta1,))),PT2((DT1(dy_dtheta2),)))
```
Now ``B_2`` can be obtained by evaluating `f` at `y_dual2_star` and the second order dual `args`, with ``B_2`` being extracted as the `partials.partials` field. `d²y/dθ₁dθ₂` is then solved for and the second order dual is reconstructed with the correct partials field. For third order duals the same process is followed, first the `partials.value.value` and `partials.value.partials` fields are obtained, followed by the `partials.partials.value` field and finally ``B_3`` is obtained from the `partials.partials.partials` field. This process naturally has a recursive structure where two branches of recursion are required, one to solve for the `partials.value` fields and another to solve for the `partials.partials` fields. 
