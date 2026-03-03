# Contains utils to update Duals number in-place and seed symmetrically when possible to avoid redundant IFT solves.

# utilities
"""
```julia
    unwrap_function(f::ComposedFunction) -> Tuple
    unwrap_function(f) -> f
```
Unwrap a composed function into a Tuple of functions or return the function itself if not composed. Uses `Base.unwrap_composed`.
"""
unwrap_function(f::F) where {F<:ComposedFunction} = Base.unwrap_composed(f) # unwrap as Tuple where inner is the final entry
unwrap_function(f) = f
"""
```julia
    inner_f(f::Tuple) -> Function
```
Extract the innermost function from a Tuple of functions stemming from an unwrapped composed function. The innermost function is always the last element.
"""
inner_f(f::Tuple) = f[end] # inner is always the final function in a unwrapped composed
"""
```julia
    outer_f(f::Tuple) -> Union{Function, Tuple}
```
Extract the outer function(s) from a Tuple of functions stemming from an unwrapped composed function. Returns a single function if only two elements remain, otherwise a sub-tuple.
"""
outer_f(f::Tuple) = length(f) === 2 ? f[1] : f[1:end-1]

# Inplace update of Duals
"""
```julia
    update_dual(x::Dual, val, f::Function)
    update_dual(x::AbstractVector{<:Dual}, val::AbstractArray{<:Real}, f::Function)
```
Update duals in-place by assigning solved directional derivatives into the appropriate field of the Dual structure. The function `f` (composed chain of `pvalue`/`extract_partials_`) determines which nested field to update. Removes the need to allocate new arrays or create wrapper type arrays to store these entries.
"""
update_dual(x::Dual,val::VV,f::F) where {VV,F<:Function} = update_dual_(x,val,unwrap_function(f),())
function update_dual(x::AbstractVector{<:Dual}, val::AbstractArray{<:Real}, f::F) where {F<:Function} # in-place assignment
    check_partials_seed_dims(length(x), (), val, Symbol("Input Vector")) 
    ff = unwrap_function(f)
    for i in eachindex(x)            
        x[i] = update_dual_(x[i], val, ff, (i,))
    end
    return x
end
update_dual_(x::DT,val::VV,::typeof(pvalue),::Tuple{}) where {T,V,N,DT<:Dual{T,V,N},VV<:Real} = DT(V(val),x.partials); #update value field
update_dual_(x::DT,val::VV,::typeof(pvalue),idx::Tuple) where {T,V,N,DT<:Dual{T,V,N},VV<:Real} = DT(V(val[idx...]),x.partials); #update value field; we set the inner most value field, equivalent to setting partials to zero
update_dual_(x::DT,val::VV,::typeof(extract_partials_),idx::Tuple) where {T,V,N,DT<:Dual{T,V,N},VV<:AbstractArray{<:Real}} = (check_partials_seed_dims(N, idx, val, :Partials); DT(x.value,Partials{N,V}(ntuple(i->V(val[idx...,i]),N)))); #update partials field
update_dual_(x::DT,val::VV,::typeof(extract_partials_),idx::Tuple) where {T,V,DT<:Dual{T,V,1},VV<:AbstractArray{<:Real}} = DT(x.value,Partials{1,V}((V(val[idx...]),))); #update partials field, check has already been done
update_dual_(x::DT,val::VV,::typeof(extract_partials_),::Tuple{}) where {T,V,DT<:Dual{T,V,1},VV<:Union{<:Real,AbstractArray{<:Real,0}}} = DT(x.value,Partials{1,V}((V(VV <: AbstractArray ? val[] : val),))); #update partials field
function update_dual_(x::DT,val::VV,f::Tuple,idx::Tuple=()) where {T,V,N,DT<:Dual{T,V,N},VV}
    fi = inner_f(f)
    fo = outer_f(f)
    if fi === pvalue
        return DT(update_dual_(x.value,val,fo,idx), x.partials) # recursively update the value field
    elseif fi === extract_partials_ # we check for symmetry here and then offload correctly if symmetric
        if N === 1 # all val are some nested partials field
            return DT(x.value, Partials{1,V}((update_dual_(x.partials[1], val, fo, idx),)))
        else # multiple partial directions
            check_partials_seed_dims(N, idx, val, :Partials)
            return DT(x.value, Partials{N,V}(ntuple(i->update_dual_(x.partials[i], val, fo, (idx...,i)), N)))
        end
    end
    throw(ArgumentError("Unsupported function: $f")) # nested pvalue not supported here, should always be the final function in the composite
end

# seed symmetric case 
"""
```julia
    seed_symm(x::Union{Dual, AbstractVector{<:Dual}, IFTStruct}, f::Function)
```
Seed duals in-place for the symmetric-tag case by copying `value.partials` into `partials.value`. The function `f` determines the current nesting level to seed at. This avoids redundant IFT solves when tags are identical across Dual layers.
"""
seed_symm(x::DT,f::F) where {DT<:Dual,F<:Function} = seed_symm_(x,unwrap_function(f)) # entry point
function seed_symm(x::AbstractVector{DT},f::F) where {DT<:Dual,F<:Function}
    ff = unwrap_function(f)
    for i in eachindex(x)
        x[i] = seed_symm_(x[i], ff)
    end
    return x
end
seed_symm(x::IFTStruct,f::F) where {F} = (seed_symm(x.y, f);x)
seed_symm_(x::DT,::typeof(identity)) where {T,V,N,DT<:Dual{T,V,N}} = seed_symm_(x) 
seed_symm_(x::DT,::typeof(pvalue)) where {T,V<:Dual,N,DT<:Dual{T,V,N}} = DT(seed_symm_(x.value),x.partials) # symmetric seed dual
seed_symm_(x::DT,f::F) where {T,V<:Dual,N,DT<:Dual{T,V,N},F<:Tuple} = DT(seed_symm_(x.value,outer_f(f),inner_f(f)),x.partials)
seed_symm_(x::DT,f::F,::typeof(pvalue)) where {T,V<:Dual,N,DT<:Dual{T,V,N},F<:Union{Tuple,<:Function}} = DT(seed_symm_(x.value,outer_f(f)),x.partials)
seed_symm_(x::DT) where {T2,V2,N2,T,V<:Dual{T2,V2,N2},N,DT<:Dual{T,V,N}} = DT(x.value,Partials{N,V}(ntuple(i->V(x.value.partials[i]),N))) # second order
seed_symm_(x::DT) where {T2,V2<:Dual,N2,T,V<:Dual{T2,V2,N2},N,DT<:Dual{T,V,N}} = DT(x.value,Partials{N,V}(ntuple(i->seed_symm_(V(x.value.partials[i])),N)))

"""
```julia
    is_symm(::Type{<:Dual}) -> Bool
```
Check if the outer two Dual layers have symmetric tags, meaning `value.partials == partials.value`. For `Tag{F1,V1}` and `Tag{F2,V2}`, checks if `F1 === F2` and that the number of partials match (`N2 === N`). Returns `false` for non-nested Duals.
"""
is_symm(::Type{<:Dual}) = false # default to false for non-nested duals
is_symm(::Type{Dual{T,DT,N}}) where {T,N,V,T2,N2,DT<:Dual{T2,V,N2}} = (N2 === N) && is_symm(T,T2)
is_symm(::Type{Tag{F1,V1}}, ::Type{Tag{F2,V2}}) where {F1,V1,F2,V2} = F1 === F2

"""
```julia
    get_correct_type(::Type{DT}, f) where {DT<:Dual} -> Type
```
Determine the Dual type at the nesting level corresponding to the composed function chain `f`. Since the main IFT logic applies chains of `extract_partials_ ∘ pvalue ∘ ...`, this function mirrors that by applying `pvalue` to the type once per function in the chain, returning the Dual type at the level where symmetry checks should be performed.
"""
get_correct_type(::Type{DT}, ::typeof(identity)) where {DT<:Dual} = DT
get_correct_type(::Type{DT}, f::F) where {DT<:Dual,F<:Function} = pvalue(DT)
get_correct_type(::Type{DT}, f::F) where {DT<:Dual,F<:Tuple} = get_correct_type(pvalue(DT),outer_f(f))

order_greater_than_one(::Type{DT}) where {T,N,DT<:Dual{T,<:Dual,N}} = true
order_greater_than_one(::Type{DT}) where {DT<:Dual} = false
"""
```julia
    seed_mixed(x::Union{DT, AbstractVector{DT}, IFTStruct}, f::Function) where {DT<:Dual}
```
Check for and apply symmetric seeding of mixed-tag duals. Walks the nested Dual layers using `is_symm` to find consecutive layers with matching tags (e.g. `Dual{T1,Dual{T2,Dual{T3,V,N3},N2},N1}` where `T1` and `T2` share a tag). Symmetric layers are seeded by copying `value.partials` into `partials.value`, avoiding redundant IFT solves for those levels.

Returns `(x, new_f, seeded::Bool, needs_solve::Bool)` where `seeded` indicates if any symmetric seeding was applied and `needs_solve` indicates if further `partials.value` solves are still needed at deeper levels.
"""
function seed_mixed(x::Union{DT,AbstractVector{DT}}, f::F) where {DT<:Dual, F<:Function} # entry point
    ff = unwrap_function(f)
    V = get_correct_type(DT, ff)
    counter = 0
    new_f = identity
    cond_ = is_symm(V)
    while cond_
        counter += 1
        V = pvalue(V)
        (cond_ = is_symm(V)) || break #break before assigning new_f
        new_f = extract_partials_ ∘ new_f
    end
    counter === 0 && return x, f, false, false # no symmetry at this level
    return seed_mixed(x, ff, counter), new_f, true, order_greater_than_one(V) # if final level of symmetry is not a single dual, we still need to solve partials.value field again.
end
function seed_mixed(y::IFTStruct, f::F) where {F}
    _,new_f,cond_,needs_solve = seed_mixed(y.y, f)
    return y, new_f, cond_, needs_solve
end
seed_mixed(x::DT, f::F, counter) where {DT<:Dual,F} = seed_mixed_(x, f, counter)
function seed_mixed(x::AbstractVector{DT}, f::F, counter) where {DT<:Dual,F}
    for i in eachindex(x)
        x[i] = seed_mixed_(x[i], f, counter)
    end
    return x
end

seed_mixed_(x::DT, ::typeof(identity), counter) where {T,V,N,DT<:Dual{T,V,N}} = seed_mixed_(x, counter)
seed_mixed_(x::DT, ::typeof(pvalue), counter) where {T,V<:Dual,N,DT<:Dual{T,V,N}} = DT(seed_mixed_(x.value,counter), x.partials) # mixed seed dual
seed_mixed_(x::DT, ::typeof(extract_partials_), counter) where {T,V<:Dual,N,DT<:Dual{T,V,N}} = DT(x.value, Partials{N,V}(ntuple(i->seed_mixed_(x.partials[i], counter),N)))
function seed_mixed_(x::DT, f_unwrap::Tuple, counter) where {T,N,V,DT<:Dual{T,V,N}}
    fi = inner_f(f_unwrap)
    fo = outer_f(f_unwrap)
    if fi === pvalue
        return DT(seed_mixed_(x.value, fo, counter), x.partials)     
    elseif fi === extract_partials_
        return DT(x.value, Partials{N,V}(ntuple(i->seed_mixed_(x.partials[i], fo, counter),N)))
    end
    throw(ArgumentError("Unsupported function: $fi")) # should never get here, but just pre-empt 
end
function seed_mixed_(x::DT,counter) where {T,V,N,DT<:Dual{T,V,N}}
    counter == 1 && return DT(x.value, Partials{N,V}(ntuple(i->V(x.value.partials[i]),N)))
    return DT(x.value,Partials{N,V}(ntuple(i->seed_mixed_(V(x.value.partials[i]), counter-1),N)))
end
