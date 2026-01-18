# Functions to get eltypes (Numbers) from generic Data structures. 
## Typical scalars
"""
```julia
    NumEltype(x::V) where V
    NumEltype(::Type{V}) where V
```
Extracts the numeric eltype(s) from generic data structures. For structures containing multiple types (e.g. Tuples, Structs, Arrays of structs, Dicts) it returns the eltype of each entry as a Tuple or Array. If the eltype is not a number, we return `Nothing`. This function is used to check the Dual types contained within data structures. For efficiency, you may provide your own methods for custom data structures e.g.

```julia
struct MyStruct{T<:Number}
    a::T # variable which may contain Duals
    b::String # non-numeric variable
end

NumEltype(x::MyStruct{T}) where T<:Number = T
NumEltype(::Type{MyStruct{T}}) where T<:Number = T
```

Important: If the eltypes is `Real` or `Any`, we require `x` and not just the type `V`. This is because `Real` and `Any` are abstract types used for common propomotion, and do not provide concrete information about the actual types contained within the data structure. If you attempt to pass `NumEltype(::Type{Real})` or `NumEltype(::Type{Any})`, a `RealTypeError` or `AnyTypeError` will be thrown, respectively. Instead, provide an instance of the data structure to extract the concrete eltypes.
"""
NumEltype(::V) where V<:Number = eltype(V)
NumEltype(::Type{V}) where V<:Number = V === Real ? throw(RealTypeError()) : eltype(V) 
## Arrays
NumEltype(x::AbstractArray{V}) where V<:Number = V === Real ? Tuple(map(NumEltype, x)) : V # Real eltypes is not convenient to deal with, cannot check Dual types directly. I.e. [Dual, Int64, Float64] isa Vector{Real} which cannot extract Duals. 
NumEltype(::Type{<:AbstractArray{V}}) where V<:Number = V === Real ? Tuple(map(NumEltype, V)) : throw(RealTypeError()) # Cannot get concrete types from Real
NumEltype(x::AbstractArray{V}) where V = V === Any ? Tuple(map(NumEltype, x)) : NumEltype(V) # return all eltypes if Any
NumEltype(::Type{AbstractArray{V}}) where V = V === Any ? throw(AnyTypeError()) : Tuple(NumEltype(V)) # return all eltypes if Any
## Dictionaries
NumEltype(::Dict{K,V}) where {K,V<:Number} = V
NumEltype(x::Dict{K,V}) where {K,V<:Any} = map(NumEltype, values(x)) # return all value eltypes
NumEltype(::Type{Dict{K,V}}) where {K,V<:Number} = V
## Tuples
NumEltype(x::Tuple) = map(NumEltype, x) # tuples may have different datastructures, so we rather return eltypes for all entries
NumEltype(::NTuple{N,V}) where {N,V<:Number} = V
NumEltype(::Type{NTuple{N,V}}) where {N,V<:Number} = V
NumEltype(x::NTuple{N,V}) where {N,V} = NumEltype(first(x))
## Strings, Symbols, Nothing and Missing (all return true for isstructtype hence we need explicit logic)
NumEltype(::String) = Nothing
NumEltype(::Symbol) = Nothing
NumEltype(::Nothing) = Nothing
NumEltype(::Missing) = Nothing
NumEltype(::Type{String}) = Nothing
NumEltype(::Type{Symbol}) = Nothing
NumEltype(::Type{Nothing}) = Nothing
NumEltype(::Type{Missing}) = Nothing
## Structs and generic cases which do not match above rules
NumEltype(x::V) where V = isstructtype(V) ? Tuple(map(n -> NumEltype(getfield(x, n)), fieldnames(V))) : V === Any ? throw(AnyTypeError()) : Nothing # rather getfield here as we have instances of maybe Real or Any
NumEltype(::Type{V}) where V = isstructtype(V) ? Tuple(map(n-> NumEltype(fieldtype(V, n)), fieldnames(V))) : V === Any ? throw(AnyTypeError()) : Nothing

# Functions to extract the value field from Duals in generic Data structures
"""
```julia
    pvalue(x::V) where V
```
Extracts the primal value(s) from generic data structures containing Dual number. For `V<:Number`, it returns ForwardDiff.value(x)
For structures containing multiple types (e.g. Tuples, Structs, Arrays of structs, Dicts) it returns the primal values of each entry as a Tuple or Array. For non-Dual types, it returns the input as is.

It defaults to returning the input as is. You may consider the following example:

```julia
struct MyStruct{T<:Number}
    a::T # variable which may contain Duals
    b::String # non-numeric variable
end

pvalue(x::MyStruct{T}) where T<:Number = MyStruct{ForwardDiff.valtype(T)}(pvalue(x.a), x.b) # construct new struct with primal eltypes
```
"""
pvalue(::Type{V}) where V<:Dual = valtype(V)
function pvalue(::Type{V}) where V #return constructor for structs
    if isstructtype(V)
        construct_type = Base.typename(V).wrapper
        construct_eltypes = V.parameters
        if length(construct_eltypes) > 0
            construct_eltypes = map(pvalue, construct_eltypes) # recursion on eltyoes
            construct_type = construct_type{construct_eltypes...}
        end
        return construct_type
    end
    return V # non-Dual types
end
pvalue(x::V) where V<:Dual = value(x)
pvalue(x::AbstractArray) = map(pvalue, x)
pvalue(x::NTuple{N,T}) where {N,T} = map(pvalue, x)
pvalue(x::Tuple) = map(pvalue, x)
pvalue(x::Dict{K,V}) where {K,V} = Dict(k => pvalue(v) for (k,v) in pairs(x))
function pvalue(x::T) where T # handle structs specifically 
    if isstructtype(T)
        fldnames = fieldnames(T)
        construct_type = Base.typename(T).wrapper
        construct_eltypes = T.parameters
        if length(construct_eltypes) > 0
            construct_eltypes = map(pvalue, construct_eltypes) # recursion on eltyoes
            construct_type = construct_type{construct_eltypes...}
        end
        flds = Any[]
        contains_duals = false
        for n in fldnames
            fld = getfield(x,n)
            if check_eltypes(NumEltype(fld)) # only extract if we have Dual types
                contains_duals = true
                fld = pvalue(fld)
            end
            push!(flds, fld)
        end
        contains_duals && return construct_type(flds...)
    end
    return x # non-Dual types
end

# Functions to extract the inner most primal value from nested Duals in generic Data structures
"""
```julia
    check_eltypes(::Type{V}) where V
    check_eltypes(x::Tuple)
```
Checks if the eltypes contain any Duals. If it does, returns true. This is a helper function intended for internal use.
"""
check_eltypes(::Type{V}) where V = V<:Dual
check_eltypes(x::Tuple) = any(check_eltypes, x)
check_eltypes(x::AbstractArray) = any(map(check_eltypes, x))

"""
```julia
    nested_pvalue_type(::Type{V}) where V
```
Extracts the innermost primal value type(s) from generic data structures containing nested Dual numbers. For `V<:Dual`, it recursively extracts the ForwardDiff.valtype(V) until reaching a non-Dual.
For structures containing multiple types (e.g. Tuples, Structs, Arrays of structs, Dicts) it returns the innermost primal value types of each entry as a Tuple or Array. For non-Dual types, it returns the input type as is.

This function is used to reconstruct struct types with primal eltypes, maintaining eltypes in struct definitions. For generic structure, we have an implementation which attempts to reconstruct the struct type with primal eltypes, but is possible that this may fail. See `nested_pvalue` for more details.
"""
nested_pvalue_type(::Type{Dual{T,V,N}}) where {T,V,N} = valtype(V)
nested_pvalue_type(::Type{<:Dual{T,V,N}}) where {T,V<:Dual,N} = nested_pvalue_type(V)
nested_pvalue_type(x::V) where V<:Number = x
nested_pvalue_type(::Type{T}) where T<:Tuple = Tuple{map(nested_pvalue_type, T.parameters)...}
nested_pvalue_type(::Type{NTuple{N,T}}) where {N,T} = NTuple{N, nested_pvalue_type(T)}
nested_pvalue_type(::Type{Dict{K,V}}) where {K,V} = Dict{K, nested_pvalue_type(V)}
function nested_pvalue_type(::Type{T}) where T 
    if isstructtype(T) # attempt to reconstruct struct type with primal eltypes
        fldnames = fieldnames(T)
        construct_type = Base.typename(T).wrapper
        construct_eltypes = T.parameters
        if length(construct_eltypes) > 0
            construct_eltypes = map(nested_pvalue_type, construct_eltypes) # recursion on eltyoes
            construct_type = construct_type{construct_eltypes...}
        end
        return construct_type
    else
        return valtype(T) # fallback to valtype for non-struct types
    end
end

"""
```julia
    nested_pvalue(x::V) where V
```
Extracts the innermost primal value type(s) from generic data structures containing nested Dual numbers. For `V<:Dual`, it recursively extracts the valtype until reaching a non-Dual type. For structures containing multiple types (e.g. Tuples, Structs, Arrays of structs, Dicts) it returns the innermost primal value types of each entry as a Tuple or Array. For non-Dual types, it returns the input type as is.

For generic structure, we have an implementation which attempts to reconstruct the struct type with primal eltypes, but is possible that this may fail. You may consider providing your own method for custom data structures, using recursive calls to `pvalue` or `ForwardDiff.value` as needed.
"""
function nested_pvalue(x::V) where V
    if V <: Dual
        return nested_pvalue(pvalue(x))
    elseif isstructtype(V)
        return nested_pvalue_structs(x)
    end
    return x
end
nested_pvalue(x::AbstractArray) = map(nested_pvalue, x)
nested_pvalue(x::NTuple{N,T}) where {N,T} = map(nested_pvalue, x)
nested_pvalue(x::Tuple) = map(nested_pvalue, x)
nested_pvalue(x::Dict{K,V}) where {K,V} = Dict(k => nested_pvalue(v) for (k,v) in pairs(x))

function nested_pvalue_structs(x::T) where T # handle structs specifically
    !any(check_eltypes,NumEltype(x)) && return x # base case when no Duals are present
    fldnames = fieldnames(T)
    construct_type = Base.typename(T).wrapper
    construct_eltypes = T.parameters
    if length(construct_eltypes) > 0 # when we have eltypes associated with structures, we get the primal eltypes into the signature for reconstruction
        construct_eltypes = map(nested_pvalue_type, construct_eltypes)
        construct_type = construct_type{construct_eltypes...}
    end
    return construct_type([nested_pvalue(getfield(x, n)) for n in fldnames]...)
end

# Functions to promote from one Dual type to another in generic Data structures
"""
```julia
    promote_common_dual_type(x::V, DT::Type{<:Dual}) where V
```
Promotes Dual numbers in generic data structures to a common Dual type `DT`. For `V<:Dual`, it constructs a new instance of `DT` with the same value and partials as `x`.
"""
promote_common_dual_type(x::Tuple, DT::Type{<:Dual}) = map(xi->promote_common_dual_type(xi,DT), x)
promote_common_dual_type(x::V, DT::Type{<:Dual}) where V<:Dual = DT(x)
promote_common_dual_type(x::AbstractArray{V}, DT::Type{<:Dual}) where V = map(xi->promote_common_dual_type(xi,DT), x)
promote_common_dual_type(x::Dict{K,V}, DT::Type{<:Dual}) where {K,V} = Dict(k => promote_common_dual_type(v,DT) for (k,v) in pairs(x))
function promote_common_dual_type(x::T, DT::Type{<:Dual}) where T
    if isstructtype(T) 
        fldnames = fieldnames(T)
        construct_type = Base.typename(T).wrapper
        flds = Any[]
        contains_duals = false
        for n in fldnames
            fld = getfield(x,n)
            feltypes = NumEltype(fld) # Tuple of eltypes
            lenf = 1
            if feltypes isa Tuple
                feltypes = filter(et -> et <: Dual, feltypes) # only Dual types
                feltypes = unique(feltypes)
                lenf = length(feltypes)
                lenf == 1 && (feltypes = first(feltypes)) # simplify if single type
            end
            if lenf > 2 || (feltypes !== DT && check_eltypes(feltypes)) # only promote if we have multiple Dual types or the type is different than the promotion type
                contains_duals = true
                fld = promote_common_dual_type(fld,DT)
            end
            push!(flds, fld)
        end
        contains_duals && return construct_type(flds...) # only reconstruct if we have Duals
    end
    return x # non-Dual types
end

export pvalue, nested_pvalue, promote_common_dual_type

