const noADTypes = Union{String, Symbol, Nothing, Missing, Function}

# Functions to extract the inner most primal value from nested Duals in generic Data structures
"""
```julia
    check_eltypes(::Type{V}) where V
    check_eltypes(x::Tuple)
```
Checks if the eltypes contain any Duals. If it does, returns true. This is a helper function intended for internal use.
"""
check_eltypes(V::Type) = V<:Dual
check_eltypes(x::Tuple) = any(check_eltypes, x)
check_eltypes(x::AbstractArray) = any(check_eltypes, x)
needs_promotion(V1::Type, V2::Type{<:Dual}) = check_eltypes(V1) && (V1 != V2)

# Functions to extract the value field from Duals in generic Data structures
"""
```julia
    pvalue(x::V) where V
    pvalue(::Type{V}) where V
```
Extracts the value field of Duals contained within generic data structures. `pvalue(x::V)` return `ForwardDiff.value(x)`, and
pvalue(::Type{V}) returns `ForwardDiff.valtype(V)`. For structures containing multiple types (e.g. Tuples, Structs, Arrays of 
structs, Dicts) it loops through all fields/entries in the data structure and performs `pvalue` on each entry. For custom structs it 
is recommended to provide your own method for `pvalue` which only applies pvalue to the fields which may contain Duals.

Example:
```julia
struct MyStruct{T1<:Real,T2<:Real}
    a::T1 # variable which may contain Duals
    b::String # non-numeric variable
    c::T2 # variable which may contain Duals
end

pvalue(x::MyStruct{T1,T2}) where {T1<:Real,T2<:Real} = MyStruct{pvalue(T1),pvalue(T2)}(pvalue(x.a), x.b, pvalue(x.c)) # construct new struct with primal eltypes, need not loop through all fields
```

We encourage the usage of `pvalue` to ensure `x.a` and/or `x.c` are correctly handled if they are custom structs themselves.
"""
pvalue(::Type{V}) where V<:Dual = valtype(V)
pvalue(::Type{NTuple{N,T}}) where {N,T} = NTuple{N, pvalue(T)}
pvalue(::Type{Dict{K,V}}) where {K,V} = Dict{K, pvalue(V)}
pvalue(::Type{T}) where T<:noADTypes = T # non-Dual types
@generated function pvalue_structs(::Type{V}) where V
    construct_type = Expr(:curly,Base.typename(V).wrapper)
    construct_eltypes = V.parameters
    for elt in construct_eltypes
       push!(construct_type.args,:(pvalue($elt)))
    end
    return construct_type
end
function pvalue(::Type{V}) where V #return constructor for structs. I.e. for MyStruct{Dual{T,V,N},String} apply pvalue to each field type
   if isstructtype(V)
        construct_eltypes = V.parameters
        length(construct_eltypes) > 0 || return V # base case Struct does not contain any field types
        return pvalue_structs(V)
   end
   return V # non-Dual types
end
pvalue(x::V) where V<:Dual = value(x)
pvalue(x::T) where T<:Tuple = map(pvalue, x)
pvalue(x::Dict{K,V}) where {K,V} = Dict(k => pvalue(v) for (k,v) in pairs(x))
pvalue(x::T) where T<:noADTypes = x # non-Dual types
@generated function pvalue_structs(x::T) where T
    fldnames = fieldnames(T)
    construct_type = Expr(:call, pvalue(T)) # call to reconstruct type with primal eltypes
    for n in fldnames
        push!(construct_type.args, :(pvalue(getfield(x, $(QuoteNode(n))))))
    end
    return construct_type
end
function pvalue(x::T) where T # handle structs specifically 
    if isstructtype(T)
        check_eltypes(promote_my_type(x)) || return x # base case when no Duals are present
        return pvalue_structs(x)
    end
    return x # non-Dual types
end
pvalue(x::V) where {T<:Dual,N,V<:AbstractArray{T,N}} = PValueArray(x)
function pvalue(x::V) where {T,N,V<:AbstractArray{T,N}} # handle arrays of mixed or non-Dual types
    !check_eltypes(promote_my_type(x)) && return x # no Duals, return original array
    TT = pvalue(T)
    return PValueArray{TT,N,V}(x)
end

"""
```julia
    nested_pvalue(::Type{V}) where V
    nested_pvalue(x::V) where V
```
Extracts the innermost primal value or types from generic structures containing Duals. It recursively applies 
`pvalue` until the innermost primal value/type is not a Dual. Similar to [`pvalue'](@ref), for custom structs
`nested_pvalue` is aplied to all fields in the struct. For custom structs it is recommended to provide 
your own method for `nested_pvalue`, similar to [`pvalue`](@ref). 

Example:
```julia
struct MyStruct{T1<:Real,T2<:Real}
    a::T1 # variable which may contain Duals
    b::String # non-numeric variable
    c::T2 # variable which may contain Duals
end

nested_pvalue(x::MyStruct{T1,T2}) where {T1<:Real,T2<:Real} = MyStruct{nested_pvalue(T1),nested_pvalue(T2)}(nested_pvalue(x.a), x.b, nested_pvalue(x.c))
```

We encourage the usage of `nested_pvalue` to ensure `x.a` and/or `x.c` are correctly handled if they are custom structs themselves.
"""
nested_pvalue(::Type{Dual{T,V,N}}) where {T,V,N} = pvalue(V) # base case when V is not Dual
nested_pvalue(::Type{Dual{T,V,N}}) where {T,V<:Dual,N} = nested_pvalue(V)
nested_pvalue(::Type{NTuple{N,T}}) where {N,T} = NTuple{N, nested_pvalue(T)}
nested_pvalue(::Type{Dict{K,V}}) where {K,V} = Dict{K, nested_pvalue(V)}
nested_pvalue(::Type{T}) where T<:noADTypes = T # non-Dual types

@generated function nested_pvalue_structs(::Type{T}) where T
    construct_type = Expr(:curly,Base.typename(T).wrapper)
    construct_eltypes = T.parameters
    for elt in construct_eltypes
       push!(construct_type.args,:(nested_pvalue($elt)))
    end
    return construct_type
end

function nested_pvalue(::Type{T}) where T 
    if isstructtype(T) # attempt to reconstruct struct type with primal eltypes
        construct_eltypes = T.parameters
        length(construct_eltypes) > 0 || return T 
        return nested_pvalue_structs(T)
    end
    return T # fallback to Type for non-struct types
end

@generated function nested_pvalue_structs(x::T) where T
    fldnames = fieldnames(T)
    construct_type = Expr(:call, nested_pvalue(T)) # call to reconstruct type with primal eltypes
    for n in fldnames
        push!(construct_type.args, :(nested_pvalue(getfield(x, $(QuoteNode(n))))))
    end
    return construct_type
end

function nested_pvalue(x::T) where T
    if isstructtype(T) #attempt to reconstruct struct type with primal eltypes
        check_eltypes(promote_my_type(x)) || return x # base case when no Duals are present
        return nested_pvalue_structs(x) 
    end
    return x # non-Dual types
end
nested_pvalue(x::Dual{T,V,N}) where {T,V,N} = pvalue(x)
nested_pvalue(x::Dual{T,V,N}) where {T,V<:Dual,N} = nested_pvalue(pvalue(x))
nested_pvalue(x::T) where T<:Tuple = map(nested_pvalue, x)
nested_pvalue(x::Dict{K,V}) where {K,V} = Dict(k => nested_pvalue(v) for (k,v) in pairs(x))
nested_pvalue(x::V) where V<:noADTypes = x # non-Dual types
nested_pvalue(x::V) where {T<:Dual,N,V<:AbstractArray{T,N}} = NestedPValueArray(x)
function nested_pvalue(x::V) where {T,N,V<:AbstractArray{T,N}} # handle arrays of mixed or non-Dual types
    !check_eltypes(promote_my_type(x)) && return x # no Duals, return original array
    TT = nested_pvalue(T)
    return NestedPValueArray{TT,N,V}(x)
end

# Functions to promote from one Dual type to another in generic Data structures
"""
```julia
    promote_common_dual_type(x::V, DT::Type{<:Dual}) where V
    promote_common_dual_type(::Type{V1}, ::Type{V2}) where {V1,V2}
```
Promotes Dual numbers in generic data structures to a common Dual type `DT` by using `Base.convert`.
Or when called on types, it checks if `V1` can be promoted to `V2` and returns `V2` if possible.
For structs, the current implementation applies `promote_common_dual_type` to all fields in the struct
and attempts to reconstruct the struct using the promoted values. It is hence recommended to provide 
your own method for custom structs which only promotes the relevant fields.

Example:

```julia
struct MyStruct{T1<:Real,T2<:Real}
    a::T1 # variable which may contain Duals
    b::String # non-numeric variable
    c::T2 # variable which may contain Duals
end

promote_common_dual_type_structs(x::MyStruct{T1,T2},DT::Type{<:Dual}) where {T1<:Real,T2<:Real} = begin 
    TT1 = promote_common_dual_type(T1, DT) # should be DT, but internally checks that this conversion is possible
    TT2 = promote_common_dual_type(T2, DT) # should be DT, but internally checks that this conversion is possible
    MyStruct{TT1,TT2}(promote_common_dual_type(x.a,DT), x.b, promote_common_dual_type(x.c,DT))
end
promote_common_dual_type(x::MyStruct{T,T},DT::Type{T}) where T = x # already desired type
```

We encourage the usage of `promote_common_dual_type` to ensure `x.a` and/or `x.c` are correctly handled if they are custom structs themselves.
"""
promote_common_dual_type(x::Tuple, DT::Type{<:Dual}) = map(Base.Fix2(promote_common_dual_type,DT), x)
promote_common_dual_type(x::NTuple{N,V}, ::Type{V}) where {N,V<:Dual} = x # already of desired type
promote_common_dual_type(x::V, DT::Type{<:Dual}) where V<:Dual = convert(DT,x)
promote_common_dual_type(x::V, ::Type{V}) where V<:Dual = x # already of desired type
promote_common_dual_type(x::X, DT::Type{<:Dual}) where {V,X<:AbstractArray{V}} = PromoteToDualArray(x, DT) 
promote_common_dual_type(x::X, ::Type{V}) where {V<:Dual,X<:AbstractArray{V}} = x # already of desired type
promote_common_dual_type(x::V, ::Type{<:Dual}) where V<:noADTypes = x # non-Dual types
promote_common_dual_type(x::Dict{K,V}, DT::Type{<:Dual}) where {K,V} = Dict(k => promote_common_dual_type(v,DT) for (k,v) in pairs(x))
promote_common_dual_type(x::Dict{K,V}, ::Type{V}) where {K,V<:Dual} = x # already of desired type
promote_common_dual_type(::Type{V1}, ::Type{V2}) where {V1<:Dual,V2<:Dual} = promote_type(V1,V2) == V2 ? V2 : throw(ArgumentError("Cannot promote $V1 to $V2")) # return target V2, but make sure we can promote

@generated function promote_common_dual_type_structs(::Type{V1}, ::Type{DT}) where {V1,DT<:Dual}
    construct_type = Expr(:curly,Base.typename(V1).wrapper)
    construct_eltypes = V1.parameters
    for elt in construct_eltypes
       push!(construct_type.args,:(promote_common_dual_type($elt, DT)))
    end
    return construct_type
end

function promote_common_dual_type(::Type{V1}, DT::Type{<:Dual}) where V1
    if isstructtype(V1) #attempt to reconstruct struct type with promoted eltypes
        construct_eltypes = V1.parameters
        length(construct_eltypes) > 0 || return V1 # base case when no Duals are present
        return promote_common_dual_type_structs(V1, DT)
    end
    return V1 # non-Dual and non-struct types
end

@generated function promote_common_dual_type_structs(x::T, ::Type{DT}) where {T,DT<:Dual}
    fldnames = fieldnames(T)
    construct_type = Expr(:call, promote_common_dual_type(T, DT)) # call to reconstruct type with promoted eltypes
    for n in fldnames
        push!(construct_type.args, :(promote_common_dual_type(getfield(x, $(QuoteNode(n))), DT)))
    end
    return construct_type
end

function promote_common_dual_type(x::T, DT::Type{<:Dual}) where T
    if isstructtype(T)
        check_eltypes(promote_my_type(x)) || return x # base case when no Duals are present
        return promote_common_dual_type_structs(x, DT)
    end
    return x # non-Dual types
end

# Functions to get the common Dual? supertype from generic Data structures
"""
```julia
    get_common_dual_type(x)
```
Gets the common Dual supertype from the Dual numbers contained within generic data structures by executing [`promote_my_type`](@ref).
If no Duals are present, returns the common numeric supertype. And if no numeric types are present, it will error.
"""
function get_common_dual_type(x)
    common_et = promote_my_type(x) # get common supertype, should be at least numeric
    common_et <: Real || throw(error("No numeric types found in input structure"))
    return common_et
end

# Promotion functions to get the common numeric supertype from generic Data structures
_reduce(x) = reduce(promote_my_type, x; init=Nothing) # helper function to reduce over collections, essential to not allocate
"""
```julia
    promote_my_type(::Type{T}) where T
    promote_my_type(x::T) where T
```
Get the common numeric supertype (Duals) from generic data structures. For non-numeric types 
(String, Symbol, Nothing, Missing, Function), it returns Nothing. For custom data structures
the current implementation checks all (nested) fields for numeric types, extracts these numeric
types and reduces `Base.promote_type` over all numeric types found.
**This is hence a combination of `Base.eltype` and `Base.promote_type` but specialized to only 
consider numeric types**. It is hence highly recommended to provide your own method for custom 
data structures.

Example:
```julia
struct MyStruct{T1<:Real,T2<:Real}
    a::T1 # variable which may contain Duals
    b::String # non-numeric variable
    c::T2 # variable which may contain Duals
end

promote_my_type(::MyStruct{T1,T2}) where {T1<:Real,T2<:Real} = promote_my_type(T1,T2)
promote_my_type(::Type{MyStruct{T1,T2}}) where {T1<:Real,T2<:Real} = promote_my_type(T1,T2) # needed if promote_my_type is called on Type
```

The above implementation uses `promote_my_type`, but in this specific case one could use `promote_type` directly:

```julia
promote_my_type(::MyStruct{T1,T2}) where {T1<:Real,T2<:Real} = promote_type(T1,T2)
promote_my_type(::Type{MyStruct{T1,T2}}) where {T1<:Real,T2<:Real} = promote_type(T1,T2)
```

We encourage the usage of `promote_my_type` to ensure `T1` and/or `T2` are correctly handled if they are custom structs themselves. If the struct signature does not contain type information, we perform the promotion on the fields instead:

```julia
promote_my_type(x::MyStruct) = promote_my_type(x.a, x.c)
```

**Note**: If `MyStruct` contained a single numeric field (which may contain duals), `MyStruct{T} where T<:Real`, we would 
return `promote_my_type(T)`. 
"""
promote_my_type(::Type{T}) where T<:Real = T
promote_my_type(::Type{Any}) = throw(AnyTypeError())
promote_my_type(::Type{Real}) = throw(RealTypeError())
promote_my_type(::Type{T}, ::Type{Nothing}) where T<:Real = T
promote_my_type(::Type{Nothing}, ::Type{T}) where T<:Real = T
promote_my_type(::Type{T1}, ::Type{T2}) where {T1,T2} = promote_my_type(promote_type(promote_my_type(T1),promote_my_type(T2)))
promote_my_type(x::T1, y::T2) where {T1,T2} = promote_my_type(promote_my_type(x), promote_my_type(y))
promote_my_type(::Type{T1}, x::T2) where {T1,T2} = promote_my_type(T1, promote_my_type(x))
promote_my_type(x::Tuple) = _reduce(x)
promote_my_type(::Type{NTuple{N,T}}) where {N,T} = promote_my_type(T)
promote_my_type(::NTuple{N,T}) where {N,T<:Real} = promote_my_type(T)
promote_my_type(x::Dict{K,V}) where {K,V} = _reduce(values(x))
promote_my_type(x::Dict{K,V}) where {K,V<:Real} = V === Real ? _reduce(values(x)) : V
promote_my_type(::Type{Dict{K,V}}) where {K,V} = promote_my_type(V)
promote_my_type(::V) where V<:noADTypes = Nothing
promote_my_type(::Type{V}) where V<:noADTypes = Nothing
promote_my_type(::X) where {V<:noADTypes,X<:AbstractArray{V}} = Nothing
promote_my_type(::Type{AbstractArray{T}}) where T = promote_my_type(T)
promote_my_type(x::AbstractArray) = _reduce(x) # handles AbstractArray{Any} correctly
function promote_my_type(x::X) where {V<:Real,X<:AbstractArray{V}} 
    if V === Real || typeof(V) === UnionAll # handle non-specific eltypes
        return _reduce(x)
    else 
        return V # eltype if all are same types
    end
end

function promote_my_type(x::T) where T
    if T <: Real
        return promote_type(T)
    elseif x === DataType # isstructtype(DataType) = true. Structs may contain DataTypes as fields
        return Nothing
    elseif isstructtype(T)
        return promote_my_type_struct(x)
    else
        throw(error("No method for promote_my_type(::$T). Define a custom method if needed."))
    end
end

@generated function promote_my_type_struct(x::T) where T
    args = Expr(:tuple)# wrap in tuple for instanced of a single or no args
    fldnames = fieldnames(T)
    for n in fldnames
        push!(args.args, :(promote_my_type(getfield(x, $(QuoteNode(n))))))
    end
    return Expr(:call, :_reduce, args) 
end

export pvalue, nested_pvalue, promote_common_dual_type, promote_my_type

