# Functions to extract the value field from Duals in generic Data structures
"""
```julia
    pvalue(x::V) where V
    pvalue(::Type{V}) where V
```
Extracts the primal value(s) from generic data structures containing Dual number. For `V<:Number`, it returns ForwardDiff.value(x)
For structures containing multiple types (e.g. Tuples, Structs, Arrays of structs, Dicts) it returns the primal values of each entry as a Tuple or Array. For non-Dual types, it returns the input as is. When passing a Type, it returns the corresponding primal eltype(s). The Type signature is only used for reconstructing Types in struct definitions. Hence if you provide a custom method for your struct, the Type signature may not be needed (see below)

We do attempt to reconstruct struct types with primal eltypes, but this may fail or not be performant for custom data structures. In such cases, you may consider providing your own method for your struct, using recursive calls to `pvalue` or `ForwardDiff.value` as needed. For example:

```julia
struct MyStruct{T<:Number}
    a::T # variable which may contain Duals
    b::String # non-numeric variable
end

pvalue(x::MyStruct{T}) where T<:Number = MyStruct{ForwardDiff.valtype(T)}(pvalue(x.a), x.b) # construct new struct with primal eltypes
```
"""
pvalue(::Type{V}) where V<:Dual = valtype(V)
pvalue(::Type{T}) where T<:Tuple = Tuple{map(pvalue, T.parameters)...}
pvalue(::Type{NTuple{N,T}}) where {N,T} = NTuple{N, pvalue(T)}
pvalue(::Type{Dict{K,V}}) where {K,V} = Dict{K, pvalue(V)}
pvalue(::Type{T}) where T<:Union{String, Symbol, Nothing, Missing, Function} = T # non-Dual types
@generated function pvalue_structs(::Type{V}) where V
    construct_type = Expr(:curly,Base.typename(V).wrapper)
    construct_eltypes = V.parameters
    for elt in construct_eltypes
       push!(construct_type.args,:(pvalue($elt)))
    end
    return construct_type
end

function pvalue(::Type{V}) where V #return constructor for structs
   if isstructtype(V)
        construct_eltypes = V.parameters
        length(construct_eltypes) > 0 || return V # base case when no Duals are present
        return pvalue_structs(V)
   end
   return V # non-Dual types
end
pvalue(x::V) where V<:Dual = value(x)
#=function pvalue(x::T) where T<:Array=#
#=    out = similar(x)=#
#=    map!(pvalue, out, x)=#
#=    return out=#
pvalue(x::NTuple{N,T}) where {N,T} = map(pvalue, x)
pvalue(x::Tuple) = map(pvalue, x)
pvalue(x::Dict{K,V}) where {K,V} = Dict(k => pvalue(v) for (k,v) in pairs(x))
pvalue(x::T) where T<:Union{String, Symbol, Nothing, Missing, Function} = x # non-Dual types
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

"""
```julia
    nested_pvalue(x::V) where V
```
Extracts the innermost primal value type(s) from generic data structures containing nested Dual numbers. For `V<:Dual`, it recursively extracts the valtype until reaching a non-Dual type. For structures containing multiple types (e.g. Tuples, Structs, Arrays of structs, Dicts) it returns the innermost primal value types of each entry as a Tuple or Array. For non-Dual types, it returns the input type as is.

For generic structure, we have an implementation which attempts to reconstruct the struct type with primal eltypes, but is possible that this may fail. You may consider providing your own method for custom data structures, using recursive calls to `pvalue` or `ForwardDiff.value` as needed.
"""
nested_pvalue(::Type{Dual{T,V,N}}) where {T,V,N} = valtype(V)
nested_pvalue(::Type{Dual{T,V,N}}) where {T,V<:Dual,N} = nested_pvalue(V)
nested_pvalue(::Type{T}) where T<:Tuple = Tuple{map(nested_pvalue, T.parameters)...}
nested_pvalue(::Type{NTuple{N,T}}) where {N,T} = NTuple{N, nested_pvalue(T)}
nested_pvalue(::Type{Dict{K,V}}) where {K,V} = Dict{K, nested_pvalue(V)}
nested_pvalue(::Type{T}) where T<:Union{String, Symbol, Nothing, Missing, Function} = T # non-Dual types

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
#=nested_pvalue(x::Array) = map(nested_pvalue, x)=#
nested_pvalue(x::NTuple{N,T}) where {N,T} = map(nested_pvalue, x)
nested_pvalue(x::Tuple) = map(nested_pvalue, x)
nested_pvalue(x::Dict{K,V}) where {K,V} = Dict(k => nested_pvalue(v) for (k,v) in pairs(x))
nested_pvalue(x::V) where V<:Union{String, Symbol, Nothing, Missing, Function} = x # non-Dual types

# Functions to promote from one Dual type to another in generic Data structures
"""
```julia
    promote_common_dual_type(x::V, DT::Type{<:Dual}) where V
```
Promotes Dual numbers in generic data structures to a common Dual type `DT`. For `V<:Dual`, it constructs a new instance of `DT` with the same value and partials as `x`.
"""
promote_common_dual_type(x::Tuple, DT::Type{<:Dual}) = map(Base.Fix2(promote_common_dual_type,DT), x)
promote_common_dual_type(x::NTuple{N,V}, ::Type{V}) where {N,V<:Dual} = x # already of desired type
promote_common_dual_type(x::V, DT::Type{<:Dual}) where V<:Dual = DT(x)
promote_common_dual_type(x::V, ::Type{V}) where V<:Dual = x # already of desired type
promote_common_dual_type(x::X, DT::Type{<:Dual}) where {V,X<:AbstractArray{V}} = map(Base.Fix2(promote_common_dual_type,DT), x)
promote_common_dual_type(x::X, ::Type{V}) where {V<:Dual,X<:AbstractArray{V}} = x # already of desired type
promote_common_dual_type(x::V, ::Type{<:Dual}) where V<:Union{String, Symbol, Nothing, Missing, Function} = x # non-Dual types
promote_common_dual_type(x::Dict{K,V}, DT::Type{<:Dual}) where {K,V} = Dict(k => promote_common_dual_type(v,DT) for (k,v) in pairs(x))
promote_common_dual_type(x::Dict{K,V}, ::Type{V}) where {K,V<:Dual} = x # already of desired type
promote_common_dual_type(::Type{V1}, ::Type{V2}) where {V1<:Dual,V2<:Dual} = V2 # for Dual types, just return the target Dual

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
Gets the common Dual supertype from the Dual numbers contained within generic data structures. If now Duals present, returns the common numeric supertype. And if no numeric types are present, it should error.
"""
function get_common_dual_type(x)
    common_et = promote_my_type(x) # get common supertype, should be Dual
    common_et <: Number || throw(error("No numeric types found in input structure"))
    return common_et
end

# Promotion functions to get the common numeric supertype from generic Data structures
_reduce(x) = reduce(promote_my_type, x; init=Nothing) # helper function to reduce over collections, essential to not allocate
"""
```julia
    promote_my_type(::Type{T}) where T
    promote_my_type(x::T) where T
```
Get the common numeric supertype (Duals) from generic data structures. For non-numeric types (String, Symbol, Nothing, Missing, Function), it returns Nothing. For custom data structures, it is recommended to provide your own method for `promote_my_type`. 

Example:
```julia
struct MyStruct{T<:Number}
    a::T # variable which may contain Duals
    b::String # non-numeric variable
end

promote_my_type(::MyStruct{T}) where T<:Number = T # similar to eltype
promote_my_type(::Type{MyStruct{T}}) where T<:Number = T # my be needed where tups contains NTuple{N,MyStruct{T}} types
```

Internally we call `promote_my_type(tups)` which obtains the numeric types from each entry in tups and then reduces over them to get the common numeric supertype. **This is hence a combination of `Base.eltype` and `Base.promote_type` but specialized to only consider numeric types**. An attempt is made to extract the numeric type(s) from custom structs, however this may fail or be non-performant. It is hence highly recommended to provide your own method for custom data structures.
"""
promote_my_type(::Type{T}) where T<:Number = T
promote_my_type(::Type{Any}) = throw(AnyTypeError())
promote_my_type(::Type{Real}) = throw(RealTypeError())
promote_my_type(::Type{<:Union{T,Nothing}}) where T<:Number = T
promote_my_type(::Type{T}, ::Type{Nothing}) where T<:Number = T
promote_my_type(::Type{Nothing}, ::Type{T}) where T<:Number = T
promote_my_type(::Type{T1}, ::Type{T2}) where {T1,T2} = promote_my_type(promote_type(promote_my_type(T1),promote_my_type(T2)))
promote_my_type(x::T1, y::T2) where {T1,T2} = promote_my_type(promote_my_type(x), promote_my_type(y))
promote_my_type(::Type{T1}, x::T2) where {T1,T2} = promote_my_type(T1, promote_my_type(x))
promote_my_type(x::Tuple) = _reduce(x)
promote_my_type(::Type{NTuple{N,T}}) where {N,T} = promote_my_type(T)
promote_my_type(::NTuple{N,T}) where {N,T<:Number} = promote_my_type(T)
promote_my_type(x::Dict{K,V}) where {K,V} = _reduce(values(x))
promote_my_type(x::Dict{K,V}) where {K,V<:Real} = V === Real ? _reduce(values(x)) : V
promote_my_type(::Type{Dict{K,V}}) where {K,V} = promote_my_type(V)
promote_my_type(::V) where V<:Union{String,Symbol,Nothing,Missing,Function} = Nothing
promote_my_type(::Type{V}) where V<:Union{String,Symbol,Nothing,Missing,Function} = Nothing
promote_my_type(::Type{T}) where T<:Array = promote_my_type(T.parameters[1])
promote_my_type(x::AbstractArray) = _reduce(x)
function promote_my_type(x::X) where {V<:Number,X<:AbstractArray{V}} 
    if V === Real 
        return _reduce(x)
    else 
        return V
    end
end
function promote_my_type(::X) where {V<:Union{String, Symbol, Nothing, Missing, Function},X<:AbstractArray{V}}
    return Nothing
end

function promote_my_type(x::T) where T
    if T <: Number
        return promote_type(T)
    elseif x === DataType # isstructtype(DataType) = true
        return Nothing
    elseif isstructtype(T)
        return promote_my_type_struct(x)
    else
        throw(error("No method for promote_my_type(::$T). Define a custom method if needed."))
    end
end

@generated function promote_my_type_struct(x::T) where T
    args = Expr[]
    fldnames = fieldnames(T)
    for n in fldnames
        push!(args, :(promote_my_type(getfield(x, $(QuoteNode(n))))))
    end
    return Expr(:call, :_reduce, Expr(:tuple, args...)) # wrap in tuple for instanced of a single or no args
end

export pvalue, nested_pvalue, promote_common_dual_type, promote_my_type

