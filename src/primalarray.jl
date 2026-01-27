# Wrapper for Arrays of Duals, to extract primal values without allocating new arrays.
struct PValueArray{T,N,V} <: AbstractArray{T,N}
   vec::V
end
function PValueArray(vec::V) where {T,N,V<:AbstractArray{T,N}}
    TT = pvalue(T)
    return PValueArray{TT,N,V}(vec)
end
Base.size(A::PValueArray) = size(A.vec)
Base.getindex(x::PValueArray,I...) = pvalue(getindex(x.vec,I...))
pvalue(::Type{PValueArray{T,N,V}}) where {T,N,V} = pvalue(T)
pvalue(x::V) where {T<:Dual,N,V<:AbstractArray{T,N}} = PValueArray(x)
promote_my_type(x::PValueArray{T,N,V}) where {T<:Dual,N,V} = T
function pvalue(x::V) where {T,N,V<:AbstractArray{T,N}} # handle arrays of mixed or non-Dual types
    !check_eltypes(promote_my_type(x)) && return x # no Duals, return original array
    TT = pvalue(T)
    return PValueArray{TT,N,V}(x)
end

struct NestedPValueArray{T,N,V} <: AbstractArray{T,N}
   vec::V
end
function NestedPValueArray(vec::V) where {T,N,V<:AbstractArray{T,N}}
    TT = pvalue(T)
    return NestedPValueArray{TT,N,V}(vec)
end
Base.size(A::NestedPValueArray) = size(A.vec)
Base.getindex(x::NestedPValueArray,I...) = nested_pvalue(getindex(x.vec,I...))
nested_pvalue(::Type{NestedPValueArray{T,N,V}}) where {T,N,V} = nested_pvalue(T)
nested_pvalue(x::V) where {T<:Dual,N,V<:AbstractArray{T,N}} = NestedPValueArray(x)
function nested_pvalue(x::V) where {T,N,V<:AbstractArray{T,N}} # handle arrays of mixed or non-Dual types
    !check_eltypes(promote_my_type(x)) && return x # no Duals, return original array
    TT = nested_pvalue(T)
    return NestedPValueArray{TT,N,V}(x)
end

