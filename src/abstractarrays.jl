# Wrapper for Arrays of Duals, to extract primal values without allocating new arrays.
"""
```julia
    PValueArray{T,N,V} <: AbstractArray{T,N}
```
Wrapper for Array of Duals to extract the value field without allocating new arrays.
"""
struct PValueArray{T,N,V} <: AbstractArray{T,N}
   vec::V
end
function PValueArray(vec::V) where {T,N,V<:AbstractArray{T,N}}
    TT = pvalue(T)
    return PValueArray{TT,N,V}(vec)
end
Base.size(A::PValueArray) = size(A.vec)
Base.getindex(x::PValueArray,I...) = pvalue(getindex(x.vec,I...))

# Wrapper for Arrays of Duals, to extract nested primal values without allocating new arrays.
"""
```julia
    NestedPValueArray{T,N,V} <: AbstractArray{T,N}
```
Wrapper for Array of Duals to (recursively) extract the primal value without allocating new arrays.
"""
struct NestedPValueArray{T,N,V} <: AbstractArray{T,N}
   vec::V
end
function NestedPValueArray(vec::V) where {T,N,V<:AbstractArray{T,N}}
    TT = nested_pvalue(T)
    return NestedPValueArray{TT,N,V}(vec)
end
Base.size(A::NestedPValueArray) = size(A.vec)
Base.getindex(x::NestedPValueArray,I...) = nested_pvalue(getindex(x.vec,I...))

# Wrapper for Arrays to promote from one Dual to another without allocating; assuming symmetry of partials.
"""
```julia
    SeedDualArray{T,N,V,S<:Symbol} <: AbstractArray{T,N}
```
Wrapper for Array to promote from one Dual type to another without allocating new arrays.
"""
struct SeedDualArray{T,N,V,S<:Symbol} <: AbstractArray{T,N}
    vec::V
    ad_type::S # symbol :mixed or :symmetric indicating the type of dual promotion
    function SeedDualArray{DT,N,VV,Symbol}(vec::VV, ad_type::Symbol) where {DT,N,V,VV<:AbstractArray{V,N}}
        @assert in(ad_type, (:mixed, :symmetric)) "ad_type must be either :mixed or :symmetric"
        return new{DT,N,VV,Symbol}(vec, ad_type)
    end
end
function SeedDualArray(vec::VV, DT::Type{<:Dual}; ad_type::Symbol=:symmetric) where {N,V,VV<:AbstractArray{V,N}}
    return SeedDualArray{DT,N,VV,Symbol}(vec, ad_type)
end
Base.size(A::SeedDualArray) = size(A.vec)
Base.getindex(x::SeedDualArray{DT,N,V,S},I...) where {DT,N,V,S} = seed_nested_dual(getindex(x.vec,I...), DT; ad_type=x.ad_type)

# Wrapper type to extract partials fields from Dual numbers without allocating new arrays. Acts as AbstractVecOrMat{T}.
"""
```julia
    PartialsArray{T,N,V} <: AbstractArray{T,N}
```
Wrapper for Duals (Scalar with multiple partials) or AbstractVector of Duals (one or multiple partials) to extract the partials fields as 
an AbstractArray without allocating new arrays. For the Scalar case, we return a AbstractVector, where each entry corresponds to a partial.
For a Vector of Duals with a single partial we return an AbstractVector. For a Vector of Duals with multiple partials we return an AbstractMatrix, where each row corresponds to a Dual and each column to a partial.
"""
struct PartialsArray{T,N,V} <: AbstractArray{T,N} # <: AbstractVecOrMat{T}
   vec::V
end
function PartialsArray(vec::V) where {TT,VV,NN,T<:Dual{TT,VV,NN},N,V<:AbstractArray{T,N}}  
    NV = N
    NN == 1 || (NV += 1) # if multiple partials, add one dimension to accommodate them in the output array
    return PartialsArray{VV,NV,V}(vec)
end
PartialsArray(vec::V) where {TT,VV,NN,V<:Dual{TT,VV,NN}} = PartialsArray{VV,1,V}(vec) # for scalar Duals, return a vector of partials
# Base.getindex(x::PartialsArray{VV,3,V},i,j,k) where {TT,VV,NN,T<:Dual{TT,VV,NN},V<:AbstractMatrix{T}} = extract_partials_(x.vec[i,j], k) 
# Base.size(x::PartialsArray{VV,N,V}) where {TT,VV,NN,T<:Dual{TT,VV,NN},N,NV,V<:AbstractArray{T,NV}} = (size(x.vec)..., NN)
#
# Base.getindex(x::PartialsArray{VV,2,V},i,j) where {TT,VV,NN,T<:Dual{TT,VV,NN},V<:AbstractVector{T}} = extract_partials_(x.vec[i],j) # extract j-th partial from i-th dual
# Base.size(x::PartialsArray{VV,2,V}) where {TT,VV,NN,T<:Dual{TT,VV,NN},V<:AbstractVector{T}} = (length(x.vec),NN)
#
# Base.getindex(x::PartialsArray{VV,1,V},i) where {TT,VV,T<:Dual{TT,VV,1},V<:AbstractVector{T}} = extract_partials_(x.vec[i],1) # extract only partial from i-th dual
# Base.size(x::PartialsArray{VV,1,V}) where {TT,VV,T<:Dual{TT,VV,1},V<:AbstractVector{T}} = size(x.vec)

Base.getindex(x::PartialsArray{VV,1,V},i) where {TT,VV,NN,V<:Dual{TT,VV,NN}} = extract_partials_(x.vec,i) # extract i-th partial from single dual
function Base.getindex(x::PartialsArray{VV,N,V},inds::Vararg{Union{Integer, AbstractUnitRange, Colon}, N}) where {N,TT,VV,NN,V<:AbstractArray{Dual{TT,VV,NN}}}
    dual_idx = inds[1:end-1]
    use_view = ~all(Base.Fix2(isa,Integer), dual_idx) # if any indices are not integers, use view to avoid unnecessary allocations
    partial_idx = inds[end]
    dual = use_view ? view(x.vec, dual_idx...) : getindex(x.vec, dual_idx...)
    extract_partials_(dual, partial_idx)
end
function Base.getindex(x::PartialsArray{VV,N,V},inds::Vararg{Union{Integer, AbstractUnitRange, Colon}, N}) where {N,TT,VV,V<:AbstractArray{Dual{TT,VV,1}}}
    use_view = ~all(Base.Fix2(isa,Integer), inds) # if any indices are not integers, use view to avoid unnecessary allocations
    dual = use_view ? view(x.vec, inds...) : getindex(x.vec, inds...)
    extract_partials_(dual) 
end
Base.size(::PartialsArray{VV,1,V}) where {TT,VV,NN,V<:Dual{TT,VV,NN}} = (NN,)
Base.size(x::PartialsArray{VV,N,V}) where {N,TT,VV,V<:AbstractArray{Dual{TT,VV,1}}} = size(x.vec)
Base.size(x::PartialsArray{VV,N,V}) where {N,TT,VV,NN,V<:AbstractArray{Dual{TT,VV,NN}}} = (size(x.vec)...,NN)

# Wrapper for conversion of Arrays{Any},to Arrays of Duals without allocating new arrays.
"""
```julia
    PromoteToDualArray{T,N,V,DT<:Dual} <: AbstractArray{T,N}
```
Wrapper for Array{Any} to convert to Array of Duals without allocating new arrays.
"""
struct PromoteToDualArray{T,N,V,DT<:Dual} <: AbstractArray{T,N}
    vec::V
end
function PromoteToDualArray(vec::VV, DT::Type{<:Dual}) where {N,V,VV<:AbstractArray{V,N}}
    V === DT && return vec # already of desired Dual type
    has_dual(vec) || return vec # elements do not contain duals
    T = V <: Dual ? DT : V # if V is a Dual, the eltype is the promoted DT otherwise maintain eltype, such as Any
    return PromoteToDualArray{T,N,VV,DT}(vec)
end
Base.size(A::PromoteToDualArray) = size(A.vec)
Base.getindex(x::PromoteToDualArray{T,N,V,DT},I...) where {T,N,V,DT} = begin
    el = getindex(x.vec,I...)
    has_dual(el) || return el # element does not contain duals
    return promote_common_dual_type(el, DT)
end
Base.getindex(x::PromoteToDualArray{T,N,V,DT},I...) where {T<:Dual,N,V,DT} = begin # skip check if T is already a Dual
    el = getindex(x.vec,I...)
    return promote_common_dual_type(el, DT)
end
