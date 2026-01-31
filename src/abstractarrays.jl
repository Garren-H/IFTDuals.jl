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
pvalue(x::V) where {T<:Dual,N,V<:AbstractArray{T,N}} = PValueArray(x)
function pvalue(x::V) where {T,N,V<:AbstractArray{T,N}} # handle arrays of mixed or non-Dual types
    !check_eltypes(promote_my_type(x)) && return x # no Duals, return original array
    TT = pvalue(T)
    return PValueArray{TT,N,V}(x)
end

# Wrapper for Arrays of Duals, to extract nested primal values without allocating new arrays.
struct NestedPValueArray{T,N,V} <: AbstractArray{T,N}
   vec::V
end
function NestedPValueArray(vec::V) where {T,N,V<:AbstractArray{T,N}}
    TT = nested_pvalue(T)
    return NestedPValueArray{TT,N,V}(vec)
end
Base.size(A::NestedPValueArray) = size(A.vec)
Base.getindex(x::NestedPValueArray,I...) = nested_pvalue(getindex(x.vec,I...))
nested_pvalue(x::V) where {T<:Dual,N,V<:AbstractArray{T,N}} = NestedPValueArray(x)
function nested_pvalue(x::V) where {T,N,V<:AbstractArray{T,N}} # handle arrays of mixed or non-Dual types
    !check_eltypes(promote_my_type(x)) && return x # no Duals, return original array
    TT = nested_pvalue(T)
    return NestedPValueArray{TT,N,V}(x)
end

# Wrapper for Arrays to promote from one Dual to another without allocating; assuming symmetry of partials.
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
seed_nested_dual(y::Y, DT::Type{Dual{T,V,N}}; ad_type::Symbol=:symmetric) where {Y<:AbstractVecOrMat,T,V,N} = SeedDualArray(y, DT; ad_type=ad_type)

# Wrapper type to extract partials fields from Dual numbers without allocating new arrays. Acts as AbstractVecOrMat{T}.
struct PartialsArray{T,N,V} <: AbstractArray{T,N} # <: AbstractVecOrMat{T}
   vec::V
end
function PartialsArray(vec::V) where {TT,VV,NN,T<:Dual{TT,VV,NN},V<:AbstractVector{T}} # AbstractVector{Dual} with NN-partials -> AbstractMatrix    
    return PartialsArray{VV,2,V}(vec)
end
Base.getindex(x::PartialsArray{VV,2,V},i,j) where {TT,VV,NN,T<:Dual{TT,VV,NN},V<:AbstractVector{T}} = extract_partials_(x.vec[i],j) # extract j-th partial from i-th dual
Base.size(x::PartialsArray{VV,2,V}) where {TT,VV,NN,T<:Dual{TT,VV,NN},V<:AbstractVector{T}} = (length(x.vec),NN)

function PartialsArray(vec::V) where {TT,VV,T<:Dual{TT,VV,1},V<:AbstractVector{T}} # AbstractVector{Dual} with 1-partial -> AbstractVector
    return PartialsArray{VV,1,V}(vec)
end
Base.getindex(x::PartialsArray{VV,1,V},i) where {TT,VV,T<:Dual{TT,VV,1},V<:AbstractVector{T}} = extract_partials_(x.vec[i],1) # extract only partial from i-th dual
Base.size(x::PartialsArray{VV,1,V}) where {TT,VV,T<:Dual{TT,VV,1},V<:AbstractVector{T}} = size(x.vec)

function PartialsArray(vec::V) where {TT,VV,NN,V<:Dual{TT,VV,NN}} # Scalar Dual, multiple partials -> AbstractVector
    return PartialsArray{VV,1,V}(vec)
end
Base.getindex(x::PartialsArray{VV,1,V},i) where {TT,VV,NN,V<:Dual{TT,VV,NN}} = extract_partials_(x.vec,i) # extract i-th partial from single dual
Base.size(::PartialsArray{VV,1,V}) where {TT,VV,NN,V<:Dual{TT,VV,NN}} = (NN,)
extract_partials_(x::Dual{T,V,N}) where {T,V,N} = PartialsArray(x) # wrap to extract partials
extract_partials_(x::X) where {X<:AbstractVector{<:Dual}} = PartialsArray(x) # wrap to extract partials
extract_partials_(x::X,idx::ID) where {X<:AbstractVector{<:Dual},T2<:Union{Int,CartesianIndex{1}},ID<:ScalarOrAbstractVec{T2}} = @view PartialsArray(x)[:,idx] # wrap to extract partials
