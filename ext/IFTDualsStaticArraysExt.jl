module IFTDualsStaticArraysExt

using IFTDuals
import IFTDuals: make_dual, solve_ift, PromoteToDualArray
import ForwardDiff: Dual, Partials
import StaticArrays: StaticArray, StaticVector, StaticMatrix, similar_type, LU, ldiv! # ldiv! is essentially from LinearAlgebra

# derivatives.jl overloads
function make_dual(y::Y, DT::Type{Dual{T,V,N}}, PT::Type{Partials{N,V}}, parts::P) where {T,V,N,NY,Y<:StaticVector{NY,V},P<:AbstractVecOrMat{V}} #maintain S/M Array structure if input is S/M Array
    return similar_type(y,DT)(
        begin
            partsi = P <: AbstractVector ? parts[i] : view(parts, i, :)
            make_dual(y[i], DT, PT, partsi)
        end 
    for i in eachindex(y))
end

function solve_ift(::Y, BNi::B, neg_A::LU) where {V<:Real,Y<:AbstractVector,B<:AbstractMatrix{V}} # vector case, LU from StaticArrays
    # StaticArrays.jl \ implementation uses F.U \ ( F.L \ b[F.p,:] ) when the input is a AbstractMatrix, which is inefficient if b is not a SMatrix (i.e. b[F.p,:] creates a new Matrix, then does 2 triangular solves, allocating new arrays at each step). We instead utilize ldiv!
    b = BNi
    if ismutable(BNi)
        @inbounds for (i,p) in enumerate(neg_A.p) # apply pivoting in-place
            for j in axes(BNi,2)
                b[i,j],b[p,j] = BNi[p,j], BNi[i,j]
            end
        end
    else # we have wrapper types, NestedPValueArray or PartialsArray or view thereof. Extracting indices with getindex may not allocate new array
        if !(B <: SubArray) # get a view first, @view(AbstractMatrix,:,:)[neg_A.p.:] allocates a new array
            b = view(b,:,:) 
        end
        b = b[neg_A.p,:] # apply pivoting, returns a new array, should be minimal allocs
    end
    ldiv!(neg_A.L,b)
    ldiv!(neg_A.U,b)
    return b
end

function solve_ift(::Y, BNi::B, neg_A::LU) where {S,V<:Real,N,Y<:AbstractMatrix,B<:Union{StaticArray{S,V,N},AbstractVector{V}}}
    return neg_A \ BNi # when we have a AbstractVector, then BNi[neg_A.p] creates a new SVector, so no additonal allocations occur here. 
end

# abstractarrays.jl overloads
PromoteToDualArray(x::V, DT::Type{<:Dual}) where {S,T<:Dual,N,V<:StaticArray{S,T,N}} = begin
    V === DT && return x # already of desired Dual type
    tmp = PromoteToDualArray{DT,length(N),V,DT}(x)
    M = Base.typename(V).wrapper
    return M{S,DT,N}(tmp) # reconstruct same type of StaticArray with promoted Dual type
end

# utils.jl overloads
pvalue(x::V) where {S,T<:Dual,N,V<:StaticArray{S,T,N}} = pvalue.(x) # handle with broadcastnig, ensures that we return the same type of StaticArray
nested_pvalue(x::V) where {S,T<:Dual,N,V<:StaticArray{S,T,N}} = nested_pvalue.(x) # handle with broadcastnig, ensures that we return the same type of StaticArray
promote_common_dual_type(x::V, DT::Type{<:Dual}) where {S,T<:Dual,N,V<:StaticArray{S,T,N}} = begin
    V === DT && return x # already of desired Dual type
    tmp = PromoteToDualArray{DT,N,V,DT}(x)
    M = Base.typename(V).wrapper
    return M{S,DT,N}(tmp) # reconstruct same type of StaticArray with promoted Dual type
end

end
