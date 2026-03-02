module IFTDualsStaticArraysExt

import IFTDuals: solve_ift, PromoteToDualArray, pvalue, nested_pvalue, update_dual, IFTStruct, solve_partials, init_y, update_dual, check_partials_seed_dims, unwrap_function, update_dual_, seed_symm, seed_symm_, seed_mixed, seed_mixed_
import ForwardDiff: Dual, Partials
import StaticArrays: SVector, MArray, StaticArray, StaticVector, StaticMatrix, similar_type, LU, ldiv! # ldiv! is essentially from LinearAlgebra

# Additional helpers
swaprows!(a::AbstractMatrix,i,j) = Base.swaprows!(a,i,j);
@inline function swaprows!(a::AbstractVector,i,j)
    @inbounds a[i],a[j] = a[j],a[i]
    return a
end
permuterows!(a::AbstractVecOrMat,p::SVector) = permuterows!(a, MArray(p))
@inline function permuterows!(a::AbstractVecOrMat,p::AbstractVector{Int}) # copy of Base._permute! (https://github.com/JuliaLang/julia/blob/9aff288d6de6a729b7c71d51642ee30eb6b752ad/base/combinatorics.jl#L151-L165) but handles vectors as well
    Base.require_one_based_indexing(a, p)
    p .= .-p
    @inbounds for i in 1:length(p)
        p[i] > 0 && continue
        j = i
        in = p[j] = -p[j]
        while p[in] < 0
            swaprows!(a, in, j)
            j = in
            in = p[in] = -p[in]
        end
    end
    a
end
function solve_ift!(BNi::AbstractVecOrMat,neg_A::LU) # BNi is already permuted
    ldiv!(neg_A.L,BNi)
    ldiv!(neg_A.U,BNi)
    return BNi
end

# update_dual.jl overloads
function update_dual(x::SVector{S,DT}, val::AbstractArray{<:Real}, f::F) where {S,DT<:Dual,F<:Function} # in-place assignment
    check_partials_seed_dims(length(x), (), val, Symbol("Input Vector")) 
    ff = unwrap_function(f)
    return similar_type(x)(update_dual_(x[i], val, ff, (i,)) for i in eachindex(x)) # reconstruct SVector with updated Duals
end
function seed_symm(x::SVector{S,DT},f::F) where {S,DT<:Dual,F<:Function}
    ff = unwrap_function(f)
    return similar_type(x)(seed_symm_(x[i], ff) for i in eachindex(x))
end
seed_mixed(x::SVector{S,DT}, f::F, counter) where {S,DT<:Dual,F<:Function} = similar_type(x)(seed_mixed_(x[i], f, counter) for i in eachindex(x))
function seed_mixed(x::IFTStruct{Y,FF}, f::F, counter) where {Y<:SVector,FF,F<:Function} # doesnt really make to makes to have StaticArrays and inplace functions assignments, but we still support anyway
    ff = unwrap_function(f)
    return IFTStruct(seed_mixed(x.y, ff, counter), x.F) # remake struct
end

# derivatives.jl overloads
solve_partials(y::IFTStruct{Y,F}, BNi::B, fB::FB, neg_A, fE::FE=extract_partials_) where {B<:Union{<:Dual,AbstractArray{<:Dual}},FB<:Function,FE<:Function,Y<:SVector,F} = IFTStruct(solve_partials(y.y,BNi,fB,neg_A,fE),y.F) # remake struct; Doesnt really make to makes to have StaticArrays and inplace functions assignments, but we still support anyway

init_y(y::Y, ::Type{DT}) where {Y<:StaticVector,DT<:Dual} = similar_type(y,DT)(y)

function solve_ift(BNi::B, neg_A::LU) where {V<:Real,B<:AbstractMatrix{V}} # vector case, LU from StaticArrays
    # StaticArrays.jl \ implementation uses F.U \ ( F.L \ b[F.p,:] ) when the input is a AbstractMatrix, which is inefficient if b is not a SMatrix (i.e. b[F.p,:] creates a new Matrix, then does 2 triangular solves, allocating new arrays at each step). We instead utilize ldiv!
    b = BNi
    if ismutable(BNi)
        permuterows!(b,neg_A.p) # in-place pivoting, no allocations
        return solve_ift!(b,neg_A)
    end
end

function solve_ift(BNi::B, neg_A::LU) where {S,V<:Real,N,B<:Union{StaticArray{S,V,N},AbstractVector{V}}}
    if ismutable(BNi)
        permuterows!(BNi,neg_A.p)
        return solve_ift!(BNi,neg_A)
    end
    return neg_A \ BNi # when we have a AbstractVector, then BNi[neg_A.p] creates a new SVector, so no additonal allocations occur here, and SVector should be an acceptable conversion, if LU isa StaticArray.LU then lenght(BNi) == size(neg_A, 1) should be efficient according to user input 
end

# abstractarrays.jl overloads
PromoteToDualArray(x::V, DT::Type{<:Dual}) where {S,T<:Dual,N,V<:StaticArray{S,T,N}} = begin
    V === DT && return x # already of desired Dual type
    tmp = PromoteToDualArray{DT,length(N),V,DT}(x)
    M = Base.typename(V).wrapper
    return M{S,DT,N}(tmp) # reconstruct same type of StaticArray with promoted Dual type
end

# utils.jl overloads
pvalue(x::V) where {S,T<:Dual,N,V<:StaticArray{S,T,N}} = pvalue.(x) # handle with broadcasting, ensures that we return the same type of StaticArray
nested_pvalue(x::V) where {S,T<:Dual,N,V<:StaticArray{S,T,N}} = nested_pvalue.(x) # handle with broadcasting, ensures that we return the same type of StaticArray

end
