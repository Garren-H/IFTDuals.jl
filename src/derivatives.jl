# Functions to take a dual as input, and transforms the dual into one of order + 1 and seeds this order + 1 directional derivatives with zeros. We assume symetric derivative structures.
"""
```julia
    promote_dual_order(y::V,::Type{<:Dual{T,V,N}}) where {T,V,N})
```
Promotes a `Dual` number to one of order + 1, seeding the new directional derivatives with zeros. Or if the input is of concrete type Int,Float64,etc, it simply constructs a Dual number of the target type with zero partials.
"""
function promote_dual_order(y::V, DT::Type{<:Dual{T,V,N}}) where {T,V,N}
    order_ = order(DT) # order of the target dual
    if order_ == 1 # just promote, in case user passed order 1
        return DT(y)
    elseif order_ == 2 # promote from order 1 to order 2
        parts = Partials{N,V}(ntuple(i -> V(y.partials[i]),N))
        return DT(y,parts)
    else # promote recursively
        func = Base.Fix2(promote_dual_order, V)
        parts = Partials{N,V}(ntuple(i -> func(y.partials[i]), N)) 
        return DT(y,parts)
    end
end
promote_dual_order(y::AbstractVector{V}, DT::Type{<:Dual{T,V,N}}) where {T,V,N} = map(d -> promote_dual_order(d,DT), y)

# Functions to extract relevant data from Duals
"""
```julia
    extract_values_field_from_partials(parts::Partials{N,V}) where {N,V}
```
Extracts the `Tuple` from a `ForwardDiff.Partials` struct. If `N==1`, it returns a scalar instead of a one-element Tuple.
"""
extract_values_field_from_partials(parts::Partials{N,V}) where {N,V} = parts.values
extract_values_field_from_partials(parts::Partials{1,V}) where {V} = parts.values[1]

"""
```julia
    extract_partials_field_from_dual(x::Dual{T,V,N}) where {T,N,V}
    extract_partials_field_from_dual(x::Dual{T,V,N},idx::Union{T2,Vector{T2}}) where {T,V,N,T2<:Union{Int,CartesianIndex{1}}}
```
Extracts the `partials` field as a `Tuple` from a `ForwardDiff.Dual`. When an index `idx` is provided, it extracts the `idx`-th partial derivative.
"""
extract_partials_field_from_dual(x::Dual{T,V,N}) where {T,V,N} = extract_values_field_from_partials(partials(x))
extract_partials_field_from_dual(x::AbstractVector{<:Dual{T,V,N}}) where {T,V,N} = map(extract_partials_field_from_dual, x)
extract_partials_field_from_dual(x::Dual{T,V,N},idx::Union{T2,Vector{T2}}) where {T,V,N,T2<:Union{Int,CartesianIndex{1}}} = x.partials[idx]
extract_partials_field_from_dual(x::AbstractVector{<:Dual{T,V,N}},idx::Union{T2,Vector{T2}}) where {T,V,N,T2<:Union{Int,CartesianIndex{1}}} = map(xi -> extract_partials_field_from_dual(xi,idx), x)

"""
```julia
    custom_stack(x::Union{V,AbstractArray{V}}) where {V<:Dual}
    custom_stack(x::Union{<:Dual{T,V,N},AbstractArray{<:Dual{T,V,N}}},idx::Union{T2,Vector{T2}}) where {T,V,N,T2<:Union{Int,CartesianIndex{1}}}
```
Stacks the partial derivatives of a Dual (or Vector of Duals) into a Matrix of M x N, where M is the size of the input, N is the number of partials. When an index `idx` is provided, it returns a Vector, the 'idx'-th partial derivative(s) (`idx`-th column of the stacked Matrix but efficiently).
"""
custom_stack(x::Union{V,AbstractArray{V}}) where {V<:Dual} = stack(extract_partials_field_from_dual,x;dims=1)
function custom_stack(x::Union{V,<:AbstractArray{V}},idx::Union{T2,Vector{T2}}) where {V<:Dual,T2<:Union{Int,CartesianIndex{1}}}
    func = Base.Fix2(extract_partials_field_from_dual,idx)
    return stack(func,x;dims=1)
end
custom_stack(x::Dual{T,V,1}) where {T,V} = extract_partials_field_from_dual(x) # extract single partial

# Functions to actually compute higher order derivatives
"""
```julia
    create_partials_duals(y::V,DT::Type{<:Dual{T,V,N}},PT::Type{<:Partials{N,V}},parts::Union{V,<:AbstractVector{V}}) where {T,V,N}
```
Creates a `Dual` number of type `DT` with value `y` and partial derivatives given by `parts`, which can be a scalar or a vector of partial derivatives. If `y` is a vector, it creates a Vector of `Dual` numbers accordingly.
"""
create_partials_duals(y::V,DT::Type{<:Dual{T,V,N}},PT::Type{<:Partials{N,V}},parts::Union{V,<:AbstractVector{V}}) where {T,V,N} = DT(y,PT(Tuple(parts)));
create_partials_duals(y::AbstractVector{V},DT::Type{<:Dual{T,V,N}},PT::Type{<:Partials{N,V}},parts::AbstractVecOrMat{V}) where {T,V,N} = [DT(y[i], PT(Tuple(row))) for (i,row) in enumerate(eachrow(parts))];

"""
```julia
    solve_ift(y::Union{V,<:AbstractVector{V}},BNi::Union{V,AbstractVecOrMat{V}},neg_A::Union{V,<:LU{V,<:AbstractMatrix{V},<:AbstractVector{<:Integer}}},DT::Type{<:Dual{T,V,N}}) where {T,V<:Real,N}
```
Solves the implicit function theorem system for a single directional derivative and constructs `Dual` numbers of type `DT` with the computed partial derivatives. This is part of a recursive process to compute higher-order derivatives.
"""
function solve_ift(y::Union{V,<:AbstractVector{V}},BNi::Union{V,AbstractVecOrMat{V}},neg_A::Union{V,<:LU{V,<:AbstractMatrix{V},<:AbstractVector{<:Integer}}},DT::Type{<:Dual{T,V,N}}) where {T,V<:Real,N} # case for a single directional derivative
    #assert ForwardDiff.npartials(DT) == 1 "For this function call, the Dual type must have only one directional derivative."
    parts = neg_A \ BNi # Solve for directional derivatives
    PT = Partials{N,V}
    return create_partials_duals(y,DT,PT,parts) # construct Dual numbers
end

"""
```julia
    ift_(y::Union{V,<:AbstractVector{V}},BNi::Union{Dual{T,V,N},<:AbstractVector{Dual{T,V,N}}},neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}}) where {T,V<:Real,N,V2<:Real}
```
For a given order of differentiation, recusrively computes all directional derivatives using the implicit function theorem and recreates the appropriate `ForwardDiff.Dual` structure.
"""
function ift_(y::Union{V,<:AbstractVector{V}},BNi::Union{Dual{T,V,N},<:AbstractVector{Dual{T,V,N}}},neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}}) where {T,V<:Real,N,V2<:Real} # case for a single directional derivative
    if V <: Dual # recursion
        DT = Dual{T,V,N}
        PT = Partials{N,V}
        vect_duals = N == 1 ? Vector{V}(undef, length(y)) : Matrix{V}(undef, length(y), N) # Storage for the extracted directional derivatives of order N-1
        for i in 1:N
            BNi_i = extract_partials_field_from_dual(BNi,i) # get nested Duals
            dy = extract_partials_field_from_dual(y,i) # get nested Duals
            vect_duals[:,i] = ift_(dy,BNi_i,neg_A) # recursive call, creates nested Dual partials
        end        
        return create_partials_duals(y,DT,PT,vect_duals) # construct Dual numbers
    end
    return solve_ift(y,custom_stack(BNi),neg_A,Dual{T,V,N}) # base case, solve for directional derivatives and create Duals
end

"""
```julia
    ift_recursive(y::Union{V,<:AbstractVector{V}},f::Function,tups,neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}},der_order::Int) where {V<:Real,V2<:Real}
```
Recursively applies the implicit function theorem to compute higher-order derivatives up to `der_order`. It evaluates the function `f` at the current `y`, solves for the directional derivatives using `ift_`, and promotes `y` to the next order of `Dual` numbers as needed.
"""
function ift_recursive(y::Union{V,<:AbstractVector{V}},f::Function,tups,neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}},der_order::Int) where {V<:Real,V2<:Real}
    if der_order == 1 # inner most call
        BNi = f(y,tups) # evaluate function at primal y
        return ift_(y,BNi,neg_A) # first order IFT and return Dual
    else
        y = ift_recursive(y,f,pvalue(tups),neg_A,der_order - 1) # recursive create_partials_duals
        y_star = promote_dual_order(y,get_common_dual_type(tups)) # promote to next order with zero inner-most partials
        BNi = f(y_star,tups) # evaluate function at y_star
        return ift_(y,BNi,neg_A) # next order IFT and return Dual
    end
end

"""
```julia
    ift(y::Union{V,<:AbstractVector{V}},f::Function,tups) where {V<:Real}
```
Function to compute higher-order derivatives using the implicit function theorem and (nested) Dual numbers. 
Input:
    `y`    : primal input solution to the root finnding problem (scalar or vector)
    `f`    : function handle that takes 'y' and 'tups' as inputs
    `tups` : tuple or data structure containing Dual numbers indicating the differentiation structure

`f(y,tups) = 0` is assumed to define the implicit relationship between `y` and values given as `tups`.

**Note**: This function currently does not support mixed-mode AD, i.e. differentiating wrt different variables given as nested Duals. As a workaround you may concatenate all variables into a single vector and differentiate jointly.
"""
function ift(y::Union{V,<:AbstractVector{V}},f::Function,tups) where {V<:Real}
    der_order,DT = check_multiple_duals_and_return_order(tups) # check for multiple duals)
    der_order == 0 && return y # No differentiation needed
    # Get primal value to compute Fy
    tups_primal = Constant(nested_pvalue(tups))
    if y isa AbstractVector
        neg_A = -jacobian(f,AFD,y,tups_primal)
        checksquare(neg_A) # Ensure square matrix
        neg_A = lu(neg_A) # LU factorization for later solves
    else
        neg_A = -derivative(f,AFD,y,tups_primal)
    end
    der_order == 1 && return ift_recursive(y,f,tups,neg_A,der_order) # no promotion needed
    return ift_recursive(y,f,promote_common_dual_type(tups,DT),neg_A,der_order) # promote Duals in tups to common Dual type
end

function ift(y::Union{V,<:AbstractVector{V}},f::Function,tups,tups_primal) where {V<:Real}
    der_order,DT = check_multiple_duals_and_return_order(tups) # check for multiple duals)
    der_order == 0 && return y # No differentiation needed
    # Get primal value to compute Fy
    if y isa AbstractVector
        neg_A = -jacobian(f,AFD,y,Constant(tups_primal))
        checksquare(neg_A) # Ensure square matrix
        neg_A = lu(neg_A) # LU factorization for later solves
    else
        neg_A = -derivative(f,AFD,y,Constant(tups_primal))
    end
    der_order == 1 && return ift_recursive(y,f,tups,neg_A,der_order) # no promotion needed
    return ift_recursive(y,f,promote_common_dual_type(tups,DT),neg_A,der_order) # promote Duals in tups to common Dual type
end

export ift


