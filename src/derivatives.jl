# Functions to take a dual as input, and transforms the dual into one of order + 1 and seeds this order + 1 directional derivatives with zeros. We assume symetric derivative structures.
"""
```julia
    seed_nested_dual(y::V,::Type{<:Dual{T,V,N}}) where {T,V,N}
```
Promotes a `Dual` number to one of order + 1, seeding the new directional derivatives with zeros. Or if the input is of concrete type Int,Float64,etc, it simply constructs a Dual number of the target type with zero partials.
"""
function seed_nested_dual(y::Y, DT::Type{Dual{T,V,N}}; ad_type::Symbol=:symmetric) where {Y<:Real,T,V,N}#seed_nested_dual(y::V, DT::Type{Dual{T,V,N}}; ad_type::Symbol=:symmetric) where {T,V,N} # case where value.partials = partials.values
    order_ = order(DT) # order of the target dual
    if order_ == 1 # just promote, in case user passed order 1
        return DT(y)
    end
    if ad_type == :symmetric
        if order_ == 2 # promote from order 1 to order 2
            parts = Partials{N,V}(ntuple(i -> V(y.partials[i]),N))
            return DT(y,parts)
        else # promote recursively
            func = Base.Fix2(seed_nested_dual, V)
            parts = Partials{N,V}(ntuple(i -> func(y.partials[i]), N)) 
            return DT(y,parts)
        end
    elseif ad_type == :mixed
        return DT(y) # in mixed mode, just create new dual with zero partials
    else
        throw(ArgumentError("ad_type must be either :mixed or :symmetric"))
    end
end

# Functions to extract relevant data from Duals
"""
```julia
    extract_partials_(x::Dual{T,V,N}) where {T,N,V}
    extract_partials_(x::Dual{T,V,N},idx::Union{T2,Vector{T2}}) where {T,V,N,T2<:Union{Int,CartesianIndex{1}}}
```
Extracts the `partials` field as a `Tuple` from a `ForwardDiff.Dual`. When an index `idx` is provided, it extracts the `idx`-th partial derivative.
"""

extract_partials_(x::Dual{T,V,N},idx::ID) where {T,V,N,T2<:Union{Int,CartesianIndex{1}},ID<:ScalarOrAbstractVec{T2}} = x.partials[idx]
extract_partials_(x::Dual{T,V,1}) where {T,V} = x.partials[1]
extract_partials_(x::Dual{T,V,1},idx::Union{Int,CartesianIndex{1}}) where {T,V} = idx == 1 ? x.partials[1] : throw(ArgumentError("Index out of bounds for Dual with 1 partial"))

# Functions to actually compute higher order derivatives
"""
```julia
    make_dual(y::V,DT::Type{<:Dual{T,V,N}},PT::Type{<:Partials{N,V}},parts::Union{V,<:AbstractVector{V}}) where {T,V,N}
```
Creates a `Dual` number of type `DT` with value `y` and partial derivatives given by `parts`, which can be a scalar or a vector of partial derivatives. If `y` is a vector, it creates a Vector of `Dual` numbers accordingly.
"""
make_dual(y::V,DT::Type{Dual{T,V,N}},PT::Type{Partials{N,V}},parts::P) where {T,V,N,P<:Union{ScalarOrAbstractVec{V},<:AbstractArray{V,0}}} = DT(y,PT(NTuple{N,V}(parts)))
function make_dual(y::Y,DT::Type{Dual{T,V,N}},PT::Type{Partials{N,V}},parts::P) where {T,V,N,Y<:AbstractVector{V},P<:AbstractVecOrMat{V}}
    out = Vector{DT}(undef, length(y)) # preallocate output
    @inbounds for i in eachindex(y)
        out[i] = make_dual(y[i],DT,PT,view(parts,i,:))
    end
    return out
end
make_dual(y::Y,DT::Type{Dual{T,V,N}},parts::P) where {T,V,N,Y,P} = make_dual(y,DT,Partials{N,V},parts)

"""
```julia
    solve_ift(y::Union{V,<:AbstractVector{V}},BNi::Union{V,AbstractVecOrMat{V}},neg_A::Union{V,<:LU{V,<:AbstractMatrix{V},<:AbstractVector{<:Integer}}},DT::Type{<:Dual{T,V,N}}) where {T,V<:Real,N}
```
Solves the implicit function theorem system for a single directional derivative and constructs `Dual` numbers of type `DT` with the computed partial derivatives. This is part of a recursive process to compute higher-order derivatives.
"""
function solve_ift(::Y,BNi::B,neg_A) where {V<:Real,Y<:AbstractVector,B<:AbstractVecOrMat{V}} # vector case
    if ismutable(BNi)
        ldiv!(neg_A,BNi)
    else
        BNi = ldiv(neg_A,BNi)
    end
    return BNi # construct Dual numbers
end

function solve_ift(::Y,BNi::B,neg_A) where {V<:Real,Y<:Real,B<:AbstractVector{V}} # scalar with multiple partials
    if ismutable(BNi)
        BNi ./= neg_A
    else
        BNi = BNi ./ neg_A
    end
    return BNi # construct Dual numbers
end

function solve_ift(::Y,BNi::V,neg_A) where {V<:Real,Y<:Real} # scalar case with single partial
    BNi /= neg_A # Solve for directional derivatives
    return BNi # construct Dual numbers
end

function store_ift_cache(y::V,BNi::DT,neg_A,PT=Partials{1,V}) where {T,V<:Real,DT<:Dual{T,V,1}} 
    BNi_i = extract_partials_(BNi,1) # get nested Duals
    dy = extract_partials_(y,1) # get nested Duals
    dual_cache = ift_(dy,BNi_i,neg_A)
    return make_dual(y,DT,PT,dual_cache) # construct Dual numbers
end

function store_ift_cache(y::V,BNi::DT,neg_A,PT=Partials{N,V}) where {T,V<:Real,N,DT<:Dual{T,V,N}} # offload logic to be more efficient with storage types
    dual_cache = Vector{V}(undef, N)
    for i in 1:N
        BNi_i = extract_partials_(BNi,i) # get nested Duals
        dy = extract_partials_(y,i) # get nested Duals
        dual_cache[i] = ift_(dy,BNi_i,neg_A)
    end
    return make_dual(y,DT,PT,dual_cache) # construct Dual numbers
end

function store_ift_cache(y::Y,BNi::B,neg_A,PT=Partials{1,V}) where {T,V<:Real,DT<:Dual{T,V,1},Y<:AbstractVector{V},B<:AbstractVector{DT}} # offload logic to be more efficient with storage types
    BNi_i = extract_partials_(BNi,1) # get nested Duals
    dy = extract_partials_(y,1) # get nested Duals
    dual_cache = ift_(dy,BNi_i,neg_A)
    return make_dual(y,DT,PT,dual_cache) # construct Dual numbers
end

function store_ift_cache(y::Y,BNi::B,neg_A,PT=Partials{N,V}) where {T,V<:Real,N,DT<:Dual{T,V,N},Y<:AbstractVector{V},B<:AbstractVector{DT}} # offload logic to be more efficient with storage types
    dual_cache = Matrix{V}(undef, length(y), N)
    for i in 1:N
        BNi_i = extract_partials_(BNi,i) # get nested Duals
        dy = extract_partials_(y,i) # get nested Duals
        dual_cache[:,i] .= ift_(dy,BNi_i,neg_A)
    end
    return make_dual(y,DT,PT,dual_cache) # construct Dual numbers
end

"""
```julia
    ift_(y::Union{V,<:AbstractVector{V}},BNi::Union{Dual{T,V,N},<:AbstractVector{Dual{T,V,N}}},neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}}) where {T,V<:Real,N,V2<:Real}
```
For a given order of differentiation, recusrively computes all directional derivatives using the implicit function theorem and recreates the appropriate `ForwardDiff.Dual` structure.
"""
function ift_(y::Y,BNi::B,neg_A) where {T,V<:Real,N,DT<:Dual{T,V,N},Y<:ScalarOrAbstractVec{V},B<:ScalarOrAbstractVec{DT}} 
    if V <: Dual # recursion
        return store_ift_cache(y,BNi,neg_A)
    end
    dy = solve_ift(y,extract_partials_(BNi),neg_A) # base case, solve for directional derivatives and create Duals
    return make_dual(y,DT,Partials{N,V},dy) # construct Dual numbers
end

"""
```julia
    ift_recursive(y::Union{V,<:AbstractVector{V}},f::Function,args,neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}},der_order::Int) where {V<:Real,V2<:Real}
```
Recursively applies the implicit function theorem to compute higher-order derivatives up to `der_order`. It evaluates the function `f` at the current `y`, solves for the directional derivatives using `ift_`, and promotes `y` to the next order of `Dual` numbers as needed.
"""
function ift_recursive(y::Y,f::F,args,neg_A,der_order::Int) where {V<:Real,Y<:ScalarOrAbstractVec{V},F<:Function}
    if der_order == 1 # inner most call
        BNi = f(y,args) # evaluate function at primal y
        return ift_(y,BNi,neg_A) # first order IFT and return Dual
    else
        y = ift_recursive(y,f,pvalue(args),neg_A,der_order - 1) # recursive make_dual
        y_star = seed_nested_dual(y,get_common_dual_type(args)) # promote to next order with zero inner-most partials
        BNi = f(y_star,args) # evaluate function at y_star
        return ift_(y,BNi,neg_A) # next order IFT and return Dual
    end
end

"""
```julia
    store_ift_cache_mixed(y,yy,BNi,f,args,neg_A,target_DT,fB,curr_order)
```
Functions to recursurvely solve for and store the partials field when using multiple tags.
"""
function store_ift_cache_mixed(dy::dY,y::Y,BNi::B,f::F,args,neg_A,fB::FB,target_DT::Type{<:Dual}) where {T,V,N,dY<:AbstractMatrix,Y<:AbstractVector,DT<:Dual{T,V,N},B<:AbstractMatrix{DT},F<:Function,FB<:Function}
    dy = ift_mixed_(dy,y,pvalue(BNi),f,args,neg_A,target_DT,x->pvalue(fB)) # Initially, dy is the partials.value.value field. This recursion solves for partials.value.partials and combines to get partials.value 
    dy_new = similar(dy,DT) # stores the reconstructed partials field as a matrix of Duals 
    for i in axes(dy,2) # solves partials.partials.value field 
        dy_i = @view dy[:,i] # i-th partials slice with all functions, i.e. partials.value[i]
        BNi_i = extract_partials_(@view BNi[:,i]) # build the i-th partials field
        dyy_i = solve_ift(dy_i,nested_pvalue(BNi_i),neg_A) # solves the partials.partials[i].value fields
        dy_new[:,i] = make_dual(dy_i,DT,seed_nested_dual(dyy_i,V;ad_type=:mixed))# combine partials.value[i] and partials.partials[i].value fields
    end
    y_star = make_dual(y,target_DT,seed_nested_dual(dy_new,valtype(target_DT);ad_type=:mixed)) # reconstruct y_star with zero partials
    BNi = fB(f(y_star,args)) # Evaluate at the new dual and perform
    # Now solve for the partials.partials field
    dual_cache = similar(dy,DT) # stores the reconstructed partials field as a matrix of Duals
    for i in axes(dy,2) # loop through inner partials dimensions
        dyy_i = nested_pvalue(extract_partials_(@view dy_new[:,i])) # get partials.partials[i].value field
        BNi_i = extract_partials_(@view BNi[:,i]) # build the i-th partials field
        dy_parts_i = ift_mixed_(dyy_i,y,BNi_i,f,args,neg_A,target_DT,x->extract_partials_(@view fB[:,i])) # get/solve for partials.partials.partials and combine with partials.partials.value field, so yields partials.partials field. 
        dual_cache[:,i] .= make_dual(view(dy,:,i),DT,dy_parts_i) # combine partials.value and partials.partials fields to get full partials field
    end
    return dual_cache # now contains partials field as a Matrix of Duals
end

function store_ift_cache_mixed(dy::dY,y::Y,BNi::B,f::F,args,neg_A,fB::FB,target_DT::Type{<:Dual}) where {T,V,N,dY<:ScalarOrAbstractVec,Y<:ScalarOrAbstractVec,DT<:Dual{T,V,N},B<:ScalarOrAbstractVec{DT},F<:Function,FB<:Function} # single partial case, with vector y 
    dy = ift_mixed_(dy,y,pvalue(BNi),f,args,neg_A,target_DT,x->pvalue(fB)) # Initially, dy is the partials.value.value field. This recursion solves for partials.value.partials and combines to get partials.value
    BNi_i = nested_pvalue(extract_partials_(BNi))
    dyy_i = solve_ift(dy,BNi_i,neg_A)
    dy_new = make_dual(dy,DT,seed_nested_dual(dyy_i,V;ad_type=:mixed))
    y_star = make_dual(y,target_DT,seed_nested_dual(dy_new,valtype(target_DT);ad_type=:mixed)) # reconstruct y_star with zero partials
    BNi = fB(f(y_star,args)) # Evaluate at the new dual and perform
    # Now solve for the partials.partials field
    dyy_i = nested_pvalue(extract_partials_(dy_new)) # get partials.partials.value field
    BNi_i = extract_partials_(BNi) # build the i-th partials field
    dy_parts_i = ift_mixed_(dyy_i,y,BNi_i,f,args,neg_A,target_DT,x->extract_partials_(fB)) # get/solve for partials.partials.partials and combine with partials.partials.value field, so yields partials.partials field. 
    dual_cache = make_dual(dy, DT,Partials{N,V}, dy_parts_i) # combine partials.value and partials.partials fields to get full partials field
    return dual_cache # now contains partials field as a Vector of Duals
end

    
"""
Solves the partials.partials fields knowing the partials.value as dy
"""
function solve_mixed_partials(dy::dY,BNi::B,neg_A) where {T,V,N,DT<:Dual{T,V,N},dY<:AbstractMatrix{V}, B<:AbstractMatrix{DT}}
    dual_cache = similar(dy,DT) # stores the recnstructed partials field as a matrix of Duals
    for i in axes(dy,2) # loop through inner partials dimensions
        dy_i = @view dy[:,i] # i-th partials slice with all functions
        BNi_i = extract_partials_(@view BNi[:,i]) # build the i-th partials field
        dyy = solve_ift(dy_i,BNi_i,neg_A) # solve for i-th partials field
        dual_cache[:,i] .= make_dual(dy_i,DT,dyy) # reconstruct duals for i-th partials field
    end
    return dual_cache # new dy, may be able to make in-place
end

function solve_mixed_partials(dy::dY,BNi::B,neg_A) where {T,V,N,DT<:Dual{T,V,N},dY<:ScalarOrAbstractVec{V}, B<:ScalarOrAbstractVec{DT}}
    dyy = solve_ift(dy,extract_partials_(BNi),neg_A) # solve for partials field
    return make_dual(dy,DT,dyy) # reconstruct duals for partials field
end

function ift_mixed_(dy::dY,y::Y,BNi::B,f::F,args,neg_A,target_DT::Type{Dual{T,V,N}},fB::FB=extract_partials_) where {T,V,N,VB<:Dual,F<:Function,FB<:Function,dY,Y<:ScalarOrAbstractVec{V},B<:ScalarOrAbstractVecOrMat{VB}} 
    curr_order = order(VB)
    if curr_order > 1 # Recursion
        return store_ift_cache_mixed(dy,y,BNi,f,args,neg_A,fB,target_DT)
    end
    dy = solve_mixed_partials(dy,BNi,neg_A) # compute partials.value field first, accumulated in dy
    return dy
end

function ift_recursive_mixed(y::Y,f::F,args,neg_A,der_order::Int) where {V<:Real,Y<:ScalarOrAbstractVec{V},F<:Function}
    if der_order == 1 # base case
        return ift_recursive(y,f,args,neg_A,der_order)
    end
    y = ift_recursive_mixed(y,f,pvalue(args),neg_A,der_order-1) # get previous values first
    BNi = f(y,args)
    dy = solve_ift(BNi,nested_pvalue(extract_partials_(BNi)),neg_A) # compute the inner partials.value....value (inner most value field of partials first). 
    target_DT = eltype(BNi)
    y_star = make_dual(y,target_DT,seed_nested_dual(dy,valtype(target_DT);ad_type=:mixed)) # promote to next order dual, with inner mos tvalue field solved ,all other fields are zero. 
    BNi = f(y_star,args) # evaluate at the new dual and perform logic to get BNi in the correct format
    dy = ift_mixed_(dy,y,extract_partials_(BNi),f,args,neg_A,target_DT) # compute partials and store as matrix/vector of scalar Duals
    return make_dual(y,target_DT,dy) # reconstruct duals for i-th partials field
end

"""
```julia
    ift(y::Union{V,<:AbstractVector{V}},f::Function,args) where {V<:Real}
```
Function to compute higher-order derivatives using the implicit function theorem and (nested) Dual numbers. 
Input:
- `y`    : primal input solution to the root finnding problem (scalar or vector)
- `f`    : function handle that takes 'y' and 'args' as inputs
- `args` : tuple or data structure containing Dual numbers indicating the differentiation structure

`f(y,args) = 0` is assumed to define the implicit relationship between `y` and values given as `args`.

**Note**: This function currently does not support mixed-mode AD, i.e. differentiating wrt different variables given as nested Duals. As a workaround you may concatenate all variables into a single vector and differentiate jointly.
"""
function ift(y::Y,f::F,args; DT::DTT=nothing, tag_is_mixed::TM=nothing, args_needs_promotion::Bool=true) where {V<:Real,Y<:ScalarOrAbstractVec{V},F<:Function,DTT<:Union{Nothing,Type{<:Real}},TM<:Union{Nothing,Bool}}
    args_primal = nested_pvalue(args)
    return ift(y,f,args,args_primal;DT=DT,tag_is_mixed=tag_is_mixed,args_needs_promotion=args_needs_promotion)
end

function ift(y::Y,f::F,args,args_primal; DT::DTT=nothing, tag_is_mixed::TM=nothing, args_needs_promotion::Bool=true) where {V<:Real,Y<:ScalarOrAbstractVec{V},F<:Function,DTT<:Union{Nothing,Tuple{<:Real}},TM<:Union{Nothing,Bool}}
    # perform some check first
    if isnothing(tag_is_mixed) # check tags and get type
        if isnothing(DT)
            der_order,DT,tag_is_mixed = check_mixed_tags_and_return_order(args)
        else
            der_order = order(DT)
            tag_is_mixed = check_if_mixed_tag(DT)
        end
    else # user provided tag_is_mixed info 
        if isnothing(DT)
            der_order,DT,_tag_is_mixed = check_mixed_tags_and_return_order(args)
            if !tag_is_mixed && _tag_is_mixed # mixed tags found, but user says no mixed tags. We check that npartials are the same, in this case it might be ok. Up to the user to ensure correctness.
                _DT = valtype(DT)
                nparts = npartials(DT)
                while _DT <: Dual
                    npartials(_DT) == nparts || throw(ArgumentError("Mixed tags found in args, but tag_is_mixed=false was provided. Cannot proceed."))
                    _DT = valtype(_DT)
                end
            end
        else
            der_order = order(DT)
        end
    end
    # Now diff
    der_order == 0 && return y # No differentiation needed
    der_order == 1 && return ift(y,f,args,args_primal,der_order,tag_is_mixed) # first order case, no promotion needed
    args_ = args_needs_promotion ? promote_common_dual_type(args,DT) : args # promote to common dual type if needed
    return ift(y,f,args_,args_primal,der_order,tag_is_mixed) # higher order case
end

function ift(y::Y,f::F,args,args_primal,der_order::I,tag_is_mixed::Bool) where {V<:Real,Y<:ScalarOrAbstractVec{V},I<:Int,F<:Function}
    if Y <: AbstractVector
        neg_A = jacobian(f,AFD,y,Constant(args_primal))::AbstractMatrix{V}
        checksquare(neg_A) # Ensure square matrix
        neg_A .*= -one(V)
        neg_A = lu(neg_A) # LU factorization for later solves
    else
        neg_A = -derivative(f,AFD,y,Constant(args_primal))
    end
    if tag_is_mixed
        return ift_recursive_mixed(y,f,args,neg_A,der_order)
    else
        return ift_recursive(y,f,args,neg_A,der_order)
    end
end

export ift
