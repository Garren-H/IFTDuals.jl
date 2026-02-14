# Functions to take a dual as input, and transforms the dual into one of order + 1 and seeds this order + 1 directional derivatives with zeros. We assume symetric derivative structures.
"""
```julia
    seed_nested_dual(y::ScalarOrAbstractVecOrMat{V},::Type{<:Dual{T,V,N}}; ad_type::Symbol=:symmetric) where {T,V,N}
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
            parts = Partials{N,V}(ntuple(i -> V(y.partials[i]), N))
            return DT(y, parts)
        else # promote recursively
            func = Base.Fix2(seed_nested_dual, V)
            parts = Partials{N,V}(ntuple(i -> func(y.partials[i]), N))
            return DT(y, parts)
        end
    elseif ad_type == :mixed
        return convert(DT, y) # in mixed case, just convert -> will zero outer partials
    else
        throw(ArgumentError("ad_type must be either :mixed or :symmetric"))
    end
end
seed_nested_dual(y::Y, DT::Type{Dual{T,V,N}}; ad_type::Symbol=:symmetric) where {Y<:AbstractVecOrMat,T,V,N} = SeedDualArray(y, DT; ad_type=ad_type)

# Functions to extract relevant data from Duals
"""
```julia
    extract_partials_(x::Dual{T,V,N}) where {T,N,V}
    extract_partials_(x::Dual{T,V,N},idx::Union{T2,Vector{T2}}) where {T,V,N,T2<:Union{Int,CartesianIndex{1}}}
```
Extracts the `partials` field as a `Tuple` from a `ForwardDiff.Dual`. When an index `idx` is provided, it extracts the `idx`-th partial derivative.
"""
extract_partials_(x::V,idx::ID) where {V<:Dual,ID<:IDX} = x.partials[idx]
extract_partials_(x::Dual{T,V,1}) where {T,V} = x.partials[1]
extract_partials_(x::Dual{T,V,1}, idx::ID) where {T,V,ID<:IDX} = idx == 1 ? x.partials[1] : throw(ArgumentError("Index out of bounds for Dual with 1 partial"))
extract_partials_(x::V) where {V<:Dual} = PartialsArray(x) # wrap to extract partials
extract_partials_(x::X) where {X<:AbstractArray{<:Dual}} = PartialsArray(x) # wrap to extract partials
extract_partials_(x::V,::Colon) where {V<:Dual} = PartialsArray(x) # wrap to extract partials
extract_partials_(x::X,::Colon) where {X<:AbstractArray{<:Dual}} = PartialsArray(x) # wrap to extract partials
extract_partials_(x::X,::ID) where {T,V,X<:AbstractArray{Dual{T,V,1}},ID<:IDX} = PartialsArray(x)
function extract_partials_(x::X,idx::ID) where {N,X<:AbstractArray{<:Dual,N},ID<:IDX}
    pa = PartialsArray(x)
    idxs = ntuple(_ -> Colon(), N)
    return @view pa[idxs...,idx] # wrap to extract partials
end

# Functions to actually compute higher order derivatives
"""
```julia
    make_dual(y::V,DT::Type{<:Dual{T,V,N}},PT::Type{<:Partials{N,V}},parts::Union{V,<:AbstractVector{V}}) where {T,V,N}
    make_dual(y::AbstractVector{V},DT::Type{Dual{T,V,N}},PT::Type{Partials{N,V}},parts::AbstractVecOrMat{V}) where {T,V,N}
```
Creates a `Dual` number of type `DT` with value `y` and partial derivatives given by `parts`, which can be a scalar or a vector of partial derivatives. If `y` is a vector, it creates a Vector of `Dual` numbers accordingly.
"""
make_dual(y::V, DT::Type{Dual{T,V,N}}, PT::Type{Partials{N,V}}, parts::P) where {T,V,N,P<:Union{ScalarOrAbstractVec{V},<:AbstractArray{V,0}}} = begin
    DT(y, PT(NTuple{N,V}(parts)))
end
function make_dual(y::Y, DT::Type{Dual{T,V,N}}, PT::Type{Partials{N,V}}, parts::P) where {T,V,N,Y<:AbstractVector{V},P<:AbstractVecOrMat{V}}
    out = similar(y, DT) # preallocate output
    @inbounds for i in eachindex(y)
        partsi = P <: AbstractVector ? parts[i] : view(parts, i, :)
        out[i] = make_dual(y[i], DT, PT, partsi)
    end
    return out
end

function make_dual(y::Y, DT::Type{Dual{T,V,N}}, PT::Type{Partials{N,V}}, parts::P) where {T,V,N,Y<:AbstractMatrix{V},P<:AbstractArray{V,3}}
    out = similar(y, DT) # preallocate output
    @inbounds for i in axes(y, 1), j in axes(y, 2)
        out[i, j, :] .= make_dual(y[i, j], DT, PT, view(parts, i, j, :))
    end
    return out
end
make_dual(y::Y, DT::Type{Dual{T,V,N}}, parts::P) where {T,V,N,Y,P} = make_dual(y, DT, Partials{N,V}, parts)

"""
```julia
    solve_ift(BNi::AbstractVecOrMat{V},neg_A::Union{LU,StaticArrays.LU}) where V<:Real
    solve_ift(BNi::ScalarOrAbstractVecOrMat{V},neg_A) where V
```
Solves the implicit function theorem system for the directional derivatives. It solves one of three cases for efficiency. When the
type of `y` is an `AbstractVector`, `neg_A` is an LU factorization hence we solve by left-division. When `y` is a scalar, then `neg_A` 
is a scalar as well and we have two cases, `BNi` being a vector (indicating multiple partials) or a scalar (indicating a single partial).
When `BNi` is a vector and mutable, we perform in-place division for efficiency. When we a scalars we simply divide.
"""
function solve_ift(BNi::B, neg_A::LU) where {V<:Real,B<:AbstractVecOrMat{V}} # vector case, LU from LinearAlgebra
    if ismutable(BNi)
        ldiv!(neg_A, BNi)
    else
        BNi = ldiv(neg_A, BNi)
    end
    return BNi # construct Dual numbers
end

function solve_ift(BNi::B, neg_A) where {V<:Real,B<:AbstractVecOrMat{V}} # scalar with multiple partials
    if ismutable(BNi)
        BNi ./= neg_A
    else
        BNi = BNi ./ neg_A
    end
    return BNi # construct Dual numbers
end

function solve_ift(BNi::V, neg_A) where {V<:Real} # scalar case with single partial
    BNi /= neg_A # Solve for directional derivatives
    return BNi # construct Dual numbers
end

"""
```julia
    store_ift_cache(y::ScalarOrAbstractVector{V},BNi::ScalarOrAbstractVecOrMat{Dual{T,V,N}},neg_A,PT=Partials{N,V}) where {T,V<:Real,N}
```
Store the computed directional derivatives for higher order derivatives. Logic is dispatched based on the input types for efficiency.
Some combination of types needs a preallocated storage for the directional derivatives. For a given order, we recursively solve for
the directional derivatives, stores the result for each partial derivative, reconstructs the `Dual` structure, and returns it.
"""
function store_ift_cache(y::V, BNi::DT, neg_A, PT=Partials{1,V}) where {T,V<:Real,DT<:Dual{T,V,1}}
    BNi_i = extract_partials_(BNi, 1) # get nested Duals
    dy = extract_partials_(y, 1) # get nested Duals
    dual_cache = ift_(dy, BNi_i, neg_A)
    return make_dual(y, DT, PT, dual_cache) # construct Dual numbers
end

function store_ift_cache(y::V, BNi::DT, neg_A, PT=Partials{N,V}) where {T,V<:Real,N,DT<:Dual{T,V,N}} # offload logic to be more efficient with storage types
    dual_cache = PT(ntuple(i -> begin
        BNi_i = extract_partials_(BNi, i) # get nested Duals
        dy = extract_partials_(y, i) # get nested Duals
        ift_(dy, BNi_i, neg_A)
    end, N))
    return DT(y, dual_cache) # construct Dual numbers
end

function store_ift_cache(y::Y, BNi::B, neg_A, PT=Partials{1,V}) where {T,V<:Real,DT<:Dual{T,V,1},Y<:AbstractVector{V},B<:AbstractVector{DT}} # offload logic to be more efficient with storage types
    BNi_i = extract_partials_(BNi, 1) # get nested Duals
    dy = extract_partials_(y, 1) # get nested Duals
    dual_cache = ift_(dy, BNi_i, neg_A)
    return make_dual(y, DT, PT, dual_cache) # construct Dual numbers
end

function store_ift_cache(y::Y, BNi::B, neg_A, PT=Partials{N,V}) where {T,V<:Real,N,DT<:Dual{T,V,N},Y<:AbstractVector{V},B<:AbstractVector{DT}} # offload logic to be more efficient with storage types
    dual_cache = Matrix{V}(undef, length(y), N)
    for i in 1:N
        BNi_i = extract_partials_(BNi, i) # get nested Duals
        dy = extract_partials_(y, i) # get nested Duals
        dual_cache[:, i] .= ift_(dy, BNi_i, neg_A)
    end
    return make_dual(y, DT, PT, dual_cache) # construct Dual numbers
end

"""
```julia
    ift_(y::Union{V,<:AbstractVector{V}},BNi::Union{Dual{T,V,N},<:AbstractVector{Dual{T,V,N}}},neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}}) where {T,V<:Real,N,V2<:Real}
```
For a given order of differentiation, recursively computes all directional derivatives using the implicit function theorem and recreates the appropriate `ForwardDiff.Dual` structure.
"""
function ift_(y::Y, BNi::B, neg_A) where {T,V<:Real,N,DT<:Dual{T,V,N},Y<:ScalarOrAbstractVec{V},B<:ScalarOrAbstractVec{DT}}
    if V <: Dual # recursion
        return store_ift_cache(y, BNi, neg_A)
    end
    dy = solve_ift(extract_partials_(BNi), neg_A) # base case, solve for directional derivatives and create Duals
    return make_dual(y, DT, Partials{N,V}, dy) # construct Dual numbers
end

"""
```julia
    ift_recursive(y::Union{V,<:AbstractVector{V}},f::Function,args,neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}},der_order::Int) where {V<:Real,V2<:Real}
```
Recursively applies the implicit function theorem to compute higher-order derivatives up to `der_order`. It evaluates the function `f` at the current `y`, solves for the directional derivatives using `ift_`, and promotes `y` to the next order of `Dual` numbers as needed.
"""
function ift_recursive(y::Y, f::F, args, neg_A, der_order::Int) where {V<:Real,Y<:ScalarOrAbstractVec{V},F<:Function}
    if der_order == 1 # inner most call
        BNi = f(y, args) # evaluate function at primal y
        return ift_(y, BNi, neg_A) # first order IFT and return Dual
    else
        y = ift_recursive(y, f, pvalue(args), neg_A, der_order - 1) # recursive make_dual
        y_star = seed_nested_dual(y, get_common_dual_type(args)) # promote to next order with zero inner-most partials
        BNi = f(y_star, args) # evaluate function at y_star
        return ift_(y, BNi, neg_A) # next order IFT and return Dual
    end
end

"""
```julia
    store_ift_cache_mixed(dy,y,BNi,f,args,neg_A,target_DT,fB,curr_order)
```
Store the computed directional derivatives of higher order, mixed tag, Dual numbers. The logic here is that we seperate the partials field into 2 Dual number, the partials.value field and the partials.partials field. Both contain a value and partial field. These can essentially be viewed as 2 Duals, where we need to solve for both branches recursively. The input `dy` contains the partials.value.value field, hence we solve for the partials.value.partials field first. Secondly, the partials.partials.value field is solve. To obtain the partials.partials.partials field, `y` (the value field of the target Dual) needs to be combined with the partials.value.value, partials.value.partials and partials.partials.value fields to create a dual `y_star` whose partials.partials.partials field is zeroed. We then extract the correct BNi by evaluating `fB(f(y_star,args))` (evaluating the function at the new Dual and performing the necessary extraction of the partials, given as the function handle `fB`). Finally, we solve for the partials.partials.partials field and combine all fields to reconstruct the full partials field as a Matrix of Duals.

The explanation above is for the case when we have a 3rd order (nested) Dual, however the logic extends to higher orders partials, by bisecting the partials into its value and partials fields respectively and performing recursion on each of these branches. 
"""
function store_ift_cache_mixed end
# scalar or vector y with 1 partial
function store_ift_cache_mixed(dy::dY, dy_new::dYn, y::Y, BNi::B, f::F, args, neg_A, fB::FB, target_DT::Type{<:Dual}) where {T,V,N,dY<:ScalarOrAbstractVec,dYn<:ScalarOrAbstractVec,Y<:ScalarOrAbstractVec,DT<:Dual{T,V,N},B<:ScalarOrAbstractVec{DT},F<:Function,FB<:Function}
    dyy_i = nested_pvalue(extract_partials_(dy_new)) # get partials.partials.value field
    BNi_i = extract_partials_(BNi) # build the i-th partials field 
    dy_parts_i = ift_mixed_(dyy_i, y, BNi_i, f, args, neg_A, target_DT, x -> extract_partials_(fB(x)))
    dual_cache = make_dual(dy, DT, Partials{N,V}, dy_parts_i)
    return dual_cache
end
# scalar y with N partials
function store_ift_cache_mixed(dy::dY, dy_new::dYn, y::Y, BNi::B, f::F, args, neg_A, fB::FB, target_DT::Type{<:Dual}) where {T,V,N,dY<:AbstractVector,dYn<:AbstractVector,Y<:Real,DT<:Dual{T,V,N},B<:AbstractVector{DT},F<:Function,FB<:Function} 
    dual_cache = Matrix{V}(undef,length(dy),N) # stores the reconstructed partials field as a vector of Duals, length(dy) -> number of partials in partials.value, N -> number of partials in partials.partials
    for i in 1:N
        dyy_i = nested_pvalue(extract_partials_(dy_new,i)) # get partials.partials.value field
        BNi_i = extract_partials_(BNi,i) # build the i-th partial
        dual_cache[:,i] .= ift_mixed_(dyy_i, y, BNi_i, f, args, neg_A, target_DT, x -> extract_partials_(fB(x),i)) # get/solve for partials.partials.partials and combine with partials.partials.value field, so yields partials.partials field.
    end
    return make_dual(dy,DT,Partials{N,V},dual_cache)
end
# vector y with N partials
function store_ift_cache_mixed(dy::dY, dy_new::dYn, y::Y, BNi::B, f::F, args, neg_A, fB::FB, target_DT::Type{<:Dual}) where {T,V,N,dY<:AbstractMatrix,dYn<:AbstractMatrix,Y<:AbstractVector,DT<:Dual{T,V,N},B<:AbstractMatrix{DT},F<:Function,FB<:Function} 
    dual_cache = Array{V}(undef,size(dy)...,N) # stores the reconstructed partials field as a matrix of Duals
    for i in 1:N
        dyy_i = nested_pvalue(extract_partials_(dy_new,i)) # either vector or matrix
        BNi_i = extract_partials_(BNi,i) # either vector or matrix
        dual_cache[:, :, i] .= ift_mixed_(dyy_i, y, BNi_i, f, args, neg_A, target_DT, x -> extract_partials_(fB(x),i)) # get/solve for partials.partials.partials and combine with partials.partials.value field, so yields partials.partials field. 
    end
    return make_dual(dy,DT,Partials{N,V},dual_cache)
end

"""
Function to solve for the partials.value field and return the partial dual (zero out partials.partials)
"""
function solve_mixed_partials_value end

# scalar y
function solve_mixed_partials_value(dy::dY, BNi::B, neg_A) where {T,V,N,dY<:ScalarOrAbstractVec,DT<:Dual{T,V,N},B<:ScalarOrAbstractVec{DT}}
    BNi_i = nested_pvalue(extract_partials_(BNi)) # a scalar or vector of size N
    dyy_i = solve_ift(BNi_i, neg_A)
    dy_new = make_dual(dy, DT, seed_nested_dual(dyy_i, V; ad_type=:mixed))
    return dy_new
end
# vector y with 1 partial. Same as above, but args should have the correct structure to extract the right BNi_i
# function solve_mixed_partials_value(dy::dY, BNi::B, neg_A) where {T,V,N,dY<:AbstractVector,DT<:Dual{T,V,N},B<:AbstractVector{DT}} 
#     BNi_i = nested_pvalue(extract_partials_(BNi)) # A Vector of size length(y)
#     dyy_i = solve_ift(BNi_i, neg_A)
#     dy_new = make_dual(dy, DT, seed_nested_dual(dyy_i, V; ad_type=:mixed))
#     return dy_new
# end
# vector y with N partials
function solve_mixed_partials_value(dy::dY, BNi::B, neg_A) where {T,V,N,dY<:AbstractMatrix,DT<:Dual{T,V,N},B<:AbstractMatrix{DT}}
    dy_new = similar(dy, DT)
    for i in axes(dy,2)
        BNi_i = nested_pvalue(extract_partials_(@view BNi[:, i])) # A Matrix of size (length(y),N)
        dyy_i = solve_ift(BNi_i, neg_A)
        dy_new[:, i] .= make_dual(view(dy, :, i), DT, seed_nested_dual(dyy_i, V; ad_type=:mixed))
    end
    return dy_new
end

"""
Solves the partials.partials fields knowing the partials.value as dy and combines to get the full partials field.
"""
function solve_mixed_partials(dy::dY, BNi::B, neg_A) where {T,V,N,DT<:Dual{T,V,N},dY<:AbstractMatrix{V},B<:AbstractMatrix{DT}}
    dual_cache = similar(dy, DT) # stores the reconstructed partials field as a matrix of Duals
    for i in axes(dy, 2) # loop through inner partials dimensions
        dy_i = @view dy[:, i] # i-th partials slice with all functions
        BNi_i = extract_partials_(@view BNi[:, i]) # build the i-th partials field
        dyy = solve_ift(BNi_i, neg_A) # solve for i-th partials field
        dual_cache[:, i] .= make_dual(dy_i, DT, dyy) # reconstruct duals for i-th partials field
    end
    return dual_cache # new dy, may be able to make in-place
end

# vector y with 1 partial
# function solve_mixed_partials(dy::dY, BNi::B, neg_A) where {T,V,N,DT<:Dual{T,V,N},dY<:AbstractVector{V},B<:AbstractVector{DT}}
#     dyy = solve_ift(extract_partials_(BNi), neg_A) # solve for partials field
#     return make_dual(dy, DT, dyy) # reconstruct duals for partials field
# end

# scalar y
function solve_mixed_partials(dy::dY, BNi::B, neg_A) where {T,V,N,DT<:Dual{T,V,N},dY<:ScalarOrAbstractVec{V},B<:ScalarOrAbstractVec{DT}}
    dyy = solve_ift(extract_partials_(BNi), neg_A) # solve for partials field
    return make_dual(dy, DT, dyy) # reconstruct duals for partials field
end

"""
```julia
    ift_mixed_(dy::Union{V,<:AbstractVector{V}},y::Union{V,<:AbstractVector{V}},BNi::Union{Dual{T,V,N},<:AbstractVector{Dual{T,V,N}}},f::Function,args,neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}},target_DT::Type{Dual{T,V,N}},fB::Function=extract_partials_) where {T,V,N,V2<:Real}
```
For a given order of differentiation with mixed tags, recursively computes all directional derivatives using the implicit function theorem and (nested) Dual numbers.
"""
function ift_mixed_(dy::dY, y::Y, BNi::B, f::F, args, neg_A, target_DT::Type{Dual{T,V,N}}, fB::FB=extract_partials_) where {T,V,N,VB<:Dual,F<:Function,FB<:Function,dY,Y<:ScalarOrAbstractVec{V},B<:ScalarOrAbstractVecOrMat{VB}}
    curr_order = order(VB)
    if curr_order > 1 # Recursion
        dy = ift_mixed_(dy, y, pvalue(BNi), f, args, neg_A, target_DT, x -> pvalue(fB(x)))
        dy_new = solve_mixed_partials_value(dy, BNi, neg_A) # returns a scalar, vector or matrix of Duals, which is the partials.partials.value field. A scalar if y isa scalar and N == 1, a vector of size N if y isa scalar and N > 1, a vector of size length(y) if y isa vector and N == 1, and a matrix if y isa vector and N > 1.
        # offload this logic to solve for partials.partials.value field for each partial.
        y_star = make_dual(y, target_DT, seed_nested_dual(dy_new, valtype(target_DT); ad_type=:mixed)) # reconstruct y_star with zero partials
        BNi = fB(f(y_star, args)) # Evaluate at the new dual and perform
        return store_ift_cache_mixed(dy, dy_new, y, BNi, f, args, neg_A, fB, target_DT) # compute the partials.partials.partials field, combine with partials.partials.value field and store in the correct structure as a scalar, vector or matrix of Duals. Offload logic to be more efficient with storage types.
    end
    dy = solve_mixed_partials(dy, BNi, neg_A) # compute partials.partials and combine with partials,value (dy).
    return dy
end

function ift_recursive_mixed(y::Y, f::F, args, neg_A, der_order::Int, DT::Type{<:Dual}) where {V<:Real,Y<:ScalarOrAbstractVec{V},F<:Function}
    if der_order == 1 # base case
        return ift_recursive(y, f, args, neg_A, der_order) 
    end
    y = ift_recursive_mixed(y, f, pvalue(args), neg_A, der_order - 1, pvalue(DT)) # get previous values first
    BNi = f(seed_nested_dual(y, DT; ad_type=:mixed), args) # apparently need to seed with zero partials to avoid some confusion when tag type is similar, i.e. Tag{typeof(f),Float64} gets confused with Tag{typeof(f),Dual{Tag{typeof(f),Float64}}}, resulting in BNi being evaluated incorrectly, the value which should have occupied value.partials, now occupies partials.value
    dy = solve_ift(nested_pvalue(extract_partials_(BNi)), neg_A) # compute the inner partials.value....value (inner most value field of partials first). 
    y_star = make_dual(y, DT, seed_nested_dual(dy, valtype(DT); ad_type=:mixed)) # promote to next order dual, with inner most value field of partials solved ,all other fields are zero. 
    BNi = f(y_star, args) 
    dy = ift_mixed_(dy, y, extract_partials_(BNi), f, args, neg_A, DT) # compute partials and store as matrix/vector of scalar Duals
    return make_dual(y,DT,dy)
end

"""
```julia
    function ift(y::Union{V,<:AbstractArray{V}},f::Function,args; DT::Type{Union{Nothing,<:Dual}}=nothing, tag_is_mixed::Union{Nothing,Bool}=nothing, args_needs_promotion::Bool=true) where {V<:Real}
    function ift(y::Union{V,<:AbstractArray{V}},f::Function,args,args_primal; DT::Type{Union{Nothing,<:Dual}}=nothing, tag_is_mixed::Union{Nothing,Bool}=nothing, args_needs_promotion::Bool=true) where {V<:Real}
```
Function to compute higher-order derivatives using the implicit function theorem (IFT) and (nested) Dual numbers. 
Input:
- `y`    : primal input solution to the root finding problem (scalar or vector)
- `f`    : function handle that takes 'y' and 'args' as inputs. `f(y,args) = 0` is assumed to define the implicit relationship between `y` and values given as `args`.
- `args` : tuple or data structure containing Dual numbers indicating the differentiation structure
- `args_primal` : primal values of `args`
Optional Input:
- `DT`                  : Target Dual type for the output. If not provided, it is inferred from `args`.
- `tag_is_mixed`        : Boolean indicating if mixed tags are present in `args`. If not provided, it is inferred from `args`.
- `args_needs_promotion`: Boolean indicating if `args` need to be promoted to a common Dual type.
Output:
- Returns `y` as a Dual number with the appropriate order and partial derivatives computed using the IFT.
The function works by first determining the order of differentiation and whether mixed tags are present.

!!! warning
    When mixed tags are detected, but `tag_is_mixed=false` is provided, a check is performed to ensure that the number of partials are consistent across the Dual types in `args`. If inconsistencies are found, an error is thrown. No other checks are performed and it is the user's responsibility to ensure symmetry of partial derivatives in this case.

!!! warning
    If `args_needs_promotion=false` is provided, it is the user's responsibility to ensure that all Dual types in `args` are of the same type. If the Dual type differs, unexpected behavior may occur.
"""
function ift(y::Y, f::F, args; DT::DTT=nothing, tag_is_mixed::TM=nothing, args_needs_promotion::Bool=true) where {V<:Real,Y<:ScalarOrAbstractVec{V},F<:Function,DTT<:Union{Nothing,Type{<:Real}},TM<:Union{Nothing,Bool}}
    args_primal = nested_pvalue(args)
    return ift(y, f, args, args_primal; DT=DT, tag_is_mixed=tag_is_mixed, args_needs_promotion=args_needs_promotion)
end

function ift(y::Y, f::F, args, args_primal; DT::DTT=nothing, tag_is_mixed::TM=nothing, args_needs_promotion::Bool=true) where {V<:Real,Y<:ScalarOrAbstractVec{V},F<:Function,DTT<:Union{Nothing,Tuple{<:Real}},TM<:Union{Nothing,Bool}}
    # perform some check first
    V <: Dual && throw(ArgumentError("Input y cannot be a Dual.")) # pre-empt
    if isnothing(tag_is_mixed) # check tags and get type
        if isnothing(DT)
            der_order, DT, tag_is_mixed = check_mixed_tags_and_return_order(args)
        else
            der_order = order(DT)
            tag_is_mixed = check_if_mixed_tag(DT)
        end
    else # user provided tag_is_mixed info 
        if isnothing(DT)
            der_order, DT, _tag_is_mixed = check_mixed_tags_and_return_order(args)
            if !tag_is_mixed && _tag_is_mixed # mixed tags found, but user says no mixed tags. We check that npartials are the same, in this case it might be ok. Up to the user to ensure symmetry of partials.
                _DT = valtype(DT)
                nparts = npartials(DT)
                while _DT <: Dual # only check if der_order > 1
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
    der_order == 1 && return ift(y, f, args, args_primal, der_order, tag_is_mixed, DT) # first order case, no promotion needed
    args_ = args_needs_promotion ? promote_common_dual_type(args, DT) : args # promote to common dual type if needed
    return ift(y, f, args_, args_primal, der_order, tag_is_mixed, DT) # higher order case
end

function ift(y::Y, f::F, args, args_primal, der_order::I, tag_is_mixed::Bool, DT::Type{<:Dual}) where {V<:Real,Y<:ScalarOrAbstractVec{V},I<:Int,F<:Function}
    V <: Dual && throw(ArgumentError("Input y cannot be a Dual.")) # pre-empt
    if Y <: AbstractVector
        neg_A = jacobian(f, AFD, y, Constant(args_primal))::AbstractMatrix{V}
        checksquare(neg_A) # Ensure square matrix
        _1 = -one(V)
        if ismutable(neg_A) 
            neg_A .*= _1
            neg_A = lu!(neg_A)
        else # safely handle cases of StaticArrays
            neg_A = _1 * neg_A
            neg_A = lu(neg_A) # LU factorization for later solves
        end
    else
        neg_A = -derivative(f, AFD, y, Constant(args_primal))
    end
    if tag_is_mixed
        yy = ift_recursive_mixed(y, f, args, neg_A, der_order, DT)
    else
        yy = ift_recursive(y, f, args, neg_A, der_order)
    end
    return yy
end

export ift
