# Functions to take a dual as input, and transforms the dual into one of order + 1 and seeds this order + 1 directional derivatives with zeros. We assume symetric derivative structures.
"""
```julia
    seed_nested_dual(y::ScalarOrAbstractVecOrMat{V},::Type{<:Dual{T,V,N}}; ad_type::Symbol=:symmetric) where {T,V,N}
```
Promotes a `Dual` number to one of order + 1, seeding the new directional derivatives with zeros. Or if the input is of concrete type Int,Float64,etc, it simply constructs a Dual number of the target type with zero partials.
"""
function seed_nested_dual(y::Y, DT::Type{Dual{T,V,N}}; ad_type::Symbol=:symmetric) where {Y<:Real,T,V,N}#seed_nested_dual(y::V, DT::Type{Dual{T,V,N}}; ad_type::Symbol=:symmetric) where {T,V,N} # case where value.partials = partials.values
    Y === DT && return y # no seeding
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
seed_nested_dual(y::Y, ::Type{DT}; ad_type::Symbol=:symmetric) where {VY,Y<:AbstractArray{VY},DT<:Dual} = VY === DT ? y : SeedDualArray(y, DT; ad_type=ad_type)

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
    return pa[idxs...,idx]
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
function make_dual(y::Y, DT::Type{Dual{T,V,N}}, PT::Type{Partials{N,V}}, parts::P) where {T,V,N,NY,Y<:AbstractArray{V,NY},P<:AbstractArray{V,NY}} # 1 partial
    @assert (size(y) == size(parts)) "Incompatable dimensions"
    out = similar(y, DT) # preallocate output
    @inbounds for i in eachindex(y)
        out[i] = make_dual(y[i], DT, PT, parts[i])
    end
    return out
end
function make_dual(y::Y, DT::Type{Dual{T,V,N}}, PT::Type{Partials{N,V}}, parts::P) where {T,V,N,NY,NP,Y<:AbstractArray{V,NY},P<:AbstractArray{V,NP}} # multiple partials
    @assert ((NP == (NY + 1)) && (size(y) == size(parts)[1:end-1])) "Incompatable dimensions"
    out = similar(y, DT) # preallocate output
    @inbounds for i in CartesianIndices(y)
        partsi = @view parts[i,:]
        out[i] = make_dual(y[i], DT, PT, partsi)
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
    ift_(y::Union{V,<:AbstractVector{V}},BNi::Union{Dual{T,V,N},<:AbstractVector{Dual{T,V,N}}},neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}}) where {T,V<:Real,N,V2<:Real}
```
For a given order of differentiation, recursively computes all directional derivatives using the implicit function theorem and recreates the appropriate `ForwardDiff.Dual` structure.
"""
function ift_(y::Y, BNi::B, neg_A) where {T,V<:Real,N,DT<:Dual{T,V,N},Y<:Union{V,AbstractArray{V}},B<:Union{DT,AbstractArray{DT}}}
    if V <: Dual # recursion
        return make_dual(y,DT,ift_(extract_partials_(y),extract_partials_(BNi),neg_A))
    end
    return solve_partials(y,BNi,neg_A)
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
Function to solve for the partials.value field and return the partial dual (zero out partials.partials)
"""
function solve_partials_value end
# scalar y
function solve_partials_value(BNi::B, neg_A) where {T,V,N,DT<:Dual{T,V,N},B<:DT}
    BNi_ = nested_pvalue(extract_partials_(BNi)) # a scalar
    dyy = solve_ift(BNi_, neg_A)
    return dyy
end
# generic case
function solve_partials_value(BNi::B, neg_A) where {T,V,N,DT<:Dual{T,V,N},B<:AbstractArray{DT}}
    dyy = collect(nested_pvalue(extract_partials_(BNi))) #materialize to solve in-place
    dyy_ = ndims(dyy) > 2 ? reshape(dyy, (size(dyy,1),:)) : dyy # always solve a matrix/vector
    solve_ift(dyy_, neg_A) # solve for partials.field in-place, updates dyy
    return dyy
end

"""
Solves the partials.partials fields knowing the partials.value as dy and combines to get the full partials field.
"""
function solve_partials end
# scalar y with single partial in all directions
function solve_partials(dy::dY, BNi::B, neg_A) where {T,V,N,DT<:Dual{T,V,N},dY<:V,B<:DT}
    dyy = solve_ift(extract_partials_(BNi), neg_A) # solve for partials field
    return make_dual(dy, DT, dyy) # reconstruct duals for partials field
end
# generic case
function solve_partials(dy::dY, BNi::B, neg_A) where {T,V,N,DT<:Dual{T,V,N},dY<:Union{V,AbstractArray{V}},B<:AbstractArray{DT}}
    dyy = collect(extract_partials_(BNi)) # materialize for in-place solves
    dyy_ = ndims(dyy) > 2 ? reshape(dyy, (size(dyy,1),:)) : dyy # always solve a matrix/vector
    solve_ift(dyy_, neg_A) # solve for partials field in-place, updates dyy
    return make_dual(dy, DT, dyy) # reconstruct duals for partials field
end

"""
Helper function to make duals where appropriate
"""
mixed_make_dual(tups::TT, ::Type{DT}) where {T,V,N,DT<:Dual{T,V,N},TT<:Tuple{Union{V,AbstractArray{V}},Union{V,AbstractArray{V}}}} = make_dual(tups[1],DT,tups[2])
function mixed_make_dual(tups::TT, ::Type{DT}) where {T,V,N,DT<:Dual{T,V,N},TT<:Tuple{Union{V,AbstractArray{V}},<:Tuple}}
    inner = mixed_make_dual(tups[2],V)
    inner isa Tuple && return (tups[1],inner)
    return mixed_make_dual((tups[1],inner),DT)
end
mixed_make_dual(tups,::Type) = tups # no duals to make currently

unpack_tuple_and_solve(dy::dY, y::Y, BNi::B, f::F, args, neg_A, target_DT, fB::FB) where {dY<:Tuple{Union{<:Real,<:AbstractArray},Nothing},Y,B,F<:Function,FB<:Function} = ift_mixed_(dy[1], y, pvalue(BNi), f, args, neg_A, target_DT, pvalue ∘ fB)
unpack_tuple_and_solve(dy::dY, y::Y, BNi::B, f::F, args, neg_A, target_DT, fB::FB) where {dY<:Tuple{Union{<:Real,<:AbstractArray},Union{<:Real,<:AbstractArray}},Y,B,F<:Function,FB<:Function} = mixed_make_dual((dy[1],ift_mixed_(dy[2], y, pvalue(BNi), f, args, neg_A, target_DT, pvalue ∘ fB)), eltype(BNi))
unpack_tuple_and_solve(dy::dY, y::Y, BNi::B, f::F, args, neg_A, target_DT, fB::FB) where {dY<:Tuple{Union{<:Real,<:AbstractArray},<:Tuple},Y,B,F<:Function,FB<:Function} = mixed_make_dual((dy[1],unpack_tuple_and_solve(dy[2], y, BNi, f, args, neg_A, target_DT, fB)), eltype(BNi))
unpack_tuple_and_solve(dy::dY, y::Y, BNi::B, f::F, args, neg_A, target_DT, fB::FB) where {dY<:Union{<:Real,<:AbstractArray},Y,B,F<:Function,FB<:Function} = ift_mixed_(dy, y, pvalue(BNi), f, args, neg_A, target_DT, pvalue ∘ fB)

pack_(dy::dY,dy_pv) where {dY<:Union{<:Real,<:AbstractArray}} = (dy,dy_pv)
pack_(dy::dY,dy_pv) where {dY<:Tuple} = (dy[1],pack_(dy[2], dy_pv))

function mixed_seed_dual(dy::dY,::Type{DT}) where {T,V,N,DT<:Dual{T,V,N},VY,dY<:Tuple{Union{VY,<:AbstractArray{VY}},Union{<:Real,<:AbstractArray}}}
    if V === VY
        return make_dual(dy[1],DT,seed_nested_dual(dy[2],V; ad_type=:mixed))
    end
    DT_ = pvalue(DT)
    V_ = pvalue(DT_)
    while V_ !== VY # demote until we get correct inner type
        DT_ = V_
        V_ = pvalue(DT_)
    end
    return seed_nested_dual(make_dual(dy[1],DT_,seed_nested_dual(dy[2],V_; ad_type=:mixed)), DT; ad_type=:mixed) # firstly get a Dual, then seed dual to zeros
end
function mixed_seed_dual(dy::dY,::Type{DT}) where {T,V,N,DT<:Dual{T,V,N},VY,dY<:Tuple{Union{VY,<:AbstractArray{VY}},<:Tuple}}
    if V === VY
        return make_dual(dy[1],DT,mixed_seed_dual(dy[2],V))
    end
    DT_ = pvalue(DT)
    V_ = pvalue(DT_)
    while V_ !== VY # demote until we get correct inner type
        DT_ = V_
        V_ = pvalue(DT_)
    end
    return seed_nested_dual(make_dual(dy[1],DT_,mixed_seed_dual(dy[2],V_)), DT; ad_type=:mixed) 
end

unpack_tuple_and_solve_partials(dy::dY, BNi, neg_A) where {dY<:Tuple{Union{<:Real,<:AbstractArray},Nothing}} = solve_partials(dy[1], BNi, neg_A)
unpack_tuple_and_solve_partials(dy::dY, BNi, neg_A) where {dY<:Tuple{Union{<:Real,<:AbstractArray},Union{<:Real,<:AbstractArray}}} = (dy[1], solve_partials(dy[2], BNi, neg_A))
unpack_tuple_and_solve_partials(dy::dY, BNi, neg_A) where {dY<:Tuple{Union{<:Real,<:AbstractArray},<:Tuple}} = (dy[1], unpack_tuple_and_solve_partials(dy[2], BNi, neg_A))
unpack_tuple_and_solve_partials(dy::dY, BNi, neg_A) where {dY<:Union{<:Real,<:AbstractArray}} = solve_partials(dy, BNi, neg_A)

mixed_seed_with_val(y::Y,::Type{DT},dy) where {T,V,N,DT<:Dual{T,V,N},Y<:Union{V,<:AbstractArray{V}}} = make_dual(y,DT,mixed_seed_dual(dy,V))
"""
```julia
    ift_mixed_(dy::Union{V,<:AbstractVector{V}},y::Union{V,<:AbstractVector{V}},BNi::Union{Dual{T,V,N},<:AbstractVector{Dual{T,V,N}}},f::Function,args,neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}},target_DT::Type{Dual{T,V,N}},fB::Function=extract_partials_) where {T,V,N,V2<:Real}
```
For a given order of differentiation with mixed tags, recursively computes all directional derivatives using the implicit function theorem and (nested) Dual numbers.
"""
function ift_mixed_(dy::dY, y::Y, BNi::B, f::F, args, neg_A, target_DT::Type{Dual{T,V,N}}, fB::FB=extract_partials_) where {T,V,N,VB<:Dual,F<:Function,FB<:Function,dY,Y<:ScalarOrAbstractVec{V},B<:Union{VB,AbstractArray{VB}}}
    curr_order = order(VB)
    if curr_order > 1 # Recursion
        dy = unpack_tuple_and_solve(dy, y, BNi, f, args, neg_A, target_DT, fB) # returns the partials.value field with all fields solved. 
        dy_pv = solve_partials_value(BNi, neg_A) # returns a scalar, vector or matrix of Duals, which is the partials.partials.value field. A scalar if y isa scalar and N == 1, a vector of size N if y isa scalar and N > 1, a vector of size length(y) if y isa vector and N == 1, and a matrix if y isa vector and N > 1.
        dy = pack_(dy,dy_pv) # repack into tuple
        y_star = mixed_seed_with_val(y,target_DT,dy)# reconstruct y_star with zero partials
        BNi = fB(f(y_star, args)) # Evaluate at the new dual and perform
        dy_new = ift_mixed_(dy, y, extract_partials_(BNi), f, args, neg_A, target_DT, extract_partials_ ∘ fB)
        return mixed_make_dual(dy_new,VB)
    end
    return unpack_tuple_and_solve_partials(dy, BNi, neg_A) # compute partials.partials and combine with partials,value (dy) #dy
end

function ift_recursive_mixed(y::Y, f::F, args, neg_A, der_order::Int, DT::Type{<:Dual}) where {V<:Real,Y<:ScalarOrAbstractVec{V},F<:Function}
    if der_order == 1 # base case
        return ift_recursive(y, f, args, neg_A, der_order) # first order IFT and return Dual
    end
    y = ift_recursive_mixed(y, f, pvalue(args), neg_A, der_order - 1, pvalue(DT)) # get previous values first
    BNi = f(seed_nested_dual(y, DT; ad_type=:mixed), args) # apparently need to seed with zero partials to avoid some confusion when tag type is similar, i.e. Tag{typeof(f),Float64} gets confused with Tag{typeof(f),Dual{Tag{typeof(f),Float64}}}, resulting in BNi being evaluated incorrectly, the value which should have occupied value.partials, now occupies partials.value
    dy = solve_ift(nested_pvalue(extract_partials_(BNi)), neg_A) # compute the inner partials.value....value (inner most value field of partials first). 
    y_star = make_dual(y, DT, seed_nested_dual(dy, valtype(DT); ad_type=:mixed)) # promote to next order dual, with inner most value field of partials solved ,all other fields are zero. 
    BNi = f(y_star, args) 
    dy = ift_mixed_((dy,nothing), y, extract_partials_(BNi), f, args, neg_A, DT) # compute partials and store as matrix/vector of scalar Duals
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
