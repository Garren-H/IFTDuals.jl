"""
```julia
    IFTStruct{Y<:AbstractVector{<:Dual},F<:AbstractVector{<:Dual}}
```
Struct to help identify cases of inplace functions and store this buffer to be reused across evaluations. 
"""
struct IFTStruct{Y<:AbstractVector{<:Dual},F<:AbstractVector{<:Dual}}
    y::Y
    F::F
end
IFTStruct(y::AbstractVector,::Type{DT}) where {DT<:Dual} = IFTStruct(init_y(y,DT), similar(y,DT))

# Functions to take a dual as input, and transforms the dual into one of order + 1 and seeds this order + 1 directional derivatives with zeros. We assume symetric derivative structures.
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
function ift_(y::Y, BNi::B, neg_A, fB::FB) where {Y,T,V,N,DT<:Dual{T,V,N},B<:Union{DT,AbstractArray{DT}},FB<:Function}
    if V <: Dual # recursion
        return ift_(y,extract_partials_(BNi),neg_A,extract_partials_∘fB)
    end
    return solve_partials(y,BNi,fB,neg_A)
end

# handle both in-place and out-of-place functions
function getBNi(y::IFTStruct,f!::F,args,fB::FB) where {F<:Function,FB<:Function}
    f!(y.F,fB(y.y),fB(args))
    return fB(y.F)
end
getBNi(y::Y,f::F,args,fB::FB) where {Y<:Union{<:Dual,AbstractVector{<:Dual}},F<:Function,FB<:Function} = f(fB(y),fB(args))

"""
```julia
    ift_recursive(y::Union{V,<:AbstractVector{V}},f::Function,args,neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}},der_order::Int) where {V<:Real,V2<:Real}
```
Recursively applies the implicit function theorem to compute higher-order derivatives up to `der_order`. It evaluates the function `f` at the current `y`, solves for the directional derivatives using `ift_`, and promotes `y` to the next order of `Dual` numbers as needed.
"""
function ift_recursive(y::Y, f::F, args, neg_A, ::Type{DT}, fY::FY=identity) where {Y,F<:Function,T,V,N,DT<:Dual{T,V,N},FY<:Function}
    if V<:Dual # recursion
        y = ift_recursive(y, f, args, neg_A, pvalue(DT), pvalue ∘ fY) # recursive make_dual
        y = seed_symm(y,fY) # seed symmetrically
        BNi = getBNi(y,f,args,fY) 
        return ift_(y, BNi, neg_A, fY) # next order IFT and return Dual
    end
    BNi = getBNi(y,f,args,fY) 
    return ift_(y, BNi, neg_A, fY) # first order IFT and return Dual
end

reshape_(x::AbstractArray{V,N}) where {V,N} = N > 2 ? reshape(x, (size(x,1),:)) : x # reshape to matrix/array by using a view
"""
Function to solve for the partials.value field and return the partial dual (zero out partials.partials)
"""
function solve_partials_value(BNi::B,y::Y,f::F,args,fY::FY,fB::FB,neg_A) where {B<:Union{<:Dual,AbstractArray{<:Dual}},Y,F<:Function,FY<:Function,FB<:Function}
    y,fB_,seeded,needs_solve = seed_mixed(y,fB∘fY)
    if seeded
        fB = fB_∘fB
        if !needs_solve
            return y,fB
        end
        # If we are here we have symmetrically seeded some levels but not until an inner most value field, so we need to adjust.
        # We may have something like Dual{T1,Dual{T1,Dual{T2,Dual{T2}}}} (outer 2 layers and inner 2 layers are symmetric). In this case
        # value.partials == partials.value, but partials.partials (a 2nd order dual of type Dual{T2,Dual{T2}}) is unseeded. We hence need to 
        # solve the partials.partials.value.value before we can continue. If we end up here, we are solving this partials.partials.value.value field
        fB = extract_partials_ ∘ fB 
        BNi = fB(getBNi(y,f,args,fY)) # previous BNi is irrelevant since we have seeded some levels previous BNi does not account for these additional seeds added 
    end    
    fE = nested_pvalue ∘ extract_partials_
    return solve_partials(y,BNi,fB,neg_A,fE),fB # need to solve here
end
function solve_partials_value(y::Y,f::F,args,fY::FY,fB::FB,neg_A) where {Y,F<:Function,FY<:Function,FB<:Function}
    y,fB_,seeded,needs_solve = seed_mixed(y,fB∘fY)
    if seeded 
        fB = fB_∘fB
        if !needs_solve
            return y,fB
        end
        # Same logic as above
        fB = extract_partials_ ∘ fB
    end
    BNi = fB(getBNi(y,f,args,fY))
    fE = nested_pvalue ∘ extract_partials_
    return solve_partials(y,BNi,fB∘fY,neg_A,fE),fB
end

"""
Solves the partials.partials fields knowing the partials.value as dy and combines to get the full partials field.
"""
function solve_partials end
# scalar y with single partial in all directions
function solve_partials(y::Dual, BNi::DT, fB::FB, neg_A, fE::FE=extract_partials_) where {T,V,DT<:Dual{T,V,1},FB<:Function,FE<:Function}
    dyy = solve_ift(fE(BNi), neg_A) # solve for partials field
    return update_dual(y,dyy,extract_partials_∘fB) 
end
# generic case
function solve_partials(y::Y, BNi::B, fB::FB, neg_A, fE::FE=extract_partials_) where {Y<:Union{<:Dual,AbstractVector{<:Dual}},B<:Union{<:Dual,AbstractArray{<:Dual}},FB<:Function,FE<:Function}
    dyy = collect(fE(BNi)) # materialize for in-place solves
    dyy_ = reshape_(dyy) # always solve a matrix/vector 
    solve_ift(dyy_, neg_A) # solve for partials field in-place, updates dyy
    return update_dual(y,dyy,extract_partials_∘fB)
end
function solve_partials(y::IFTStruct, BNi::B, fB::FB, neg_A, fE::FE=extract_partials_) where {B<:Union{<:Dual,AbstractArray{<:Dual}},FB<:Function,FE<:Function}
    solve_partials(y.y,BNi,fB,neg_A,fE)
    return y
end
"""
```julia
    ift_mixed_(dy::Union{V,<:AbstractVector{V}},y::Union{V,<:AbstractVector{V}},BNi::Union{Dual{T,V,N},<:AbstractVector{Dual{T,V,N}}},f::Function,args,neg_A::Union{V2,<:LU{V2,<:AbstractMatrix{V2},<:AbstractVector{<:Integer}}},target_DT::Type{Dual{T,V,N}},fB::Function=extract_partials_) where {T,V,N,V2<:Real}
```
For a given order of differentiation with mixed tags, recursively computes all directional derivatives using the implicit function theorem and (nested) Dual numbers.
"""
function ift_mixed_(BNi::B, fB::FB, y::Y, f::F, fY::FY, args, neg_A) where {T,V,N,VB<:Dual{T,V,N},F<:Function,FB<:Function,FY<:Function,Y,B<:Union{VB,AbstractArray{VB}}}
    if V<:Dual # Recursion
        y = ift_mixed_(pvalue(BNi), pvalue∘fB, y, f, fY, args, neg_A) # returns the partials.value field with all fields solved. 
        y,fB = solve_partials_value(BNi, y, f, args, fY, fB, neg_A)  
        BNi = fB(getBNi(y,f,args,fY)) # Evaluate at the new dual and perform
        return ift_mixed_(extract_partials_(BNi), extract_partials_ ∘ fB, y, f, fY, args, neg_A)
    end
    y = solve_partials(y,BNi,fB∘fY,neg_A)
    return y 
end
function ift_recursive_mixed(y::Y, f::F, args, neg_A, ::Type{DT}, fY::FY=identity) where {Y,F<:Function,T,V,N,DT<:Dual{T,V,N},FY<:Function}
    if V <: Dual # recursion
        y = ift_recursive_mixed(y, f, args, neg_A, pvalue(DT), pvalue∘fY) # get previous values first
        y,fB = solve_partials_value(y, f, args, fY, identity, neg_A)
        BNi = fB(getBNi(y,f,args,fY))
        return ift_mixed_(extract_partials_(BNi), extract_partials_∘fB, y, f, fY, args, neg_A)
    end
    return ift_recursive(y, f, args, neg_A, DT, fY) # first order IFT and return Dual
end
"""
```julia
    function ift(y::Union{V,<:AbstractArray{V}},f::Function,args; DT::Type{Union{Nothing,<:Dual}}=nothing, tag_is_mixed::Union{Nothing,Bool}=nothing, args_needs_promotion::Bool=true) where {V<:Real}
    function ift(y::Union{V,<:AbstractArray{V}},f::Function,args,args_primal; DT::Type{Union{Nothing,<:Dual}}=nothing, tag_is_mixed::Union{Nothing,Bool}=nothing, args_needs_promotion::Bool=true) where {V<:Real}
```
Function to compute higher-order derivatives using the implicit function theorem (IFT) and (nested) Dual numbers. 
Input:
- `y`    : primal input solution to the root finding problem (scalar or vector)
- `f`    : function handle that takes 'y' and 'args' as inputs. `f(y,args) = 0` is assumed to define the implicit relationship between `y` and values given as `args`. When `inplace=true`, then `f!(F,y,tups)` is expected
- `args` : tuple or data structure containing Dual numbers indicating the differentiation structure
- `args_primal` : primal values of `args`
Optional Input:
- `DT`                  : Target Dual type for the output. If not provided, it is inferred from `args`.
- `tag_is_mixed`        : Boolean indicating if mixed tags are present in `args`. If not provided, it is inferred from `args`.
- `args_needs_promotion`: Boolean indicating if `args` need to be promoted to a common Dual type.
- `inplace`             : Boolean indicating if an in-place version of `f` is being used. If `true`, we expect `f!(F,y,tups)`, where `F` is the output to be mutated. `F` is created internally, once with primal eltypes to compute the Jacobian, and once more when evaluating the Duals. The second time `F` is created with `F=similar(y, DT)` and reused for each differentiation order. Lower order duals are hence always promoted to the target Dual type. 
Output:
- Returns `y` as a Dual number with the appropriate order and partial derivatives computed using the IFT.
The function works by first determining the order of differentiation and whether mixed tags are present.

!!! warning
    When mixed tags are detected, but `tag_is_mixed=false` is provided, a check is performed to ensure that the number of partials are consistent across the Dual types in `args`. If inconsistencies are found, an error is thrown. No other checks are performed and it is the user's responsibility to ensure symmetry of partial derivatives in this case.

!!! warning
    If `args_needs_promotion=false` is provided, it is the user's responsibility to ensure that all Dual types in `args` are of the same type. If the Dual type differs, unexpected behavior may occur.
"""
function ift(y::Y, f::F, args; DT::DTT=nothing, tag_is_mixed::TM=nothing, args_needs_promotion::Bool=true,Fbuff::FF=nothing,inplace::Bool=false) where {V<:Real,Y<:ScalarOrAbstractVec{V},FF<:Union{Nothing,AbstractVector{V}},F<:Function,DTT<:Union{Nothing,Type{<:Real}},TM<:Union{Nothing,Bool}}
    args_primal = nested_pvalue(args)
    return ift(y, f, args, args_primal; DT=DT, tag_is_mixed=tag_is_mixed, args_needs_promotion=args_needs_promotion,Fbuff,inplace=inplace)
end

function ift(y::Y, f::F, args, args_primal; DT::DTT=nothing, tag_is_mixed::TM=nothing, args_needs_promotion::Bool=true,Fbuff::FF=nothing,inplace::Bool=false) where {V<:Real,Y<:ScalarOrAbstractVec{V},FF<:Union{Nothing,Vector{V}},F<:Function,DTT<:Union{Nothing,Tuple{<:Real}},TM<:Union{Nothing,Bool}}
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
    !inplace && !(FF <: Nothing) && (inplace=true) # in case a buffer is provided, but user did not set inplace to true
    der_order == 1 && return ift(y, f, args, args_primal, tag_is_mixed, DT, Fbuff, Val(inplace)) # first order case, no promotion needed
    args_ = args_needs_promotion ? promote_common_dual_type(args, DT) : args # promote to common dual type if needed
    return ift(y, f, args_, args_primal, tag_is_mixed, DT, Fbuff, Val(inplace)) # higher order case
end

function ift(y::Y, f::F, args, args_primal, tag_is_mixed::Bool, ::Type{DT}, ::Nothing, ::Val{false}) where {Y<:Real,F<:Function,DT<:Dual}
    neg_A = -derivative(f, AFD, y, Constant(args_primal))
    yy = DT(y)
    return ift(yy,f,args,neg_A,DT,tag_is_mixed)
end
function ift(y::Y, f::F, args, args_primal, tag_is_mixed::Bool, ::Type{DT}, ::Nothing, ::Val{false}) where {V<:Real,Y<:AbstractVector{V},F<:Function,DT<:Dual}
    neg_A = jacobian(f, AFD, y, Constant(args_primal))::AbstractMatrix{V}
    neg_A = prep_negA_(neg_A)
    yy = init_y(y,DT)
    return ift(yy,f,args,neg_A,DT,tag_is_mixed)
end
function ift(y::Y, f!::F, args, args_primal, tag_is_mixed::Bool, DT::Type{<:Dual}, Fbuff::FF, ::Val{true}) where {V<:Real,Y<:AbstractVector{V},F<:Function,FF<:Union{Nothing,AbstractVector{V}}}
    Fbuff = Fbuff === nothing ? similar(y) : (@assert size(Fbuff) == size(y) "Provided buffer Fbuff must have the same size as y."; Fbuff)
    neg_A = jacobian(f!, Fbuff, AFD, y, Constant(args_primal))::AbstractMatrix{V}
    neg_A = prep_negA_(neg_A)
    yy = IFTStruct(y,DT)
    return ift(yy,f!,args,neg_A,DT,tag_is_mixed)
end

function ift(y::Y,f::F,args,neg_A,::Type{DT},tag_is_mixed::Bool) where {DT<:Dual,Y<:Union{DT,<:AbstractVector{DT},IFTStruct},F<:Function}
    yy = tag_is_mixed ? ift_recursive_mixed(y, f, args, neg_A, DT) : ift_recursive(y, f, args, neg_A, DT)
    return yy isa IFTStruct ? yy.y : yy
end

function prep_negA_(neg_A::AbstractMatrix{V}) where V<:Real
    checksquare(neg_A) # Ensure square matrix
    _1 = -one(V)
    if ismutable(neg_A) 
        neg_A .*= _1
        neg_A = lu!(neg_A)
    else # safely handle cases of StaticArrays
        neg_A = _1 * neg_A
        neg_A = lu(neg_A) # LU factorization for later solves
    end
    return neg_A
end

init_y(y::Y,::Type{DT}) where {Y<:AbstractVector,DT<:Dual} = convert(Vector{DT},y) # Use init_y such that we can overload for StaticArrays 

export ift
