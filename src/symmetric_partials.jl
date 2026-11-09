has_multiple_partials(::Number) = true
has_multiple_partials(D::Type{<:Dual}) = npartials(D) > 1 || has_multiple_partials(valtype(D))
# Type params: (number of times var is differentiated per group, number of diff variables per group, total number of unique permutations of differentiation to consider, number of groups of variables, order of differentiation + 1) 
struct DTuple{Ks,Ns,NDers,M,NDims}
    offsetMultiplier::NTuple{M,Int}
    idxStartMin1::NTuple{M,Int}
end

DTuple{Tuple{1},Tuple{N}}() where {N} = DTuple{Tuple{1},Tuple{N},N,1,2}((1,), (0,)) # first order
@generated function DTuple{Ks,Ns}() where {Ks,Ns}
    K = Ks.parameters
    N = Ns.parameters
    M = length(Ks.parameters)
    NDims = sum(Ks.parameters) + 1
    S = 0
    O = 1
    s_tups_expr = Expr(:tuple, S)
    o_tups_expr = Expr(:tuple, O)
    for i in 2:M
        S += K[i - 1]
        idx_back = M - i + 2
        O *= NUniqueDers(N[idx_back], K[idx_back])
        push!(s_tups_expr.args, S)
        pushfirst!(o_tups_expr.args, O)
    end
    O *= NUniqueDers(N[1], K[1])
    :(DTuple{$Ks,$Ns,$O,$M,$NDims}($o_tups_expr, $s_tups_expr))
end

getKs(D::Type{<:DTuple}) = D.parameters[1].parameters
getNs(D::Type{<:DTuple}) = D.parameters[2].parameters
getNders(D::Type{<:DTuple}) = D.parameters[3]
getNgroups(D::Type{<:DTuple}) = D.parameters[4]

NUniqueDers(N::Int, K::Int) = binomial(N + K - 1, K) # for a group of variables of length N and diff order K, we have this many unique derivatives

DTuple(::Type{Dual{T,V,N}}) where {T,V,N} = DTuple{Tuple{1},Tuple{N},N,1,2}((1,), (0,)) # first order diff
@generated function DTuple(::Type{Dual{T,V,N}}) where {T,V<:Dual,N} # generic Nth order diff
    T0 = T
    N0 = N
    V0 = Dual{T,V,N}
    dorder = order(V0)
    K = 1
    M = 1
    s = 0
    Ks = [] # tracks diff order per group
    Ns = [] # tracks number of variable per group
    s_tups_expr = Expr(:tuple, s) # track start:end to parse CartesianIndex into groups. Used to get the linear index
    for _ in 2:dorder
        V1 = valtype(V0)
        T1 = tagtype(V1)
        N1 = npartials(V1)
        if is_symm(T0, T1) && (N0 === N1)
            K += 1
        else
            M += 1
            s += K
            push!(Ks, K)
            push!(Ns, N0)
            push!(s_tups_expr.args, s)
            K = 1
        end
        V0 = V1
        T0 = T1
        N0 = N1
    end
    push!(Ks, K)
    push!(Ns, N0)

    O = 1 # total number of unique derivatives
    o_tups_expr = Expr(:tuple, O) # multiplier of groups, used in computation of linear index
    for group in M:-1:2
        O *= NUniqueDers(Ns[group], Ks[group])
        pushfirst!(o_tups_expr.args, O)
    end
    O *= NUniqueDers(Ns[1], Ks[1])
    :(DTuple{$(Tuple{Ks...}),$(Tuple{Ns...}),$O,$M,$(dorder + 1)}($o_tups_expr, $s_tups_expr))
end

# copy of julia mergsort for tuples, but removing unnecasary details
mergesort(x::Union{NTuple{0},NTuple{1}}) = x
function mergesort(x::NTuple)
    a, b = Base.IteratorsMD.split(x, Val(length(x) >> 1))
    merge(mergesort(a), mergesort(b))
end
merge(x::NTuple, y::NTuple{0}) = x
merge(x::NTuple{0}, y::NTuple) = y
merge(x::NTuple{0}, y::NTuple{0}) = x # Method ambiguity
merge(x::NTuple, y::NTuple) =
    y[1] < x[1] ? (y[1], merge(x, Base.tail(y))...) : (x[1], merge(Base.tail(x), y)...)

@generated function get_size(data::AbstractVecOrMat, ::Type{D}) where {D<:DTuple}
    Ks = getKs(D)
    Ns = getNs(D)
    M = getNgroups(D)
    Expr(:tuple, :(size(data, 2)), [Ns.parameters[i] for i in 1:M for _ in 1:Ks.parameters[i]]...) # use generated to avoid conditionals and complex logic
end

struct SymmetricPartials{T,N,Ks,Ns,NDers,M,D<:AbstractVecOrMat{T}} <: AbstractArray{T,N}
    data::D
    dInfo::DTuple{Ks,Ns,NDers,M,N}
    size::NTuple{N,Int}
end

function SymmetricPartials(
    d::Union{Number,AbstractVector{<:Number}},
    D::Type{<:Dual},
    useStatic::Val,
)
    dInfo = DTuple(D)
    return SymmetricPartials(d, dInfo, useStatic)
end

function SymmetricPartials(::T, dInfo::DTuple, ::Val{false}) where {T} # final arg is whether to use static arrays instead
    data = Vector{T}(undef, getNders(typeof(dInfo)))
    return SymmetricPartials(data, dInfo)
end

function SymmetricPartials(d::AbstractVector{T}, dInfo::DTuple, ::Val{false}) where {T}
    data = Matrix{T}(undef, getNders(typeof(dInfo)), length(d))
    return SymmetricPartials(data, dInfo)
end

function SymmetricPartials(
    data::D,
    dInfo::DTuple{Ks,Ns,NDers,M,N},
) where {T,N,Ks,Ns,NDers,M,D<:AbstractVecOrMat{T}}
    size_ = get_size(data, dInfo)
    return SymmetricPartials{T,N,Ks,Ns,NDers,M,D}(data, dInfo, size_)
end

Base.IndexStyle(::Type{<:SymmetricPartials}) = Base.IndexCartesian()
Base.IndexStyle(::SymmetricPartials) = Base.IndexCartesian()

Base.size(X::SymmetricPartials) = X.size

@inline function Base.getindex(X::SymmetricPartials{<:Any,2,Tuple{1}}, indices::Vararg{Int,2}) # overload for 1st order diff
    fidx, xidx = first(indices), last(indices)
    @boundscheck checkbounds(X.data, xidx, fidx)
    return @inbounds X.data[xidx, fidx]
end

function Base.getindex(X::SymmetricPartials{T,N,Ks}, indices::Vararg{Int,N}) where {T,N,Ks}
    @boundscheck checkbounds(X, indices...)
    dx_idxs = Base.tail(indices)
    lin_idx = _get_linear_index(X.dInfo, dx_idxs)
    fidx = first(indices)
    @boundscheck checkbounds(X.data, lin_idx, fidx) # precaution, but shouldnt fail
    return @inbounds X.data[lin_idx, fidx]
end

function _get_linear_index(dInfo, dx_idxs)
    idxStartMin1 = dInfo.idxStartMin1
    offsetMultiplier = dInfo.offsetMultiplier
    D = typeof(dInfo)
    Ks = getKs(D)
    Ns = getNs(D)
    lin_idx = 1
    for (i, ki) in enumerate(Ks)
        subset_idxs = ntuple(k -> dx_idxs[idxStartMin1[i] + k], Val(ki))
        sorted_idxs = mergesort(subset_idxs)
        lin_idx += rel_group_idx(Ns[i], sorted_idxs, offsetMultiplier[i])
    end
    return lin_idx
end

# function rel_group_idx(::Val{Ki}, Ni::Int, Si::Int, Oi::Int, dx_idxs::NTuple, i::Int) where {Ki}
#     subset_idxs = ntuple(k -> dx_idxs[Si + k], Val(Ki))
#     sorted_idxs = mergesort(subset_idxs)
#     return rel_group_idx(Ni, sorted_idxs, Oi)
# end

_rel_group_idx(N::Int, Kminj::Int, idxj::Int, idxjprev::Int) =
    idxj > idxjprev ? sum(binomial(N - v + Kminj, Kminj) for v in idxjprev:(idxj - 1)) : 0 # within a group, counts the number of elements which are skipped. From stats, we have fixed the first j elements leaving K-j derivative positions to fill - hence K-j as the second argument of binomial. The N-v part stems from how many elements we can consider. We have the indices sorted from smallest to largest, so we can only consider indices greater or equal to the previous index. If the next index is fixed, the number of indices/elements which may be chosen from shrinks as well, hence the summation over shrinking number of elements to be chosen from. Reduction over empty sequence errors, so we have conditional to return zero offset if empty 
rel_group_idx(N::Int, indices::NTuple{K,Int}) where {K} =
    _rel_group_idx(N, K - 1, indices[1], 1) +
    sum(_rel_group_idx(N, K - j, indices[j], indices[j - 1]) for j in 2:K)
rel_group_idx(N::Int, indices::NTuple{1,Int}) = indices[1] - 1 # shorcut for 1st order ders, which simplifies to zero based indexing
rel_group_idx(N_or_NAndK, indices, offsetMultiplier::Int) =
    rel_group_idx(N_or_NAndK, indices) * offsetMultiplier

#TODO: Implement a drop(A::SymmetricPartials) and drop(dInfo::DTuple) methods based on extract_partial, pvalue and nested_pvalue as input. 
#The intention here is to 

function drop_SymmArr(X, f) # for a given symmetric partials  
    new_dInfo = drop_dtuple(X.dInfo, f)
    Nders = getNders(new_dInfo)
    Nfuncs = size(X, 1)
    new_data = view(X.data, 1:(Nders * Nfuncs))
    SymmetricPartials(
        X.data isa AbstractVector ? new_data : reshape(new_dInfo, Nders, Nfuncs),
        new_dInfo,
    )
end

drop_dtuple(dInfo, f) = _drop!(dInfo, Base.unwrap_composed(f))

@generated function _drop!(::DTuple{Ks,Ns}, ::Fs) where {Ks,Ns,Fs<:Tuple}
    AllFs = Fs.parameters[end:-1:1] # innert most function first. Outer function may have nested_pvalue which will yield incorrect result
    NewKs = []
    NewNs = []
    idx = 1
    curr_kval = 0
    add_first = true
    for F in AllFs
        curr_kval += 1
        if F == typeof(extract_partials_)
            if curr_kval === 1
                push!(NewKs, curr_kval)
                push!(NewNs, Ns.parameters[idx])
            elseif add_first
                push!(NewKs, 1)
                push!(NewNs, Ns.parameters[idx])
            else
                NewKs[end] += 1
            end
        end
        if curr_kval === Ks.parameters[idx] # move to next level
            idx += 1
            curr_kval = 0
        end
    end
    :(DTuple{Tuple{$(NewKs...)},Tuple{$(NewNs...)}}())
end
