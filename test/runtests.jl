using StaticArrays,IFTDuals,DifferentiationInterface,Test
import ForwardDiff
import ForwardDiff: Dual, Tag, Partials
const AFD = AutoForwardDiff()

# some functions to create Dual numbers for testing
function make_dual(f::F, x::IFTDuals.ScalarOrAbstractVec{V}) where {F<:Function,V<:Real}
    Tagx = Tag{typeof(f),V}
    return make_dual(Tagx, x)
end
function make_dual(Tagx::Type{<:Tag}, x::IFTDuals.AbstractVector{V}) where {V<:Real}
    N = length(x)
    _1 = one(V)
    _0 = zero(V)
    DT = Dual{Tagx,V,N}
    PT = Partials{N,V}
    dx = similar(x,DT)
    for i in eachindex(x)
        dx[i] = DT(x[i], PT(ntuple(j -> i == j ? _1 : _0, N)))
    end
    return dx
end
function make_dual(Tagx::Type{<:Tag}, x::V) where {V<:Real}
    DT = Dual{Tagx,V,1}
    PT = Partials{1,V}
    _1 = one(V)
    return DT(x, PT((_1,)))
end
function make_dual(f::F, x::IFTDuals.ScalarOrAbstractVec{V}, der_order::Int) where {F<:Function,V<:Real}
    Tagx = Tag{typeof(f),V}
    return make_dual(Tagx, x, der_order)
end
function make_dual(Tagx::Type{<:Tag}, x::AbstractVector{V}, der_order::Int) where {V<:Real}
    dx = make_dual(Tagx, x) # first order duals
    der_order > 1 || return dx
    dx = convert(Vector{Dual},dx)
    N = length(x)
    for _ in 2:der_order
        DT_prev = eltype(first(dx))
        Tag_prev = ForwardDiff.tagtype(DT_prev)
        Tag_new = Tag{Tag_prev.parameters[1],DT_prev}
        DT_new = Dual{Tag_new,DT_prev,N}
        PT_new = Partials{N,DT_prev}
        for i in eachindex(dx)
            dx[i] = DT_new(dx[i], PT_new(ntuple(j -> DT_prev(dx[i].partials[j]), N)))
        end
    end
    return convert(Vector{eltype(first(dx))},dx)
end
function make_dual(Tagx::Type{<:Tag}, x::V, der_order::Int) where {V<:Real}
    dx = make_dual(Tagx, x) # first order dual
    der_order > 1 || return dx
    N = 1
    for _ in 2:der_order
        DT_prev = eltype(dx)
        Tag_prev = ForwardDiff.tagtype(DT_prev)
        Tag_new = Tag{Tag_prev.parameters[1],DT_prev}
        DT_new = Dual{Tag_new,DT_prev,N}
        PT_new = Partials{N,DT_prev}
        dx = DT_new(dx, PT_new((DT_prev(dx.partials[1]),)))
    end
    return dx
end

# tests
include("test_derivatives.jl")
include("test_utils.jl")
