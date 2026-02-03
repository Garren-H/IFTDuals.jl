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

f(x) = x;

x = [1.0, 2.0];

@testset "Checking pvalue and nested_pvalue" begin
    # check that pvalue and nested_pvalue give correct results for first order duals
    dx1 = make_dual(f,x)
    test11 = ForwardDiff.value.(dx1) .== nested_pvalue(dx1)
    test12 = nested_pvalue(dx1) .== pvalue(dx1)
    @test all(test11) && all(test12)
    
    # check that pvalue and nested_pvalue give correct results for second order duals
    dx2 = make_dual(f,x,2)
    test21 = ForwardDiff.value.(dx2) .== pvalue(dx2)
    test22 = nested_pvalue(dx2) .== begin
            tmp = ForwardDiff.value.(dx2)
            while eltype(tmp) <: ForwardDiff.Dual
                tmp = ForwardDiff.value.(tmp)
            end
            tmp
        end
    @test all(test21) && all(test22)
end

@testset "Checks for dual promotion" begin
    Tagx = Tag{:x,Float64}
    dx = make_dual(Tagx,1.0)
    Tagy = Tag{:y,Float64}
    dy = make_dual(Tagy,2.0)
    struct NewStruct{Tx,Ty}
        x::Tx
        y::Ty
    end
    NewStruct(x::Tx, y::Ty) where {Tx,Ty} = NewStruct{Tx,Ty}(x,y)
    ds = NewStruct(dx, dy)
    # get common Dual Type
    DT_common_ = IFTDuals.get_common_dual_type(ds)
    DT_common = promote_type(eltype(dx),eltype(dy))
    @test DT_common_ === DT_common # check that get_common_dual_type gives correct Dual promotion
    ds_promoted_ = promote_common_dual_type(ds,DT_common_)
    ds_promoted = NewStruct(convert(DT_common,dx), convert(DT_common,dy))
    @test (ds_promoted_.x == ds_promoted.x) && (ds_promoted_.y == ds_promoted.y) && (typeof(ds_promoted_).parameters == typeof(ds_promoted).parameters)
end
