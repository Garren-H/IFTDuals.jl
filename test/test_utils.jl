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
    ds = NewStruct(dx, dy)
    # get common Dual Type
    DT_common_ = IFTDuals.get_common_dual_type(ds)
    DT_common = promote_type(eltype(dx),eltype(dy))
    @test DT_common_ === DT_common # check that get_common_dual_type gives correct Dual promotion
    ds_promoted_ = promote_common_dual_type(ds,DT_common_)
    ds_promoted = NewStruct(convert(DT_common,dx), convert(DT_common,dy))
    @test (ds_promoted_.x == ds_promoted.x) && (ds_promoted_.y == ds_promoted.y) && (typeof(ds_promoted_).parameters == typeof(ds_promoted).parameters)
    # test vectors, and extraction of pvalue and nested_pvalue on promoted_duals
    x = [1.0, 2.0]
    y = [3.0, 4.0, 4.0]
    dx = make_dual(Tagx, x)
    dy = make_dual(Tagy, y)
    dt = (dx,dy)
    DT_common_ = IFTDuals.get_common_dual_type(dt)
    DT_common = promote_type(eltype(dx),eltype(dy))
    @test DT_common_ === DT_common
    dt_promoted_ = promote_common_dual_type(dt,DT_common_)
    dt_promoted = (convert(Vector{DT_common},dx), convert(Vector{DT_common},dy))
    @test all(dt_promoted[1] .== dt_promoted_[1]) && all(dt_promoted[2] .== dt_promoted_[2])
    pval_t = pvalue(dt_promoted)
    pval_t_ = (ForwardDiff.value.(dt_promoted_[1]), ForwardDiff.value.(dt_promoted_[2]))
    @test all(pval_t[1] .== pval_t_[1]) && all(pval_t[2] .== pval_t_[2])
    nested_pval_t = nested_pvalue(dt_promoted)
    @test all(nested_pval_t[1] .== x) && all(nested_pval_t[2] .== y)
end
