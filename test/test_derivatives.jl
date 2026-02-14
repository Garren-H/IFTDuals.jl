@testset "Compute derivatives for: Vector, Tuple and Struct" begin
    struct MyStruct{T<:Real}
        θ::Vector{T}
    end

    function f(x,θ)
        f1 = x[1]^2 + x[2]^2 + x[3]^2 - θ[1]^2
        f2 = θ[2]*x[1] + x[2]
        f3 = θ[3]*x[1] + θ[4]*x[3] + θ[5]
        return [f1, f2, f3]
    end
    f_static(x,θ) = SVector{3}(f(x,θ));

    closed_form_solution(θ; plusmin::Symbol=:plus) = begin
        c = θ[5]^2 / θ[4]^2 - θ[1]^2
        b = 2 * θ[3] * θ[5] / θ[4]^2
        a = 1 + θ[2]^2 + θ[3]^2 / θ[4]^2
        if plusmin == :plus
            x1 = (-b + sqrt(b^2 - 4a*c)) / (2a)
        elseif plusmin == :min
            x1 = (-b - sqrt(b^2 - 4a*c)) / (2a)
        else
            error("plusmin must be :plus or :min")
        end
        x2 = - θ[2] * x1
        x3 = -(θ[3]/θ[4]) * x1 - (θ[5]/θ[4])
        [x1, x2, x3]
    end;

    function get_x(θ::AbstractVector; plusmin::Symbol=:plus)
        θp = nested_pvalue(θ)
        x = closed_form_solution(θp; plusmin=plusmin) # replace with some root solver
        return get_x_AD(x, θ)
    end;

    function get_x(θ::Tuple; plusmin::Symbol=:plus)
        θp = nested_pvalue(θ)
        x = closed_form_solution(θp[1]; plusmin=plusmin) # replace with some root solver
        return get_x_AD(x, θ)
    end;

    function get_x(θ::MyStruct; plusmin::Symbol=:plus)
        θp = nested_pvalue(θ)
        x = closed_form_solution(θp.θ; plusmin=plusmin) # replace with some root solver
        return get_x_AD(x, θ)
    end;


    function get_x_AD(x, θ)
        return ift(x, f, θ)
    end;

    function get_x_AD(x, θ::Tuple)
        return ift(x, (x,p)->f(x,p[1]), θ)
    end;

    function get_x_AD(x, θ::MyStruct)
        return ift(x, (x,p)->f(x,p.θ), θ)
    end;

    θ = [0.3909285031906048,
         0.3358470459314826, 
         0.9415232362877466, 
         0.13053206204810686, 
         0.30149241769328894];

    # Types to construct Duals
    V = eltype(θ)
    T = typeof(Tag(get_x, V))
    N = length(θ)

    # Construct Duals
    DT₁ = Dual{T,V,N}
    PT₁ = Partials{N,V}
    θ₁ = [DT₁(θ[i], PT₁(ntuple(j -> j == i ? 1.0 : 0.0, N))) for i in 1:N]

    DT₂ = Dual{T,DT₁,N}
    PT₂ = Partials{N,DT₁}
    θ₂ = [DT₂(θ₁[i], PT₂(ntuple(j -> DT₁(θ₁[i].partials[j]), N))) for i in 1:N]

    DT₃ = Dual{T,DT₂,N}
    PT₃ = Partials{N,DT₂}
    θ₃ = [DT₃(θ₂[i], PT₃(ntuple(j -> DT₂(θ₂[i].partials[j]), N))) for i in 1:N]

    # Compute x with 1st, 2nd, and 3rd order Duals
    x₁ₜ = closed_form_solution(θ₁);
    x₂ₜ = closed_form_solution(θ₂);
    x₃ₜ = closed_form_solution(θ₃);
    @testset "Vector Input: 1st, 2nd and 3rd order Duals" begin
        x₁ = get_x(θ₁);
        @test x₁ ≈ x₁ₜ

        x₂ = get_x(θ₂);
        @test x₂ ≈ x₂ₜ

        x₃ = get_x(θ₃);
        @test x₃ ≈ x₃ₜ
    end

    # Tuple Input
    θ_tuple₁ = (θ₁,);
    @testset "Tuple Input: 1st order Duals" begin
        x₁_tuple = get_x(θ_tuple₁);
        @test x₁_tuple ≈ x₁ₜ
    end

    # Struct Input
    θ_struct₁ = MyStruct(θ₁);
    @testset "Struct Input: 1st order Duals" begin
        x₁_struct = get_x(θ_struct₁);
        @test x₁_struct ≈ x₁ₜ
    end

    # test using SVector as input
    get_x_SVector(θ;tag_is_mixed::Bool=false,plusmin::Symbol=:plus) = begin
        θp = nested_pvalue(θ)
        x = SVector{3}(closed_form_solution(θp; plusmin=plusmin))
        return ift(x,f_static,θ,θp;tag_is_mixed=tag_is_mixed)
    end
    @testset "SVector Input: 1st and 2nd order using both single and mixed tags" begin
        x₁_svec = get_x_SVector(θ₁; tag_is_mixed=false);
        @test x₁_svec ≈ x₁ₜ
        x₂_svec = get_x_SVector(θ₂; tag_is_mixed=true);
        @test x₂_svec ≈ x₂ₜ
        x₂_svec = get_x_SVector(θ₂; tag_is_mixed=false);
        @test x₂_svec ≈ x₂ₜ
    end
end

@testset "Compute value,gradient and hessian for Vector->Scalar mappings" begin
    f(y,x) = exp(y) - sum(abs2,x)
    h(x) = log(sum(abs2, x))
    x = [1.0, 2.0, -3.0]
    test_f(x) = begin
        xp = nested_pvalue(x)
        y = h(xp)
        ift(y,f,x,xp)
    end
    y_true = value_gradient_and_hessian(h,AFD,x)
    y_ift = value_gradient_and_hessian(test_f,AFD,x)
    @test all(y_ift .≈ y_true)
end

@testset "Compute value,derivative and second derivative for Scalar->Scalar mappings" begin
    f(y,x) = exp(y) - x^3
    h(x) = 3*log(x)
    x = 2.0
    test_f(x) = begin
        xp = nested_pvalue(x)
        y = h(xp)
        ift(y,f,x,xp;tag_is_mixed=true)
    end
    y_true = value_derivative_and_second_derivative(h,AFD,x)
    y_ift = value_derivative_and_second_derivative(test_f,AFD,x)
    @test all(y_ift .≈ y_true)
    test_f_mixed(x) = begin
        xp = nested_pvalue(x)
        y = h(xp)
        ift(y,f,x,xp;tag_is_mixed=true)
    end
    y_ift_mixed = value_derivative_and_second_derivative(test_f_mixed,AFD,x)
    @test all(y_ift_mixed .≈ y_true)
end 

@testset "Mixed tags: Arbitrary order derivatives" begin
    # some functions to test with
    g1(x::Real, y::Real, w::Real) = exp(sin(x*y*w) + x^2*y + y^2*w + w^2*x) # Scalar -> Scalar mappings
    g2(x::Real, y::Real, w::Real) = [exp(sin(x*y*w) + x*y),exp(cos(x^2*w + y^2)),exp(sin(x + y + w + x*y*w))] # Scalar -> Vector mappings
    g3(x::AbstractVector, y::AbstractVector, w::Real) = begin # Vector -> Scalar mappings
        Sx = sum(x)
        Sy = sum(y)
        Qx = sum(x .^ 2)
        Qy = sum(y .^ 2)
        exp(sin(Sx * Sy * w) + Qx * Sy + Qy * Sx * w)
    end
    g4(x::AbstractVector, y::AbstractVector, w::Real) = begin # Vector -> Vector mappings
        Sx = sum(x)
        Sy = sum(y)
        Qy = sum(y .^ 2)
        exp.(sin.(x .* Sy .* w) .+ x.^2 .* Sy .+ Sx * Qy * w)
    end
    # ift function to test; using f(z,args) = z - g(args...) ↔ z = g(args...) 
    test_f(x,y,w,g) = begin
        xp = nested_pvalue(x)
        yp = nested_pvalue(y)
        wp = nested_pvalue(w)
        args_p = (xp, yp, wp)
        z = g(args_p...)
        args = (x,y,w)
        f(z,args) = z - g(args...) # change g to g1,g2,g3,g4 to test different functions
        ift(z,f,args,args_p)
    end
    Tagx = Tag{:x,Float64};
    Tagw = Tag{:w,Float64};
    Tagy = Tag{:y,Float64};

    @testset "Mixed tags: Scalar -> Scalar" begin
        x = randn(); y = randn(); w = randn();
        x1 = make_dual(Tagx,x,1); y1 = make_dual(Tagy,y,1); w1 = make_dual(Tagw,w,1); # first order duals
        x2 = make_dual(Tagx,x1,1); y2 = make_dual(Tagy,y1,1); w2 = make_dual(Tagw,w1,1); # second order duals
        g = g1
        dual_z = test_f(x2,y2,w1,g)
        dual_z_true = g(x2,y2,w1)
        @test dual_z ≈ dual_z_true
    end
    @testset "Mixed tags: Scalar -> Vector" begin
        x = randn(); y = randn(); w = randn();
        x1 = make_dual(Tagx,x,1); y1 = make_dual(Tagy,y,1); w1 = make_dual(Tagw,w,1); # first order duals
        x2 = make_dual(Tagx,x1,1); y2 = make_dual(Tagy,y1,1); w2 = make_dual(Tagw,w1,1); # second order duals
        g = g2
        dual_z = test_f(x2,y2,w1,g)
        dual_z_true = g(x2,y2,w1)
        @test all(dual_z .≈ dual_z_true)
    end
    @testset "Mixed tags: Vector -> Scalar" begin
        x = randn(3); y = randn(5); w = randn();
        x1 = make_dual(Tagx,x,1); y1 = make_dual(Tagy,y,1); w1 = make_dual(Tagw,w,1); # first order duals
        x2 = make_dual(Tagx,x1,1); y2 = make_dual(Tagy,y1,1); w2 = make_dual(Tagw,w1,1); # second order duals
        g = g3
        dual_z = test_f(x2,y2,w1,g)
        dual_z_true = g(x2,y2,w1)
        @test dual_z ≈ dual_z_true
    end
    @testset "Mixed tags: Vector -> Vector" begin
        x = randn(3); y = randn(5); w = randn();
        x1 = make_dual(Tagx,x,1); y1 = make_dual(Tagy,y,1); w1 = make_dual(Tagw,w,1); # first order duals
        x2 = make_dual(Tagx,x1,1); y2 = make_dual(Tagy,y1,1); w2 = make_dual(Tagw,w1,1); # second order duals
        g = g4
        dual_z = test_f(x2,y2,w1,g)
        dual_z_true = g(x2,y2,w1)
        @test all(dual_z .≈ dual_z_true)
        x1 = make_dual(Tagw,x,1); w1 = make_dual(Tagx,w,1); # switch ordering
        x2 = make_dual(Tagw,x,2); w2 = make_dual(Tagx,w,2); # switch ordering
        dual_z = test_f(x2,y1,w2,g) # just test that it works irrespective of ordering of tags
        dual_z_true = g(x2,y1,w2)
        @test all(dual_z .≈ dual_z_true)
        x1 = SVector{3}(x1); y1 = SVector{5}(y1)
        dual_z = test_f(x1,y1,w1,g) # just test that it works with SVector inputs
        dual_z_true = g(x1,y1,w1)
        @test all(dual_z .≈ dual_z_true)
    end
end
