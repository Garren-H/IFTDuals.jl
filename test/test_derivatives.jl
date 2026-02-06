using StaticArrays
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
    θ_svec₁ = SVector{N}(θ₁);
    @testset "SVector Input: 1st order Duals" begin
        x₁_svec = get_x(θ_svec₁);
        @test x₁_svec ≈ x₁ₜ
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
        ift(y,f,x,xp)
    end
    y_true = value_derivative_and_second_derivative(h,AFD,x)
    y_ift = value_derivative_and_second_derivative(test_f,AFD,x)
    @test all(y_ift .≈ y_true)
end 
