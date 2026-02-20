@testset "Issue #5: Ensure StaticArrays does not error" begin
    using IFTDuals,ForwardDiff,StaticArrays
    function quadratic_solver(poly::NTuple{3,T}) where T
        poly_primal = IFTDuals.nested_pvalue(poly)
        x = quadratic_solver_inner(poly_primal)
        if T <: ForwardDiff.Dual
            return quadratic_solver_ad(x,poly,poly_primal)
        else
            return x
        end
    end

    function quadratic_solver_inner(poly)
        c,b,a = poly
        delta = b*b - 4*a*c*c
        x1 = (-b + delta)/(2a)
        x2 = (-b - delta)/(2a)
        return SVector(x1,x2)
    end

    function quadratic_solver_ad(x,tups,tups_primal)
        function f(xx,tup)
            c,b,a = tup
            x1,x2 = xx
            #vieta formulas for the quadratic
            SVector(x1*x2 - c/a,x1 + x2 + b/a)
        end
        return ift(x,f,tups,tups_primal)
    end

    issue_05(x) = quadratic_solver((x,2x-1,x*x))[1]
    @test begin 
        ForwardDiff.derivative(issue_05,1.0)
        nothing
    end === nothing # just test no error are thrown
end

