module IFTDuals

import ForwardDiff: Tag, Dual, Partials, partials, value, order, tagtype, valtype, npartials
import LinearAlgebra: lu!, LU, checksquare
import DifferentiationInterface: derivative, jacobian, AutoForwardDiff, Constant

AFD = AutoForwardDiff() # Alias for AutoForwardDiff

include("errors.jl")
include("checks.jl")
include("utils.jl")
include("derivatives.jl")
end
