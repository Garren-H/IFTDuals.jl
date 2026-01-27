module IFTDuals

import ForwardDiff: Tag, Dual, Partials, partials, value, order, tagtype, valtype, npartials
import LinearAlgebra: lu, LU, checksquare, ldiv!
import DifferentiationInterface: derivative, jacobian, AutoForwardDiff, Constant
import StaticArrays: SVector
const AFD = AutoForwardDiff() # Alias for AutoForwardDiff

include("errors.jl")
include("checks.jl")
include("utils.jl")
include("derivatives.jl")
include("primalarray.jl")

end
