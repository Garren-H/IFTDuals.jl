module IFTDuals

import ForwardDiff: Tag, Dual, Partials, partials, value, order, tagtype, valtype, npartials
import LinearAlgebra: lu, LU, checksquare, ldiv!, ldiv
import DifferentiationInterface: derivative, jacobian, AutoForwardDiff, Constant
import StaticArrays: SVector
const AFD = AutoForwardDiff() # Alias for AutoForwardDiff

const ScalarOrAbstractVec{T} = Union{T, AbstractVector{T}}
const ScalarOrAbstractVecOrMat{T} = Union{T, AbstractVecOrMat{T}}

include("errors.jl")
include("checks.jl")
include("utils.jl")
include("derivatives.jl")
include("primalarray.jl")

end
