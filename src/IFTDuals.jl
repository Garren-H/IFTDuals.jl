module IFTDuals

import ForwardDiff: Tag, Dual, Partials, partials, value, order, tagtype, valtype, npartials
import LinearAlgebra: lu!, lu, LU, checksquare, ldiv!, ldiv
import DifferentiationInterface: derivative, jacobian, AutoForwardDiff, Constant

const AFD = AutoForwardDiff() # Alias for AutoForwardDiff
const ScalarOrAbstractVec{T} = Union{T, AbstractVector{T}}
const ScalarOrAbstractVecOrMat{T} = Union{T, AbstractVecOrMat{T}}
const IDX = ScalarOrAbstractVec{<:Union{Int,CartesianIndex{1}}} # Alias for valid index types for 1D arrays (scalars and vectors)

include("errors.jl")
include("checks.jl")
include("utils.jl")
include("derivatives.jl")
include("abstractarrays.jl")

end
