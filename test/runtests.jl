using IFTDuals,DifferentiationInterface,Test
import ForwardDiff
import ForwardDiff: Dual, Tag, Partials
const AFD = AutoForwardDiff()

include("test_derivatives.jl")
include("test_utils.jl")
