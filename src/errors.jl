# Errors
struct RealTypeError <: Exception 
    msg::String
end
RealTypeError() = RealTypeError("The input contains elements of Real type, cannot extract eltypes uniquely.")
Base.showerror(io::IO, e::RealTypeError) = print(io, "RealTypeError: ", e.msg)

struct AnyTypeError <: Exception 
    msg::String
end
AnyTypeError() = AnyTypeError("The input contains elements of Any type, cannot extract eltypes uniquely.")
Base.showerror(io::IO, e::AnyTypeError) = print(io, "AnyTypeError: ", e.msg)

struct MultiTagError <: Exception 
    msg::String
end
MultiTagError() = MultiTagError("The input Duals contains different function signatures in the Tag type and/or different number of partials at certain levels. This indicates nested differentiation wrt to different variables/functions. This is currently not supported. You may consider concatenating the inputs into a single vector and differentiating jointly.")
Base.showerror(io::IO, e::MultiTagError) = print(io, "MultiTagError: ", e.msg)


