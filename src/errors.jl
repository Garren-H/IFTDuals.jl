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