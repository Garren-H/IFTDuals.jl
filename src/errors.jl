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

struct PartialsSeedDimensionMismatchError <: Exception
    msg::String
end
PartialsSeedDimensionMismatchError(N::Int, idx::Int, s::Tuple, t::Symbol) = PartialsSeedDimensionMismatchError("Dimensions of $t and seed array at index $idx do not match. Got size of $t: $(s[idx]) and size of seed array: ("*join([i == idx ? "\e[1;91m$(s[i])\e[0m" : "$(s[i])" for i in 1:length(s)], ", ")*").")
PartialsSeedDimensionMismatchError(idx::Int, s::Tuple) = PartialsSeedDimensionMismatchError("Dimensions of seed array is incompatable with the dimensions of the attempted assignment of the dual. Seed array has ndims=\e[1;91m$(length(s))\e[0m but attempted assignment tries to access seed array at dim \e[1;91m$idx\e[0m.")
Base.showerror(io::IO, e::PartialsSeedDimensionMismatchError) = print(io, "PartialsSeedDimensionMismatchError: ", e.msg)
check_partials_seed_dims(N::Int, idx::Tuple, x::AbstractArray, t::Symbol) = begin
    idx_ = length(idx) + 1
    if ndims(x) < idx_ # incompatable dimensions of seed array
        throw(PartialsSeedDimensionMismatchError(idx_, size(x)))
    elseif N !== size(x, idx_) # Partials and dims of seed array do not match 
        throw(PartialsSeedDimensionMismatchError(N, idx_, size(x), t))
    end
end
