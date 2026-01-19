flatten_tuple(x) = (x,)
flatten_tuple(x::Tuple) = mapreduce(flatten_tuple, (a, b) -> (a..., b...), x)

function check_multiple_duals(et,N,T) # Number of partials and TagType are shared across nested levels
    N === npartials(et) || throw(MultiTagError())
    T === tagtype(et) || throw(MultiTagError())
    et = valtype(et)
    return et
end

function check_multiple_duals_and_return_order(x)
    common_et = get_common_dual_type(x) # get common supertype, should be Dual
    order_ = order(common_et)
    order_ == 0 && return order_,common_et# if not Dual, we are done
    et = common_et
    N = npartials(et)
    T = tagtype(et)
    while et <: Dual
        et = check_multiple_duals(et,N,T)
    end
    return order_,common_et
end

