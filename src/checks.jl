function check_multiple_duals(et,N,T) # Number of partials and TagType are shared across nested levels
    N === npartials(et) || throw(MultiTagError())
    T === tagtype(et) || throw(MultiTagError())
    et = valtype(et)
    return et
end

function check_multiple_duals_and_return_order(x)
    common_et = get_common_dual_type(x) # get common supertype, should be Dual
    common_et <: Dual || return 0,common_et# if not Dual, we are done
    order_ = order(common_et)
    et = common_et
    N = npartials(et)
    T = tagtype(et)
    lenT = length(T.parameters)
    while et <: Dual
        et = check_multiple_duals(et,N,T)
        T = get_tag_val(T)
    end
    return order_,common_et
end

get_tag_val(::Type{Tag{F,V}}) where {F,V<:Dual} = tagtype(V) # get inner tag type
get_tag_val(T::Type{Tag{F,V}}) where {F,V} = T # return as is
