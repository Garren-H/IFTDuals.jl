function check_mixed_tags_and_return_order(x)
    et = get_common_dual_type(x) # get common supertype, should be Dual
    et <: Dual || return 0,et,false# if not Dual, we are done
    der_order = order(et)
    tag_is_mixed = der_order == 1 ? false : check_if_mixed_tag(et)
    return der_order,et,tag_is_mixed
end

function check_if_mixed_tag(et::Type{<:Dual})
    N = npartials(et)
    Tf = tagtype(et).parameters[1] # function signature in tag type
    tag_is_mixed = false
    while et <: Dual && !tag_is_mixed
        N === npartials(et) || (tag_is_mixed = true; break)
        Tf === tagtype(et).parameters[1] || (tag_is_mixed = true; break) # Only check if function signature in tag types differ. Second argument is value type, which may differ in nested duals
        et = valtype(et)
    end
    return tag_is_mixed
end
