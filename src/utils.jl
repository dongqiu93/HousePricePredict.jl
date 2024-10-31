function parse_or_nan(x)
    if isa(x, Number)
        return x
    elseif x == "NA"
        return NaN
    else
        return parse(Float64, x)
    end
end

function skipnan(x::Vector)
    return x[.!isnan.(x)]
end

function access_price(x)
    if x == "Grvl"
        return 0.1
    elseif x == "Paved"
        return 1
    else
        return 0
    end
end

function quality_of_material(x)
    if isa(x, Number)
        return x
    elseif x == "Ex"
        return 1
    elseif x == "Gd"
        return 0.75
    elseif x == "TA"
        return 0.5
    elseif x == "Fa"
        return 0.25
    elseif x == "Po"
        return 0.0
    elseif x == "NA"
        return 0.0
    else
        @error "Type ($x) is not defined."
    end
end

function exposure_bsmt(x)
    if isa(x, Number)
        return x
    elseif x == "Gd"
        return 1
    elseif x == "Av"
        return 0.66
    elseif x == "Mn"
        return 0.33
    elseif x == "No"
        return 0.0
    elseif x == "NA"
        return 0.0
    else
        @error "Type ($x) is not defined."
    end
end

function rate_bsmt(x)
    if isa(x, Number)
        return x
    elseif x == "GLQ"
        return 1
    elseif x == "ALQ"
        return 0.75
    elseif x == "BLQ"
        return 0.25
    elseif x == "Rec"
        return 0.5
    elseif x == "LwQ"
        return 0.1
    elseif x == "Unf"
        return 0.0
    elseif x == "NA"
        return 0.0
    else
        @error "Type ($x) is not defined."
    end
end

function home_functionality(x)
    if isa(x, Number)
        return x
    elseif x == "Typ"
        return 1
    elseif x == "Min1"
        return 0.9
    elseif x == "Min2"
        return 0.75
    elseif x == "Mod"
        return 0.6
    elseif x == "Maj1"
        return 0.45
    elseif x == "Maj2"
        return 0.3
    elseif x == "Sev"
        return 0.15
    elseif x == "Sal"
        return 0.0
    elseif x == "NA"
        return 0.0
    else
        @error "Type ($x) is not defined."
    end
end

function garage_finished(x)
    if isa(x, Number)
        return x
    elseif x == "Fin"
        return 1
    elseif x == "RFn"
        return 0.66
    elseif x == "Unf"
        return 0.33
    elseif x == "NA"
        return 0.0
    else
        @error "Type ($x) is not defined."
    end
end

function fence_quality(x)
    if isa(x, Number)
        return x
    elseif x == "GdPrv"
        return 1
    elseif x == "MnPrv"
        return 0.75
    elseif x == "GdWo"
        return 0.5
    elseif x == "MnWw"
        return 0.25
    elseif x == "NA"
        return 0.0
    else
        @error "Type ($x) is not defined."
    end
end

function paved_drive_score(x)
    if isa(x, Number)
        return x
    elseif x == "Y"
        return 1
    elseif x == "P"
        return 0.5
    elseif x == "N"
        return 0.0
    else
        @error "Type ($x) is not defined."
    end
end
