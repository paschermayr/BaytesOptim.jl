############################################################################################
# SGD specific diagnostics
"""
$(TYPEDEF)
Diagnostics for SGD.

# Fields
$(TYPEDFIELDS)
"""
struct DiagnosticsSGD <: OptimKernelDiagnostics
end

############################################################################################
"""
$(SIGNATURES)
Show relevant diagnostic results.

# Examples
```julia
```

"""
function generate_showvalues(diagnostics::D) where {D<:DiagnosticsSGD}
    return function showvalues()
        return ((:SGD, "diagnostics"), )
    end
end

############################################################################################
#export
export DiagnosticsSGD, generate_showvalues