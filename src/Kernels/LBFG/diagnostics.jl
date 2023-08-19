############################################################################################
# LBFG specific diagnostics
"""
$(TYPEDEF)
Diagnostics for LBFG.

# Fields
$(TYPEDFIELDS)
"""
struct DiagnosticsLBFG <: OptimKernelDiagnostics
end

############################################################################################
"""
$(SIGNATURES)
Show relevant diagnostic results.

# Examples
```julia
```

"""
function generate_showvalues(diagnostics::D) where {D<:DiagnosticsLBFG}
    return function showvalues()
        return ((:LBFG, "diagnostics"), )
    end
end

############################################################################################
#export
export DiagnosticsLBFG, generate_showvalues