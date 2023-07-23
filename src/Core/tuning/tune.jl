############################################################################################
"""
$(TYPEDEF)

Stores information used throughout optimization algorithms.

# Fields
$(TYPEDFIELDS)
"""
struct OptimTune{T<:Tagged, K} <: AbstractTune
    "Tagged Parameter for Optimization routine"
    tagged::T
    "Tuning arguments for individual Optimizer"
    kernel::K
    function OptimTune(
        tagged::T, 
        kernel::K
    ) where {T<:Tagged, K}
        return new{T, K}(
            tagged, kernel
        )
    end
end

################################################################################
export OptimTune