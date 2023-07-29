############################################################################################
"""
$(TYPEDEF)
Default Configuration for LBFG optimizer.

# Fields
$(TYPEDFIELDS)
"""
struct ConfigLBFG{R<:Real} <: AbstractConfiguration
    """
    Add `-0.5 * magnitude_penalty * sum(abs2, q)` to the log posterior **when finding the local
    optimum**. This can help avoid getting into high-density edge areas of the posterior
    which are otherwise not typical (eg multilevel models).
    """
    magnitude_penalty::R
    """
    Maximum number of iterations in the optimization algorithm. Recall that we don't need to
    find the mode, or even a local mode, just be in a reasonable region.
    """
    iterations::Int64
    "Differentiable order for objective function needed to run propagate step"
    difforder::BaytesDiff.DiffOrderOne
    function ConfigLBFG(
        magnitude_penalty::R,
        iterations::Int64,
        difforder::BaytesDiff.DiffOrderOne
        ) where {R<:Real}
        @argcheck magnitude_penalty >= 0.0 "Magnitude penalty needs to be positive"
        @argcheck iterations >= 0 "Minimum of 1 iteration"
        return new{R}(
            magnitude_penalty, iterations, difforder
        )
    end
end

"""
$(SIGNATURES)
Initialize Mala custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractConfiguration},
    optim::Type{OptimLBFG},
    objective::Objective ;
    magnitude_penalty = 1e-4,
    iterations = 20
)
    return ConfigLBFG(
        magnitude_penalty,
        iterations, 
        BaytesDiff.DiffOrderOne()
    )
end


############################################################################################
"""
$(TYPEDEF)

Stores information used throughout optimization algorithms.

# Fields
$(TYPEDFIELDS)
"""
struct LBFGTune{R<:Real} <: AbstractTune
    """
    Add `-0.5 * magnitude_penalty * sum(abs2, q)` to the log posterior **when finding the local
    optimum**. This can help avoid getting into high-density edge areas of the posterior
    which are otherwise not typical (eg multilevel models).
    """
    magnitude_penalty::R
    """
    Maximum number of iterations in the optimization algorithm. Recall that we don't need to
    find the mode, or even a local mode, just be in a reasonable region.
    """
    iterations::Int64
    function LBFGTune(;
        magnitude_penalty::R = 1e-4,
        iterations = 20
        ) where {R<:Real}
        @argcheck magnitude_penalty >= 0.0 "Magnitude penalty needs to be positive"
        @argcheck iterations >= 0 "Minimum of 1 iteration"
        return new{R}(
            magnitude_penalty, iterations
        )
    end
end

function update!(tune::LBFGTune, result::ℓObjectiveResult)
    return nothing
end


"""
$(SIGNATURES)
Initialize LBFG custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractTune},
    config::ConfigLBFG,
    objective::Objective;
)
    return LBFGTune(;
        magnitude_penalty = config.magnitude_penalty,
        iterations = config.iterations
    )
end

############################################################################################
#export
export ConfigLBFG, LBFGTune
