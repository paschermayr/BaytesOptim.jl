############################################################################################
"""
$(TYPEDEF)
Default Configuration for SGD optimizer.

# Fields
$(TYPEDFIELDS)
"""
struct ConfigSGD{R<:Real, A<:BaytesCore.UpdateBool} <: AbstractConfiguration
    """
    Add `-0.5 * magnitude_penalty * sum(abs2, q)` to the log posterior **when finding the local
    optimum**. This can help avoid getting into high-density edge areas of the posterior
    which are otherwise not typical (eg multilevel models).
    """
    magnitude_penalty::R
    """
    Adapt magnitude iteratively for each step ~ currently not implemented
    """
    magnitude_adaption::A
    """
    Maximum number of iterations in the optimization algorithm. Recall that we don't need to
    find the mode, or even a local mode, just be in a reasonable region.
    """
    iterations::Int64
    "Differentiable order for objective function needed to run propagate step"
    difforder::BaytesDiff.DiffOrderOne
    function ConfigSGD(
        magnitude_penalty::R,
        magnitude_adaption::A,
        iterations::Int64,
        difforder::BaytesDiff.DiffOrderOne
        ) where {R<:Real, A<:BaytesCore.UpdateBool}
        @argcheck magnitude_penalty >= 0.0 "Magnitude penalty needs to be positive"
        @argcheck iterations >= 0 "Minimum of 1 iteration"
        return new{R, A}(
            magnitude_penalty, magnitude_adaption, iterations, difforder
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
    optim::Type{SGD},
    objective::Objective ;
    magnitude_penalty = 1e-2,
    magnitude_adaption = BaytesCore.UpdateFalse(),
    iterations = 1
)
    return ConfigSGD(
        magnitude_penalty,
        magnitude_adaption, 
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
struct SGDTune{R<:Real, A<:BaytesCore.UpdateBool} <: AbstractTune
    """
    Add `-0.5 * magnitude_penalty * sum(abs2, q)` to the log posterior **when finding the local
    optimum**. This can help avoid getting into high-density edge areas of the posterior
    which are otherwise not typical (eg multilevel models).
    """
    magnitude_penalty::R
    """
    Adapt magnitude iteratively for each step ~ currently not implemented
    """
    magnitude_adaption::A
    """
    Maximum number of iterations in the optimization algorithm. Recall that we don't need to
    find the mode, or even a local mode, just be in a reasonable region.
    """
    iterations::Int64
    function SGDTune(;
        magnitude_penalty::R = 1e-2,
        magnitude_adaption::A =  BaytesCore.UpdateFalse(),
        iterations = 1
        ) where {R<:Real, A<:BaytesCore.UpdateBool}
        @argcheck magnitude_penalty >= 0.0 "Magnitude penalty needs to be positive"
        @argcheck iterations >= 0 "Minimum of 1 iteration"
        return new{R,A}(
            magnitude_penalty, magnitude_adaption, iterations
        )
    end
end

function update!(tune::SGDTune, result::ℓObjectiveResult)
    return nothing
end


"""
$(SIGNATURES)
Initialize SGD custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractTune},
    config::ConfigSGD,
    objective::Objective;
)
    return SGDTune(;
        magnitude_penalty = config.magnitude_penalty,
        magnitude_adaption = config.magnitude_adaption,
        iterations = config.iterations
    )
end

############################################################################################
#export
export ConfigSGD, SGDTune
