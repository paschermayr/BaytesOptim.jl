############################################################################################
mutable struct OptimLBFG{M<:AbstractArray, D<:AbstractDifferentiableTune} <: OptimKernel
    "Cached Result of last propagation step."
    θᵤ::M
    "Differentiation tuning container"
    diff::D
    function OptimLBFG(
        θᵤ::M, 
        diff::D
    ) where {
        M<:AbstractArray,
        D<:AbstractDifferentiableTune
        }
        return new{M,D}(θᵤ, diff)
    end
end

function init(
    ::Type{OptimLBFG},
    objective::Objective,
    difftune::AbstractDifferentiableTune,
)
    return OptimLBFG(
        ModelWrappers.unconstrain_flatten(objective.model, objective.tagged),
        difftune
    )
end

function update!(kernel::OptimLBFG, objective::Objective, up::BaytesCore.UpdateTrue)
    ## Update log-target result with current (latent) data
    kernel.diff = update(kernel.diff, objective)
    result = BaytesDiff.log_density_and_gradient(objective, kernel.diff)
    BaytesDiff.checkfinite(objective, result)
    kernel.θᵤ = result.θᵤ
    return nothing
end
function update!(kernel::OptimLBFG, objective::Objective, up::BaytesCore.UpdateFalse)
    return nothing
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

"""
$(SIGNATURES)
Initialize HMC custom configurations.

# Examples
```julia
```

"""
function init(
    ::Type{AbstractConfiguration},
    optim::Type{OptimLBFG},
    objective::Objective;
    magnitude_penalty = 1e-4,
    iterations = 20
)
    return LBFGTune(;
        magnitude_penalty = magnitude_penalty,
        iterations = iterations
    )
end

############################################################################################
"""
$(SIGNATURES)
Propagate forward one MALA step.

# Examples
```julia
```

"""
function propagate(
    _rng::Random.AbstractRNG, kernel::OptimLBFG, tune::OptimTune, objective::Objective
)
    @unpack magnitude_penalty, iterations = tune.kernel
    fg! = function(F, G, θᵤ)
        # NOTE: Optim.optimize *minimizes*, so we *add* a penalty
        result = log_density_and_gradient(objective, kernel.diff, θᵤ)
        if G ≠ nothing
            @. G = -result.∇ℓθᵤ + θᵤ * magnitude_penalty
        end
        -result.ℓθᵤ + (0.5 * magnitude_penalty * sum(abs2, θᵤ))
    end
    optim_objective = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!(fg!), kernel.θᵤ)
    opt = Optim.optimize(optim_objective, kernel.θᵤ, Optim.LBFGS(),
                        Optim.Options(; iterations = iterations))
    θᵤᵖ = Optim.minimizer(opt)
    # Store Diagnostics
    diagnostics = OptimDiagnostics()
    ## Pack and return output
    return θᵤᵖ, diagnostics
end

############################################################################################
export 
    OptimLBFG,
    LBFGTune,
    propagate
