############################################################################################
mutable struct SGD{M<:ℓGradientResult, D<:AbstractDifferentiableTune} <: OptimKernel
    "Cached Result of last propagation step."
    result::M
    "Differentiation tuning container"
    diff::D
    function SGD(
        result::M, 
        diff::D
    ) where {
        M<:ℓObjectiveResult,
        D<:AbstractDifferentiableTune
        }
        return new{M,D}(result, diff)
    end
end

function update!(kernel::SGD, objective::Objective, up::BaytesCore.UpdateTrue)
    ## Update log-target result with current (latent) data
    kernel.diff = update(kernel.diff, objective)
    #!NOTE: Cannot use first gradient result with external library, so just compute log density and go from there
    kernel.result = BaytesDiff.log_density_and_gradient(objective, kernel.diff)
    BaytesDiff.checkfinite(objective, kernel.result)
    return nothing
end
function update!(kernel::SGD, objective::Objective, up::BaytesCore.UpdateFalse)
    return nothing
end

function init(
    ::Type{SGD},
    objective::Objective,
    difftune::AbstractDifferentiableTune,
)
    return SGD(
        BaytesDiff.log_density_and_gradient(objective, difftune),
        difftune
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
    _rng::Random.AbstractRNG, kernel::SGD, tune::OptimTune, objective::Objective
)
    @unpack magnitude_penalty, iterations = tune.kernel
## Assign Initial Variables
    resultᵖ = kernel.result
    θᵤᵖ = resultᵖ.θᵤ
## Loop through iterations
    for iter in Base.OneTo(iterations)
        # Move parameter ~ initial ∇ℓθᵤ already provided by kernel update or by last iteration
        θᵤᵖ .+ (magnitude_penalty/iter)*resultᵖ.∇ℓθᵤ
        # Compute Gradient of current parameter
        resultᵖ = BaytesDiff.log_density_and_gradient(objective, kernel.diff, θᵤᵖ)
    end
## Store Diagnostics
    diagnostics = DiagnosticsSGD()
## Pack and return output
    return resultᵖ, diagnostics
end

############################################################################################
export 
    SGD,
    propagate
