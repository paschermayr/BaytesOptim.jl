############################################################################################
mutable struct OptimLBFG{M<:ℓDensityResult, D<:AbstractDifferentiableTune} <: OptimKernel
    "Cached Result of last propagation step."
    result::M
    "Differentiation tuning container"
    diff::D
    function OptimLBFG(
        result::M, 
        diff::D
    ) where {
        M<:ℓObjectiveResult,
        D<:AbstractDifferentiableTune
        }
        return new{M,D}(result, diff)
    end
end

function update!(kernel::OptimLBFG, objective::Objective, up::BaytesCore.UpdateTrue)
    ## Update log-target result with current (latent) data
    kernel.diff = update(kernel.diff, objective)
    #!NOTE: Cannot use first gradient result with external library, so just compute log density and go from there
    kernel.result = BaytesDiff.ℓDensityResult(objective)
    BaytesDiff.checkfinite(objective, kernel.result)
    return nothing
end
function update!(kernel::OptimLBFG, objective::Objective, up::BaytesCore.UpdateFalse)
    return nothing
end

function init(
    ::Type{OptimLBFG},
    objective::Objective,
    difftune::AbstractDifferentiableTune,
)
    return OptimLBFG(
        BaytesDiff.ℓDensityResult(objective),
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
    optim_objective = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!(fg!), kernel.result.θᵤ)
    opt = Optim.optimize(optim_objective, kernel.result.θᵤ, Optim.LBFGS(),
                        Optim.Options(; iterations = iterations))
    θᵤᵖ = Optim.minimizer(opt)
    # Store Diagnostics
    diagnostics = DiagnosticsLBFG()
    ## Pack and return output
    return BaytesDiff.ℓDensityResult(objective, θᵤᵖ), diagnostics
end

############################################################################################
export 
    OptimLBFG,
    propagate
