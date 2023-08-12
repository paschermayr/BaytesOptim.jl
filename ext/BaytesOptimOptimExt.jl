module BaytesOptimOptimExt

############################################################################################
# Optim Extension
using BaytesOptim, BaytesDiff, ModelWrappers
import BaytesOptim: propagate
using Random, SimpleUnPack, DocStringExtensions

using NLSolversBase, Optim

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
export propagate


end