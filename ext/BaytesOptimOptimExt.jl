module BaytesOptimOptimExt

############################################################################################
#using BaytesOptim
import BaytesOptim: BaytesOptim, _optimize
using NLSolversBase, Optim

function _optimize(θᵤ::AbstractVector, fg!, iterations::Integer)
    
    optim_objective = NLSolversBase.OnceDifferentiable(NLSolversBase.only_fg!(fg!), θᵤ)
    opt = Optim.optimize(optim_objective, θᵤ, Optim.LBFGS(),
                        Optim.Options(; iterations = iterations))

    θᵤᵖ = Optim.minimizer(opt)
    # Return proposed values and diagnostics
    return θᵤᵖ, nothing
end

end