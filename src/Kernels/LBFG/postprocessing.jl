############################################################################################
function init(
    ::Type{OptimLBFG},
    config::ConfigLBFG,
    objective::Objective,
    difftune::AbstractDifferentiableTune,
)
    return OptimLBFG(
        BaytesDiff.log_density_and_gradient(objective, difftune),
        difftune
    )
end

function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::OptimLBFG,
    model::ModelWrapper,
    data::D,
) where {D}
    return DiagnosticsLBFG
end

############################################################################################
# Export
export init, infer
