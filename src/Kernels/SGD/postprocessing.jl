############################################################################################
function init(
    ::Type{SGD},
    config::ConfigSGD,
    objective::Objective,
    difftune::AbstractDifferentiableTune,
)
    return SGD(
        BaytesDiff.log_density_and_gradient(objective, difftune),
        difftune
    )
end

function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::SGD,
    model::ModelWrapper,
    data::D,
) where {D}
    return DiagnosticsSGD
end

############################################################################################
# Export
export init, infer
