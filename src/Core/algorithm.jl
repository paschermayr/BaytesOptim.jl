
############################################################################################
"""
$(TYPEDEF)

Default arguments for Optim constructor.

# Fields
$(TYPEDFIELDS)
"""
struct OptimDefault{T<:NamedTuple, G, I<:ModelWrappers.AbstractInitialization, U<:BaytesCore.UpdateBool}
    "Tuning Arguments for individual Optimizer"
    kernel::T
    "Gradient backend used in MCMC step. Not used if Metropolis sampler is chosen."
    GradientBackend::G
    "Method to obtain initial Modelparameter, see 'AbstractInitialization'."
    init::I
    "Boolean if generate(_rng, objective) for corresponding model is stored in MCMC Diagnostics."
    generated::U
    function OptimDefault(;
        kernel = (;),
        GradientBackend=:ForwardDiff,
        init=ModelWrappers.NoInitialization(),
        generated=BaytesCore.UpdateFalse()
    )
        ArgCheck.@argcheck (
            isa(GradientBackend, Symbol) || isa(GradientBackend, AnalyticalDiffTune)
        ) "GradientBackend keywords has to be either an AD symbol (:ForwardDiff, :ReverseDiff, :ReverseDiffUntaped, :Zyogte), or an AnalyticalDiffTune object."
        return new{
            typeof(kernel), typeof(GradientBackend), typeof(init), typeof(generated)
        }(
            kernel, GradientBackend, init, generated
        )
    end
end

############################################################################################
"""
$(TYPEDEF)

Stores information for proposal step.

# Fields
$(TYPEDFIELDS)
"""
struct Optimizer{M<:OptimKernel,N<:OptimTune} <: AbstractAlgorithm
    "MCMC sampler"
    kernel::M
    "Tuning configuration for kernel."
    tune::N
    function Optimizer(kernel::M, tune::N) where {M<:OptimKernel,N<:OptimTune}
        return new{M,N}(kernel, tune)
    end
end

function Optimizer(
    _rng::Random.AbstractRNG,
    kernel::Type{M},
    objective::Objective,
    default::OptimDefault=OptimDefault(),
    info::BaytesCore.SampleDefault = BaytesCore.SampleDefault()
) where {M<:OptimKernel}
    @unpack GradientBackend, generated = default
   ##	If a valid AD backend is provided, change it to an AutomaticDifftune Object
   if isa(GradientBackend, Symbol)
        GradientBackend = AutomaticDiffTune(objective, GradientBackend, BaytesDiff.DiffOrderOne())
    end
    ## Initiate Optimization Algorithm
    optim = init(kernel, objective, GradientBackend)
    ## Initial Optimization Tune struct
    kerneltune = init(AbstractConfiguration, kernel, objective; default.kernel...)
    optimtune = OptimTune(objective.tagged, kerneltune)
    ## Return MCMC container
    return Optimizer(optim, optimtune)
end


############################################################################################
"""
$(SIGNATURES)
Propose new parameter with optimizer. If update=true, objective function will be updated with input model and data.

# Examples
```julia
```

"""
function propose(_rng::Random.AbstractRNG, optim::Optimizer, objective::Objective)
    #!NOTE: Temperature is fixed for propose() step and will not be adjusted
    ## Make Proposal step
    θᵤᵖ, diagnostics = propagate(
        _rng, optim.kernel, optim.tune, objective
    )
    #Update Kernel and model parameter
    optim.kernel.θᵤ = θᵤᵖ
    ModelWrappers.unflatten_constrain!(objective.model, optim.tune.tagged, θᵤᵖ)
    return objective.model.val, diagnostics
end

############################################################################################
"""
$(SIGNATURES)
Inplace version of propose.

# Examples
```julia
```

"""
function propose!(
    _rng::Random.AbstractRNG,
    optim::Optimizer,
    model::ModelWrapper,
    data::D,
    proposaltune::T = BaytesCore.ProposalTune(model.info.reconstruct.default.output(1.0))
) where {D, T<:ProposalTune}
    ## Update Proposal tuning information that is shared among algorithms
    @unpack temperature, update = proposaltune
    ## Update Objective with new model parameter from other MCMC samplers and/or new/latent data
    objective = Objective(model, data, optim.tune.tagged, temperature)
    update!(optim.kernel, objective, update) #Update Kernel with current objective/configs
    ## Compute MCMC step
    val, diagnostics = propose(_rng, optim, objective)
    ## Update Model parameter
    model.val = val
    return val, diagnostics
end

############################################################################################
export 
    OptimDefault,
    Optimizer,
    propose,
    propose!