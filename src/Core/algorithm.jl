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
    "Gradient backend used in Optimization step. Not used if Metropolis sampler is chosen."
    GradientBackend::G
    "Method to obtain initial Modelparameter, see 'AbstractInitialization'."
    init::I
    "Boolean if generate(_rng, objective) for corresponding model is stored in Optimization Diagnostics."
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
    "Optimizer"
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

    ## Initiate Optimization Algorithm and Algorithm-specific tuning struct
    optimconfig = init(AbstractConfiguration, kernel, objective; default.kernel...)
    ##	If a valid AD backend is provided, change it to an AutomaticDifftune Object
   if isa(GradientBackend, Symbol)
        GradientBackend = AutomaticDiffTune(objective, GradientBackend, optimconfig.difforder)
    end
    ## Initiate Optimization Algorithm and Algorithm-specific tuning struct
    optimconfig = init(AbstractConfiguration, kernel, objective; default.kernel...)
    kerneltune = init(AbstractTune, optimconfig, objective)
    optim = init(kernel, objective, GradientBackend)
    ## Initial General optimization tune struct
    optimtune = OptimTune(objective.tagged, kerneltune, generated)
    ## Return Optim container
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
    resultᵖ, kernel_diagnostics = propagate(
        _rng, optim.kernel, optim.tune, objective
    )
    #Update Kernel and model parameter
    optim.kernel.result = resultᵖ
    ModelWrappers.unflatten_constrain!(objective.model, optim.tune.tagged, resultᵖ.θᵤ)
    # Return Diagnostics
    diagnostics = OptimDiagnostics(
        BaytesCore.BaseDiagnostics(
            optim.kernel.result.ℓθᵤ,
            objective.temperature,
            ModelWrappers.predict(_rng, optim, objective),
            optim.tune.iter.current
        ),
        kernel_diagnostics,
        ModelWrappers.generate(_rng, objective, optim.tune.generated),
        ModelWrappers.generate(_rng, optim, objective, optim.tune.generated)
    )

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
    ## Update Objective with new model parameter from other Optimizer and/or new/latent data
    objective = Objective(model, data, optim.tune.tagged, temperature)
    update!(optim.kernel, objective, update) #Update Kernel with current objective/configs
    ## Compute Optimization step
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