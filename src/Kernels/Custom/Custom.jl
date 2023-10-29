###########################################################################################
# Note: Add this to BaytesOptim, so can be used within BayesSMC
###########################################################################################
"""
$(TYPEDEF)

Stores information used throughout custom algorithm.

# Fields
$(TYPEDFIELDS)
"""
struct CustomAlgorithmTune{
    T<:Tagged, 
    B<:BaytesCore.UpdateBool,
} <: AbstractTune
    "Tagged Parameter for Algorithm routine"
    tagged::T
    "Boolean if generated quantities should be generated while sampling"
    generated::B
    "Current iteration number"
    iter::Iterator
    function CustomAlgorithmTune(
        tagged::T, 
        generated::B
    ) where {T<:Tagged, B}
        #!NOTE: Start with 0, so first proposal step will update iter to 1
        iter = Iterator(0)
        return new{T, B}(
            tagged, generated, iter
        )
    end
end

"""
$(SIGNATURES)
Update tuning fields at current iteration.

# Examples
```julia
```

"""
function update!(
    tune::CustomAlgorithmTune, result::S
) where {S<:BaytesDiff.ℓObjectiveResult}
    ##  Update Current iteration counter
    update!(tune.iter)
    ## Pack container
    return nothing
end

############################################################################################
struct CustomAlgorithmDiagnostics{P, G, A} <: AbstractDiagnostics
    "Diagnostics used for all Baytes kernels"
    base::BaytesCore.BaseDiagnostics{P}
    "Generated quantities specified for objective"
    generated::G
    "Generated quantities specified for algorithm"
    generated_algorithm::A
    function CustomAlgorithmDiagnostics(
        base::BaytesCore.BaseDiagnostics{P},
        generated::G,
        generated_algorithm::A
    ) where {P, G, A}
        return new{P,G,A}(
            base, generated, generated_algorithm
        )
    end
end

function generate_showvalues(diagnostics::D) where {D<:CustomAlgorithmDiagnostics}
    return function showvalues()
        return (:custom, "diagnostics"),
        (:iter, diagnostics.base.iter),
        (:logobjective, diagnostics.base.ℓobjective),
        (:Temperature, diagnostics.base.temperature),
        (:generated, diagnostics.generated),
        (:generated_algorithm, diagnostics.generated_algorithm)
    end
end

############################################################################################
"""
$(TYPEDEF)

Default arguments for Custom constructor.

# Fields
$(TYPEDFIELDS)
"""
struct CustomAlgorithmDefault{I<:ModelWrappers.AbstractInitialization, U<:BaytesCore.UpdateBool}
    "Method to obtain initial Modelparameter, see 'AbstractInitialization'."
    init::I
    "Boolean if generate(_rng, objective) for corresponding model is stored in Algorithm Diagnostics."
    generated::U
    function CustomAlgorithmDefault(;
        init=ModelWrappers.NoInitialization(),
        generated=BaytesCore.UpdateFalse()
    )
        return new{
            typeof(init), typeof(generated)
        }(
            init, generated
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
mutable struct CustomAlgorithm{R<:ℓDensityResult, T<:CustomAlgorithmTune} <: AbstractAlgorithm
    result::R
    tune::T
    function CustomAlgorithm(result::R, tune::T) where {R<:ℓDensityResult, T<:CustomAlgorithmTune}
        return new{R, T}(result, tune)
    end
end

function CustomAlgorithm(
    _rng::Random.AbstractRNG,
    objective::Objective,
    default::CustomAlgorithmDefault=CustomAlgorithmDefault(),
    info::BaytesCore.SampleDefault = BaytesCore.SampleDefault()
)
    ## Obtain initial parameter ~ per default use current parameter
    default.init(_rng, nothing, objective)
    ## Initiate struct
    kerneltune = CustomAlgorithmTune(objective.tagged, default.generated)
    custom = CustomAlgorithm(BaytesDiff.ℓDensityResult(objective), kerneltune)
    ## Return container
    return custom
end

############################################################################################
"""
$(SIGNATURES)
Propose new parameter with Algorithm.

# Examples
```julia
```

"""
function propose(_rng::Random.AbstractRNG, algorithm::CustomAlgorithm, objective::Objective)
    #!NOTE: Temperature is fixed for propose() step and will not be adjusted

## This is step in all other kernels - but with propagate() manually overloaded, does it make more sense to adjust objective.model too without exposing user to LogObjectiveResult?
    ## Make Proposal step
    resultᵖ = propagate(
        _rng, algorithm, objective
    )
    algorithm.result = resultᵖ
    #Update model parameter
    ModelWrappers.unflatten_constrain!(objective.model, algorithm.tune.tagged, resultᵖ.θᵤ)
#=
    ## This is the other alternative - manually adjust propagate?
    propagate(
        _rng, algorithm, objective
        )
=#
    # Return Diagnostics
    diagnostics = CustomAlgorithmDiagnostics(
        BaytesCore.BaseDiagnostics(
            resultᵖ.ℓθᵤ,
#            0.0,
            objective.temperature,
            ModelWrappers.predict(_rng, algorithm, objective),
            algorithm.tune.iter.current
        ),
        ModelWrappers.generate(_rng, objective, algorithm.tune.generated),
        ModelWrappers.generate(_rng, algorithm, objective, algorithm.tune.generated)
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
    algorithm::CustomAlgorithm,
    model::ModelWrapper,
    data::D,
    proposaltune::T = BaytesCore.ProposalTune(model.info.reconstruct.default.output(1.0))
) where {D, T<:ProposalTune}
    ## Update Proposal tuning information that is shared among algorithms
    @unpack temperature, update = proposaltune
    ## Update Objective with new model parameter from other Kernels and/or new/latent data
    objective = Objective(model, data, algorithm.tune.tagged, temperature)
    ## Compute Optimization step
    val, diagnostics = propose(_rng, algorithm, objective)
    ## Update Model parameter
    model.val = val
    return val, diagnostics
end


############################################################################################
"""
$(SIGNATURES)
Function to dispatch on objective if needed to be extended. Note that objective.model has to be updated manually with estimated parameter in this step

# Examples
```julia
```

"""
function propagate(
    _rng::Random.AbstractRNG, algorithm::CustomAlgorithm, objective::Objective
)
    return BaytesDiff.ℓDensityResult(objective)
end

############################################################################################
"""
$(SIGNATURES)
Callable struct to make initializing Algorithm easier in sampling library.

# Examples
```julia
```

"""
struct CustomAlgorithmConstructor{S<:Union{Symbol,NTuple{k,Symbol} where k},D<:CustomAlgorithmDefault} <: AbstractConstructor
    "Parmeter to be tagged in Algorithm."
    sym::S
    "Algorithm Default Arguments"
    default::D
    function CustomAlgorithmConstructor(
        sym::S, default::D
    ) where {S<:Union{Symbol,NTuple{k,Symbol} where k},D<:CustomAlgorithmDefault}
        tup = BaytesCore.to_Tuple(sym)
        return new{typeof(tup),D}(tup, default)
    end
end
function (constructor::CustomAlgorithmConstructor)(
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    proposaltune::P,
    info::BaytesCore.SampleDefault
) where {D, P<:ProposalTune}
    return CustomAlgorithm(
        _rng,
        Objective(model, data, Tagged(model, constructor.sym), proposaltune.temperature),
        constructor.default,
        info
    )
end
function CustomAlgorithm(
    sym::S; kwargs...
) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return CustomAlgorithmConstructor(sym, CustomAlgorithmDefault(; kwargs...))
end

############################################################################################
#function infer(diagnostics::Type{AbstractDiagnostics}, kernel::OptimKernel)
#    return println("No known diagnostics for given kernel")
#end

"""
$(SIGNATURES)
Infer CustomAlgorithm diagnostics type.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    algorithm::CustomAlgorithm,
    model::ModelWrapper,
    data::D,
) where {D}
    TPrediction = infer(_rng, algorithm, model, data)
    TGenerated, TGenerated_algorithm = infer_generated(_rng, algorithm, model, data)
    return CustomAlgorithmDiagnostics{TPrediction, TGenerated, TGenerated_algorithm}
end

"""
$(SIGNATURES)
Infer type of predictions of CustomAlgorithm sampler.

# Examples
```julia
```

"""
function infer(_rng::Random.AbstractRNG, algorithm::CustomAlgorithm, model::ModelWrapper, data::D) where {D}
    objective = Objective(model, data, algorithm.tune.tagged)
    return typeof(predict(_rng, algorithm, objective))
end

"""
$(SIGNATURES)
Infer type of generated quantities of CustomAlgorithm.

# Examples
```julia
```

"""
function infer_generated(
    _rng::Random.AbstractRNG, algorithm::CustomAlgorithm, model::ModelWrapper, data::D
) where {D}
    objective = Objective(model, data, algorithm.tune.tagged)
    TGenerated = typeof(generate(_rng, objective, algorithm.tune.generated))
    TGenerated_algorithm = typeof(generate(_rng, algorithm, objective, algorithm.tune.generated))
    return TGenerated, TGenerated_algorithm
end

############################################################################################
"""
$(SIGNATURES)
Generate statistics for algorithm given model parameter and data.

# Examples
```julia
```

"""
function generate(_rng::Random.AbstractRNG, algorithm::CustomAlgorithm, objective::Objective)
    return nothing
end
function generate(_rng::Random.AbstractRNG, algorithm::CustomAlgorithm, objective::Objective, gen::BaytesCore.UpdateTrue)
    return generate(_rng, algorithm, objective)
end
function generate(_rng::Random.AbstractRNG, algorithm::CustomAlgorithm, objective::Objective, gen::BaytesCore.UpdateFalse)
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Print result for a single trace.

# Examples
```julia
```

"""
function results(
    diagnosticsᵛ::AbstractVector{M}, algorithm::CustomAlgorithm, Ndigits::Integer, quantiles::Vector{T}
) where {T<:Real,M<:CustomAlgorithmDiagnostics}
    return nothing
end

############################################################################################
function result!(algorithm::CustomAlgorithm, result::L) where {L<:BaytesDiff.ℓObjectiveResult}
    algorithm.result = result
    return nothing
end

function get_result(algorithm::CustomAlgorithm)
    return algorithm.result
end

function predict(_rng::Random.AbstractRNG, algorithm::CustomAlgorithm, objective::Objective)
    return predict(_rng, objective)
end

############################################################################################
export 
    CustomAlgorithmTune,
    CustomAlgorithmDiagnostics,
    CustomAlgorithmDefault,
    CustomAlgorithm,
    CustomAlgorithmConstructor,
    propose,
    propose!,
    propagate

