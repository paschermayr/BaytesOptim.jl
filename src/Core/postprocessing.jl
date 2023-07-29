############################################################################################
"""
$(SIGNATURES)
Callable struct to make initializing Optimizer easier in sampling library.

# Examples
```julia
```

"""
struct OptimConstructor{M,S<:Union{Symbol,NTuple{k,Symbol} where k},D<:OptimDefault} <: AbstractConstructor
    "Valid Optim kernel."
    kernel::M
    "Parmeter to be tagged in Optimization."
    sym::S
    "Optim Default Arguments"
    default::D
    function OptimConstructor(
        kernel::Type{M}, sym::S, default::D
    ) where {M<:OptimKernel,S<:Union{Symbol,NTuple{k,Symbol} where k},D<:OptimDefault}
        tup = BaytesCore.to_Tuple(sym)
        return new{typeof(kernel),typeof(tup),D}(kernel, tup, default)
    end
end
function (constructor::OptimConstructor)(
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    proposaltune::P,
    info::BaytesCore.SampleDefault
) where {D, P<:ProposalTune}
    return Optimizer(
        _rng,
        constructor.kernel,
        Objective(model, data, Tagged(model, constructor.sym), proposaltune.temperature),
        constructor.default,
        info
    )
end
function Optimizer(
    kernel::Type{M}, sym::S; kwargs...
) where {M<:OptimKernel,S<:Union{Symbol,NTuple{k,Symbol} where k}}
    return OptimConstructor(kernel, sym, OptimDefault(; kwargs...))
end

############################################################################################
function infer(diagnostics::Type{AbstractDiagnostics}, kernel::OptimKernel)
    return println("No known diagnostics for given kernel")
end

"""
$(SIGNATURES)
Infer Optim diagnostics type.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    optimizer::Optimizer,
    model::ModelWrapper,
    data::D,
) where {D}
    TKernel = infer(_rng, diagnostics, optimizer.kernel, model, data)
    TPrediction = infer(_rng, optimizer, model, data)
    TGenerated, TGenerated_algorithm = infer_generated(_rng, optimizer, model, data)
    return OptimDiagnostics{TPrediction,TKernel,TGenerated, TGenerated_algorithm}
end

"""
$(SIGNATURES)
Infer type of predictions of Optim sampler.

# Examples
```julia
```

"""
function infer(_rng::Random.AbstractRNG, optimizer::Optimizer, model::ModelWrapper, data::D) where {D}
    objective = Objective(model, data, optimizer.tune.tagged)
    return typeof(predict(_rng, optimizer, objective))
end

"""
$(SIGNATURES)
Infer type of generated quantities of Optimizer.

# Examples
```julia
```

"""
function infer_generated(
    _rng::Random.AbstractRNG, optimizer::Optimizer, model::ModelWrapper, data::D
) where {D}
    objective = Objective(model, data, optimizer.tune.tagged)
    TGenerated = typeof(generate(_rng, objective, optimizer.tune.generated))
    TGenerated_algorithm = typeof(generate(_rng, optimizer, objective, optimizer.tune.generated))
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
function generate(_rng::Random.AbstractRNG, algorithm::Optimizer, objective::Objective)
    return nothing
end
function generate(_rng::Random.AbstractRNG, algorithm::Optimizer, objective::Objective, gen::BaytesCore.UpdateTrue)
    return generate(_rng, algorithm, objective)
end
function generate(_rng::Random.AbstractRNG, algorithm::Optimizer, objective::Objective, gen::BaytesCore.UpdateFalse)
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
    diagnosticsᵛ::AbstractVector{M}, algorithm::Optimizer, Ndigits::Integer, quantiles::Vector{T}
) where {T<:Real,M<:OptimDiagnostics}
    return nothing
end

############################################################################################
function result!(algorithm::Optimizer, result::L) where {L<:ℓObjectiveResult}
    algorithm.kernel.result = result
    return nothing
end

function get_result(algorithm::Optimizer)
    return algorithm.kernel.result
end

function predict(_rng::Random.AbstractRNG, algorithm::Optimizer, objective::Objective)
    return predict(_rng, objective)
end

############################################################################################
# Export
export OptimConstructor, infer, infer_generated, predict
