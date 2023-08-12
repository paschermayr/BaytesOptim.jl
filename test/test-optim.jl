
############################################################################################
# Check if correct result is reached from random starting point
model = ModelWrapper(MultiNormal(), param)
data = simulate(_RNG, model)
objective = Objective(model, data)
θ = deepcopy(model.val)
objective(θ)

# Sample from prior
model_initial = deepcopy(model)
sample!(model_initial)
model_initial.val
objective_inital = Objective(model_initial, data)

# Create optimizer
def = def = OptimDefault(; 
    kernel = (;
        magnitude_penalty = 1e-4,
        iterations = 1000
    )
)
opt = Optimizer(
    _RNG,
    OptimLBFG,
    objective_inital,
    def,
) 

objective_inital(objective_inital.model.val)
θᵤ_proposed, diag = propagate(
    _RNG, opt.kernel, opt.tune, objective_inital
)

#println("Proposed")
θ_proposed = unflatten_constrain(objective.model, objective.tagged, θᵤ_proposed.θᵤ)
θ_proposed.ρ
#println("True")
model.val
model.val.ρ
objective_inital.model.val.ρ

propose(
    _RNG, opt, objective_inital,
) 
objective_inital.model.val#.ρ

_vals, diag = propose!(
    _RNG, opt, objective_inital.model, objective_inital.data
) 
objective_inital.model.val.ρ

############################################################################################
# Now only update parts of Objective function
model2 = ModelWrapper(MultiNormal(), param)
model2.val
objective_inital2 = Objective(deepcopy(model2), data, Tagged(deepcopy(model2), :μ));
objective_inital2.model.val
def = def = OptimDefault(; 
    kernel = (;
        magnitude_penalty = 1e-4,
        iterations = 1000
    )
)
opt = Optimizer(
    _RNG,
    OptimLBFG,
    objective_inital2,
    def,
) 
_vals, diag = propose!(
    _RNG, opt, objective_inital2.model, objective_inital2.data
)
objective_inital2.model.val

##########################################################################################
# Test Constructor
_oc = OptimConstructor(OptimLBFG, :μ, 
    OptimDefault(; 
        kernel = (;
            iterations = 123)
    ) 
)
using BaytesCore
_opt = _oc(_RNG, objective.model, objective.data, BaytesCore.ProposalTune(1.), BaytesCore.SampleDefault())
Optimizer(OptimLBFG, :μ)

#?
# infer(OptimDiagnostics, opt.kernel)

infer(
    _RNG,
    AbstractDiagnostics,
    opt,
    objective_inital.model, objective_inital.data
)


TKernel = infer(_RNG, AbstractDiagnostics, opt.kernel, objective_inital.model, objective_inital.data)
TPrediction = infer(_RNG, opt, objective_inital.model, objective_inital.data)
TGenerated, TGenerated_algorithm = BaytesOptim.infer_generated(_RNG, opt, objective_inital.model, objective_inital.data)
OptimDiagnostics{TPrediction,TKernel,TGenerated, TGenerated_algorithm}
diag
diag.base
results([diag for _ in 1:10], opt, 5, [.1, .2, .3])

gsv = generate_showvalues(diag)
gsv()
generate_showvalues(diag)()

BaytesOptim.result!(opt, opt.kernel.result)
BaytesOptim.get_result(opt)

############################################################################################
predict(_RNG, opt, objective_inital)
generate(_RNG, opt, objective_inital)

# Check if predict / generated / generated_algorithm work with propose step
def = def = OptimDefault(;
    generated = BaytesCore.UpdateTrue(),
    kernel = (;
        magnitude_penalty = 0.5,
        iterations = 123
    )
)
opt = Optimizer(
    _RNG,
    OptimLBFG,
    objective_inital2,
    def,
) 
opt.tune.generated


infer(
    _RNG,
    AbstractDiagnostics,
    opt,
    objective_inital.model, objective_inital.data
)



function ModelWrappers.generate(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{MultiNormal}})
    @unpack model, data = objective
    return 1.
end

function ModelWrappers.generate(_rng::Random.AbstractRNG, algorithm::Optimizer, objective::Objective{<:ModelWrapper{MultiNormal}})
    return [2, 3, 4 ]
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{MultiNormal}})
	return [ Float32(5) Float32(6) ; Float32(7) Float32(8)]
end

infer(
    _RNG,
    AbstractDiagnostics,
    opt,
    objective_inital.model, objective_inital.data
)

_vals, diag = propose!(
    _RNG, opt, objective_inital2.model, objective_inital2.data
)
diag.generated_algorithm
diag.generated
diag.base.prediction