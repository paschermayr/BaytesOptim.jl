
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
θ_proposed = unflatten_constrain(objective.model, objective.tagged, θᵤ_proposed)
θ_proposed.ρ
#println("True")
model.val
model.val.ρ

propose(
    _RNG, opt, objective_inital,
) 

propose!(
    _RNG, opt, objective_inital.model, objective_inital.data
) 
