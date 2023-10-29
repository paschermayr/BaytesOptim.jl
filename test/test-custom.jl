############################################################################################
# Models to be used in construction
model = ModelWrapper(MultiNormal(), param)
data = simulate(_RNG, model)

objectives = [
    Objective(ModelWrapper(MultiNormal(), param, (;), FlattenDefault()), data),
    Objective(ModelWrapper(MultiNormal(), param, (;), FlattenDefault(; output = Float32)), data)
]

#=
iter = 2
=#

## Add custom Step for propagate
using BaytesDiff
import BaytesOptim: BaytesOptim, propagate

############################################################################################
@testset "Sampling, type conversion" begin
    for iter in eachindex(objectives)
            _obj = deepcopy(objectives[iter])
            _flattentype = _obj.model.info.reconstruct.default.output

            # Create Custom Algorithm
            def = CustomAlgorithmDefault(; 
                generated=UpdateTrue()
            )
            opt = CustomAlgorithm(
                _RNG,
                _obj,
                def,
            ) 
            # Check Default calls
            BaytesOptim.propagate(
                _RNG, opt, _obj
            )
            _vals, _diags = propose(
                _RNG, opt, _obj,
            ) 
            _vals, _diags = _vals, diag = propose!(
                _RNG, opt, _obj.model, _obj.data
            ) 
            _vals
            _diags.generated
            _diags.generated_algorithm
            _diags.base.prediction

            ## Add generate Methods
            function ModelWrappers.generate(_RNG::Random.AbstractRNG, objective::Objective{<:ModelWrapper{MultiNormal}})
                @unpack model, data = objective
                return 1.
            end

            function ModelWrappers.generate(_RNG::Random.AbstractRNG, algorithm::CustomAlgorithm, objective::Objective{<:ModelWrapper{MultiNormal}})
                return [2, 3, 4 ]
            end
            _vals, _diags = _vals, diag = propose!(
                _RNG, opt, _obj.model, _obj.data
            ) 
            _vals
            _diags.generated
            _diags.generated_algorithm
            _diags.base.prediction

            ## Extend Custom Method
            function propagate(
                _RNG::Random.AbstractRNG, algorithm::CustomAlgorithm, objective::Objective{<:ModelWrapper{MultiNormal}})
                logobjective = BaytesDiff.ℓDensityResult(objective)
                #logobjective.θᵤ[1] = 5
                logobjective.θᵤ[1] = rand()
                return logobjective
            end
            propagate(
                _RNG, opt, _obj
            )
            _vals, _diags = _vals, diag = propose!(
                _RNG, opt, _obj.model, _obj.data
            ) 
            _vals
            _diags.generated
            _diags.generated_algorithm
            _diags.base.prediction
            
            ## Some custom constructors to check
            _C = CustomAlgorithm(:μ)
        

            TPrediction = BaytesOptim.infer(_RNG, opt, _obj.model, _obj.data)
            TGenerated, TGenerated_algorithm = BaytesOptim.infer_generated(_RNG, opt, _obj.model, _obj.data)
            @test typeof(_diags) == CustomAlgorithmDiagnostics{TPrediction, TGenerated, TGenerated_algorithm}

            gsv = generate_showvalues(_diags)
            gsv()
            generate_showvalues(_diags)()
    end
end
