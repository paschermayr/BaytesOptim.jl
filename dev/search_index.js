var documenterSearchIndex = {"docs":
[{"location":"intro/#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"intro/","page":"Introduction","title":"Introduction","text":"Yet to be done.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = BaytesOptim","category":"page"},{"location":"#BaytesOptim","page":"Home","title":"BaytesOptim","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BaytesOptim.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [BaytesOptim]","category":"page"},{"location":"#BaytesOptim.ConfigLBFG","page":"Home","title":"BaytesOptim.ConfigLBFG","text":"struct ConfigLBFG{R<:Real} <: BaytesCore.AbstractConfiguration\n\nDefault Configuration for LBFG optimizer.\n\nFields\n\nmagnitude_penalty::Real: Add -0.5 * magnitude_penalty * sum(abs2, q) to the log posterior when finding the local optimum. This can help avoid getting into high-density edge areas of the posterior which are otherwise not typical (eg multilevel models).\n\niterations::Int64: Maximum number of iterations in the optimization algorithm. Recall that we don't need to find the mode, or even a local mode, just be in a reasonable region.\n\ndifforder::BaytesDiff.DiffOrderOne: Differentiable order for objective function needed to run propagate step\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.ConfigSGD","page":"Home","title":"BaytesOptim.ConfigSGD","text":"struct ConfigSGD{R<:Real, A<:UpdateBool} <: BaytesCore.AbstractConfiguration\n\nDefault Configuration for SGD optimizer.\n\nFields\n\nmagnitude_penalty::Real: Add -0.5 * magnitude_penalty * sum(abs2, q) to the log posterior when finding the local optimum. This can help avoid getting into high-density edge areas of the posterior which are otherwise not typical (eg multilevel models).\n\nmagnitude_adaption::UpdateBool: Adapt magnitude iteratively for each step ~ currently not implemented\n\niterations::Int64: Maximum number of iterations in the optimization algorithm. Recall that we don't need to find the mode, or even a local mode, just be in a reasonable region.\n\ndifforder::BaytesDiff.DiffOrderOne: Differentiable order for objective function needed to run propagate step\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.CustomAlgorithm","page":"Home","title":"BaytesOptim.CustomAlgorithm","text":"mutable struct CustomAlgorithm{R<:BaytesDiff.ℓDensityResult, T<:CustomAlgorithmTune} <: BaytesCore.AbstractAlgorithm\n\nStores information for proposal step.\n\nFields\n\nresult::BaytesDiff.ℓDensityResult\ntune::CustomAlgorithmTune\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.CustomAlgorithmConstructor","page":"Home","title":"BaytesOptim.CustomAlgorithmConstructor","text":"Callable struct to make initializing Algorithm easier in sampling library.\n\nExamples\n\n\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.CustomAlgorithmDefault","page":"Home","title":"BaytesOptim.CustomAlgorithmDefault","text":"struct CustomAlgorithmDefault{I<:AbstractInitialization, U<:UpdateBool}\n\nDefault arguments for Custom constructor.\n\nFields\n\ninit::AbstractInitialization: Method to obtain initial Modelparameter, see 'AbstractInitialization'.\ngenerated::UpdateBool: Boolean if generate(_rng, objective) for corresponding model is stored in Algorithm Diagnostics.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.CustomAlgorithmTune","page":"Home","title":"BaytesOptim.CustomAlgorithmTune","text":"struct CustomAlgorithmTune{T<:ModelWrappers.Tagged, B<:UpdateBool} <: BaytesCore.AbstractTune\n\nStores information used throughout custom algorithm.\n\nFields\n\ntagged::ModelWrappers.Tagged: Tagged Parameter for Algorithm routine\ngenerated::UpdateBool: Boolean if generated quantities should be generated while sampling\niter::BaytesCore.Iterator: Current iteration number\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.DiagnosticsLBFG","page":"Home","title":"BaytesOptim.DiagnosticsLBFG","text":"struct DiagnosticsLBFG <: OptimKernelDiagnostics\n\nDiagnostics for LBFG.\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.DiagnosticsSGD","page":"Home","title":"BaytesOptim.DiagnosticsSGD","text":"struct DiagnosticsSGD <: OptimKernelDiagnostics\n\nDiagnostics for SGD.\n\nFields\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.LBFGTune","page":"Home","title":"BaytesOptim.LBFGTune","text":"struct LBFGTune{R<:Real} <: BaytesCore.AbstractTune\n\nStores information used throughout optimization algorithms.\n\nFields\n\nmagnitude_penalty::Real: Add -0.5 * magnitude_penalty * sum(abs2, q) to the log posterior when finding the local optimum. This can help avoid getting into high-density edge areas of the posterior which are otherwise not typical (eg multilevel models).\n\niterations::Int64: Maximum number of iterations in the optimization algorithm. Recall that we don't need to find the mode, or even a local mode, just be in a reasonable region.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.OptimConstructor","page":"Home","title":"BaytesOptim.OptimConstructor","text":"Callable struct to make initializing Optimizer easier in sampling library.\n\nExamples\n\n\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.OptimDefault","page":"Home","title":"BaytesOptim.OptimDefault","text":"struct OptimDefault{T<:NamedTuple, G, I<:AbstractInitialization, U<:UpdateBool}\n\nDefault arguments for Optim constructor.\n\nFields\n\nkernel::NamedTuple: Tuning Arguments for individual Optimizer\nGradientBackend::Any: Gradient backend used in Optimization step. Not used if Metropolis sampler is chosen.\ninit::AbstractInitialization: Method to obtain initial Modelparameter, see 'AbstractInitialization'.\ngenerated::UpdateBool: Boolean if generate(_rng, objective) for corresponding model is stored in Optimization Diagnostics.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.OptimTune","page":"Home","title":"BaytesOptim.OptimTune","text":"struct OptimTune{T<:ModelWrappers.Tagged, K, B<:UpdateBool} <: BaytesCore.AbstractTune\n\nStores information used throughout optimization algorithms.\n\nFields\n\ntagged::ModelWrappers.Tagged: Tagged Parameter for Optimization routine\nkernel::Any: Tuning arguments for individual Optimizer\ngenerated::UpdateBool: Boolean if generated quantities should be generated while sampling\niter::BaytesCore.Iterator: Current iteration number\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.Optimizer","page":"Home","title":"BaytesOptim.Optimizer","text":"struct Optimizer{M<:OptimKernel, N<:OptimTune} <: BaytesCore.AbstractAlgorithm\n\nStores information for proposal step.\n\nFields\n\nkernel::OptimKernel: Optimizer\ntune::OptimTune: Tuning configuration for kernel.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesOptim.SGDTune","page":"Home","title":"BaytesOptim.SGDTune","text":"struct SGDTune{R<:Real, A<:UpdateBool} <: BaytesCore.AbstractTune\n\nStores information used throughout optimization algorithms.\n\nFields\n\nmagnitude_penalty::Real: Add -0.5 * magnitude_penalty * sum(abs2, q) to the log posterior when finding the local optimum. This can help avoid getting into high-density edge areas of the posterior which are otherwise not typical (eg multilevel models).\n\nmagnitude_adaption::UpdateBool: Adapt magnitude iteratively for each step ~ currently not implemented\n\niterations::Int64: Maximum number of iterations in the optimization algorithm. Recall that we don't need to find the mode, or even a local mode, just be in a reasonable region.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesCore.generate-Tuple{Random.AbstractRNG, CustomAlgorithm, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.generate","text":"generate(_rng, algorithm, objective)\n\n\nGenerate statistics for algorithm given model parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.generate-Tuple{Random.AbstractRNG, Optimizer, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.generate","text":"generate(_rng, algorithm, objective)\n\n\nGenerate statistics for algorithm given model parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.generate_showvalues-Tuple{D} where D<:DiagnosticsLBFG","page":"Home","title":"BaytesCore.generate_showvalues","text":"generate_showvalues(diagnostics)\n\n\nShow relevant diagnostic results.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.generate_showvalues-Tuple{D} where D<:DiagnosticsSGD","page":"Home","title":"BaytesCore.generate_showvalues","text":"generate_showvalues(diagnostics)\n\n\nShow relevant diagnostic results.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, CustomAlgorithm, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, algorithm, model, data)\n\n\nInfer type of predictions of CustomAlgorithm sampler.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, Optimizer, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, optimizer, model, data)\n\n\nInfer type of predictions of Optim sampler.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, Type{BaytesCore.AbstractDiagnostics}, CustomAlgorithm, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, diagnostics, algorithm, model, data)\n\n\nInfer CustomAlgorithm diagnostics type.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, Type{BaytesCore.AbstractDiagnostics}, Optimizer, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, diagnostics, optimizer, model, data)\n\n\nInfer Optim diagnostics type.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.init-Tuple{Type{BaytesCore.AbstractConfiguration}, Type{OptimLBFG}, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.init","text":"init(, optim, objective; magnitude_penalty, iterations)\n\n\nInitialize Mala custom configurations.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.init-Tuple{Type{BaytesCore.AbstractConfiguration}, Type{SGD}, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.init","text":"init(\n    ,\n    optim,\n    objective;\n    magnitude_penalty,\n    magnitude_adaption,\n    iterations\n)\n\n\nInitialize Mala custom configurations.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.init-Tuple{Type{BaytesCore.AbstractTune}, ConfigLBFG, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.init","text":"init(, config, objective)\n\n\nInitialize LBFG custom configurations.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.init-Tuple{Type{BaytesCore.AbstractTune}, ConfigSGD, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.init","text":"init(, config, objective)\n\n\nInitialize SGD custom configurations.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propagate-Tuple{Random.AbstractRNG, CustomAlgorithm, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.propagate","text":"propagate(_rng, algorithm, objective)\n\n\nFunction to dispatch on objective if needed to be extended. Note that objective.model has to be updated manually with estimated parameter in this step\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propagate-Tuple{Random.AbstractRNG, OptimLBFG, OptimTune, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.propagate","text":"propagate(_rng, kernel, tune, objective)\n\n\nPropagate forward one MALA step.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propagate-Tuple{Random.AbstractRNG, SGD, OptimTune, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.propagate","text":"propagate(_rng, kernel, tune, objective)\n\n\nPropagate forward one MALA step.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose!-Union{Tuple{T}, Tuple{D}, Tuple{Random.AbstractRNG, CustomAlgorithm, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, CustomAlgorithm, ModelWrappers.ModelWrapper, D, T}} where {D, T<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.propose!","text":"propose!(_rng, algorithm, model, data)\npropose!(_rng, algorithm, model, data, proposaltune)\n\n\nInplace version of propose.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose!-Union{Tuple{T}, Tuple{D}, Tuple{Random.AbstractRNG, Optimizer, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, Optimizer, ModelWrappers.ModelWrapper, D, T}} where {D, T<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.propose!","text":"propose!(_rng, optim, model, data)\npropose!(_rng, optim, model, data, proposaltune)\n\n\nInplace version of propose.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose-Tuple{Random.AbstractRNG, CustomAlgorithm, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.propose","text":"propose(_rng, algorithm, objective)\n\n\nPropose new parameter with Algorithm.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose-Tuple{Random.AbstractRNG, Optimizer, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.propose","text":"propose(_rng, optim, objective)\n\n\nPropose new parameter with optimizer. If update=true, objective function will be updated with input model and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.results-Union{Tuple{M}, Tuple{T}, Tuple{AbstractVector{M}, CustomAlgorithm, Integer, Vector{T}}} where {T<:Real, M<:CustomAlgorithmDiagnostics}","page":"Home","title":"BaytesCore.results","text":"results(diagnosticsᵛ, algorithm, Ndigits, quantiles)\n\n\nPrint result for a single trace.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.results-Union{Tuple{M}, Tuple{T}, Tuple{AbstractVector{M}, Optimizer, Integer, Vector{T}}} where {T<:Real, M<:OptimDiagnostics}","page":"Home","title":"BaytesCore.results","text":"results(diagnosticsᵛ, algorithm, Ndigits, quantiles)\n\n\nPrint result for a single trace.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Union{Tuple{S}, Tuple{CustomAlgorithmTune, S}} where S<:BaytesDiff.ℓObjectiveResult","page":"Home","title":"BaytesCore.update!","text":"update!(tune, result)\n\n\nUpdate tuning fields at current iteration.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.update!-Union{Tuple{S}, Tuple{OptimTune, S}} where S<:BaytesDiff.ℓObjectiveResult","page":"Home","title":"BaytesCore.update!","text":"update!(tune, result)\n\n\nUpdate Optim tuning fields at current iteration.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesOptim._optimize","page":"Home","title":"BaytesOptim._optimize","text":"Internal optimization function via 'Optim' and 'NLSolversBase' - both have to be loaded and are used as an Extension if used.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#BaytesOptim.infer_generated-Union{Tuple{D}, Tuple{Random.AbstractRNG, CustomAlgorithm, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesOptim.infer_generated","text":"infer_generated(_rng, algorithm, model, data)\n\n\nInfer type of generated quantities of CustomAlgorithm.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesOptim.infer_generated-Union{Tuple{D}, Tuple{Random.AbstractRNG, Optimizer, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesOptim.infer_generated","text":"infer_generated(_rng, optimizer, model, data)\n\n\nInfer type of generated quantities of Optimizer.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"}]
}
