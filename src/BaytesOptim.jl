module BaytesOptim

################################################################################
#Import modules
using BaytesCore:
    BaytesCore,
    AbstractAlgorithm,
    AbstractTune,
    AbstractConfiguration,
    AbstractDiagnostics,
    AbstractKernel,
    AbstractConstructor,
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    update,
    SampleDefault,
    ProposalTune,
    Iterator

import BaytesCore:
    BaytesCore,
    update,
    update!,
    infer,
    results,
    init,
    init!,
    propose,
    propose!,
    propagate,
    propagate!,
    generate_showvalues,
    generate

using ModelWrappers:
    ModelWrappers,
    ModelWrapper,
    Tagged,
    Objective,
    sample,
    sample!,
    length_unconstrained

import ModelWrappers:
    ModelWrappers,
    predict,
    generate,
    AbstractInitialization,
    NoInitialization,
    PriorInitialization,
    OptimInitialization

using BaytesDiff:
    BaytesDiff,
    DiffObjective,
    AbstractDifferentiableTune,
    AbstractDiffOrder,
    DiffOrderZero,
    DiffOrderOne,
    DiffOrderTwo,
    ℓObjectiveResult,
    ℓDensityResult,
    ℓGradientResult,
    checkfinite,
    AutomaticDiffTune,
    AnalyticalDiffTune,
    log_density_and_gradient

import BaytesDiff:
    BaytesDiff,
    checkfinite

using Random: Random, AbstractRNG, GLOBAL_RNG, randexp

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
using ArgCheck: ArgCheck, @argcheck
using SimpleUnPack: SimpleUnPack, @unpack, @pack!

# Import Solver
#using NLSolversBase, Optim

################################################################################
#Abstract types to be dispatched in Examples section
abstract type OptimKernel <: AbstractKernel end
abstract type OptimKernelDiagnostics <: AbstractDiagnostics end

include("Core/Core.jl")
include("Kernels/Kernels.jl")

################################################################################
export
    # BaytesCore
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    propose,
    propose!,
    propagate,
    propagate!,
    update!,
    SampleDefault,

    # ModelWrappers
    init,
    init!,
    predict,
    generate,
    AbstractInitialization,
    NoInitialization,
    PriorInitialization,
    OptimInitialization,

    #BaytesDiff
    checkfinite,

    #Optimizer
    OptimKernel,
    Optimiagnostics,
    OptimKernelDiagnostics
    
end
