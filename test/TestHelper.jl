############################################################################################
# Constants
"RNG for sampling based solutions"
const _RNG = Random.Xoshiro(123)    # shorthand
Random.seed!(_RNG, 1)

"Tolerance for stochastic solutions"
const _TOL = 1.0e-6

"Number of samples"
N = 10^3

############################################################################################
# Define MV Normal Model
import PDMats: PDMats, PDMat
using Distributions, LinearAlgebra
function PDMat(fac::Cholesky, scale::AbstractVector)
    factors_scaled = fac.uplo == 'U' ? fac.factors .* scale' : scale .* fac.factors
    return PDMat(Cholesky(factors_scaled, fac.uplo, 0))
end

ρ = [1. .2 .3 ; .2 1. .4 ; .3 .4 1.]
_b = bijector(Distributions.LKJCholesky(3, 10.))

struct MultiNormal <: ModelName end
param = (;
    μ = Param(
        MvNormal([0., 0., 0.], LinearAlgebra.diagm(repeat([10.], 3) ) ), 
        [1., 2., 3.]
    ),
    scale = Param(
        [Normal(1, 10.), Normal(1, 10.), Normal(1, 10.)],
        [1., 2., 3.],
    ),
    ρ = Param(
        Distributions.LKJCholesky(3, 10.),
        inverse(_b)( _b( Cholesky(ρ, :L, 0) ) )
    ),
)

function ModelWrappers.simulate(_RNG::Random.AbstractRNG, model::ModelWrapper{F}; Nsamples = 1000) where {F<:Union{MultiNormal}}

    Σ = PDMat(model.val.ρ, model.val.scale)
    return rand(_RNG, MvNormal(model.val.μ, Σ), Nsamples)
end

function (objective::Objective{<:ModelWrapper{M}})(θ::NamedTuple) where {M<:Union{MultiNormal}}
    @unpack model, tagged = objective
## Prior
    lp = log_prior(tagged.info.transform.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    ll = 0.0
#    Σ = PDMat(θ.ρ, θ.scale)
    Σ = Symmetric( diagm(θ.scale) * θ.ρ.factors * θ.ρ.factors' * diagm(θ.scale) )
#    Σ = diagm(ones(3))
    d =  MvNormal(θ.μ, Σ)
    ll = 0.0
    for t in 1:size(data, 2)
        ll += logpdf(d, data[:,t])
    end
    return ll + lp
#    return sum( logpdf(d, data[:,t]) for t in 1:size(data, 2))
end

