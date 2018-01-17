module BVB
using Distributions
using Optim: optimize, converged, LBFGS, Options, GoldenSection
using PDMats: PDMat, inv
using Calculus: second_derivative, hessian
using UnicodePlots: lineplot, lineplot!, scatterplot, scatterplot!, barplot
using StatsFuns: logsumexp

abstract type abstractParticles end

type Particles{T <: AbstractFloat} <: abstractParticles
    n               ::Int
    p               ::Int
    isMultivariate  ::Bool
    isNormalized    ::Bool
    x               ::Array{T}
    w               ::Vector{Float64}
end

const ϵ_lower_bound = 1e-4

include("type.jl")
include("boosting.jl")
include("boostInclusiveKL.jl")
include("boostExclusiveKL.jl")
include("numericalStable.jl")
include("utils.jl")
include("models.jl")
include("gradientBoosting.jl")
include("wrapper.jl")
end
