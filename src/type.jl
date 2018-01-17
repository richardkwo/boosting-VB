"constructor: univariate Particles"
function Particles{T <: AbstractFloat}(x::Vector{T}, w::Vector{Float64})
    p = 1
    n = length(x)
    @assert n == length(w)
    isMultivariate = false
    isNormalized = (sum(w) == 1.0)
    @assert all(w .>= 0)
    return Particles(n, p, isMultivariate, isNormalized, x, w)
end

"constructor: multivariate Particles"
function Particles{T <: AbstractFloat}(x::Matrix{T}, w::Vector{Float64})
    p, n = size(x)
    @assert n == length(w)
    isMultivariate = true
    isNormalized = (sum(w) == 1.0)
    @assert all(w .>= 0)
    return Particles(n, p, isMultivariate, isNormalized, x, w)
end

function normalize!(p::Particles)
    p.isNormalized = true
    p.w = p.w / sum(p.w)
end
