push!(LOAD_PATH, pwd()*"/../../src/")
using Distributions
using DataFrames
using StatsFuns.logistic
using BVB

"""
evaluate ll for logistic regression, for a single β vector.
* y: vector of +1/-1
* λ: prior on β is N(β|0, 1/λ I)
"""
@inline function ll_logistic(β::Vector{Float64}, y::Vector{Float64}, X::Matrix{Float64}; λ=0.)
    return sum(log.(logistic.(y .* (X * β)))) - 0.5 * λ * sum(β.^2)
end

"matrix version for each column of β, return a vector"
@inline function ll_logistic(β::Matrix{Float64}, y::Vector{Float64}, X::Matrix{Float64}; λ=0.)
    return Vector{Float64}([ll_logistic(β[:,i], y, X, λ=λ) for i in 1:size(β, 2)])
end

"factory producing mini-batch ll"
function ll_logistic_factory(batch_size::Int, y::Vector{Float64}, X::Matrix{Float64}; λ=0.)
    N = length(y)
    @assert N == size(X, 1)
    @assert batch_size <= N
    idx = rand(1:N, batch_size)
    # must copy! otherwise too slow!
    _y = copy(y[idx])
    _X = copy(X[idx, :])
    @inline function _ll(β::Union{Vector{Float64}, Matrix{Float64}})
        return ll_logistic(β, _y, _X, λ=λ) ./ batch_size .* N
    end
    return _ll
end

"logistic prediction: +1/-1"
@inline function predict_logistic{T <: Real}(β::Vector{T}, X::Matrix{T})
    y = ones(size(X, 1))
    y[predict_prob_logistic(β, X) .< 0.5] = -1.
    return y
end

"logistic prediction: column-wise for matrix β"
@inline function predict_logistic{T <: Real}(β::Matrix{T}, X::Matrix{T})
    return hcat([predict_logistic(β[:,i], X) for i in size(β, 2)] ...)
end

"logistic prob prediction: prob of +1"
@inline function predict_prob_logistic{T <: Real}(β::Vector{T}, X::Matrix{T})
    return logistic.(X * β)
end

"logistic prob prediction: column-wise for matrix β"
@inline function predict_prob_logistic{T <: Real}(β::Matrix{T}, X::Matrix{T})
    return hcat([predict_prob_logistic(β[:,i], X) for i in size(β, 2)] ...)
end

"logistic prediction: Bayesian model averaging"
@inline function predict_logistic_BMA{T <: Real}(β::Matrix{T}, X::Matrix{T})
    y = ones(size(X, 1))
    P = predict_prob_logistic(β, X)
    for i in 1:length(y)
        if mean(P[i, :]) < 0.5
            y[i] = -1.
        end
    end
    return y
end