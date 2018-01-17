using StatsFuns.logistic

function loglikelihood_logistic{S <: Integer, T <: Real}(β::Vector{T}, y::Vector{S}, X::AbstractArray{T, 2})
    @assert length(y) == size(X, 1) && length(β) == size(X, 2)
    z = map(logistic, X * β)
    return dot(y, log(z)) + dot(1-y, log(1-z))
end

function loglikelihood_logistic{S <: AbstractFloat, T <: Real}(β::Vector{T}, y::Vector{S}, X::AbstractArray{T, 2})
    return loglikelihood_logistic(β, convert(Vector{Int}, y), X)
end
