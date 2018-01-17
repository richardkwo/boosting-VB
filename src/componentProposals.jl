"""
  propose_new_component(old_mixture::Distribution, ll::Function; algorithm="plain", method="MLE", n=10, n_est_c=10000)

Propose new component to `old_mixture` for approximating `true_density`.

* `ll`: log of true_density
* `algorithm`: `plain`, `ratio`, `rejection` or `demean`
* `method`: method for fitting the new component with weighted samples. Default to `MLE`.
* `n`: number of particles
* `n_est_c`: number of particles for estimating the normalization factor for `demean` algorithm
"""
function propose_new_component(old_mixture::Distribution, ll::Function;
                               algorithm="plain", method="MLE", n=10, n_est_c=10000)
    if algorithm == "plain"
        return propose_new_component_plain(old_mixture, ll, n=n, method=method)
    elseif algorithm == "ratio"
        return propose_new_component_ratio(old_mixture, ll, n=n, method=method)
    elseif algorithm == "rejection"
        return propose_new_component_rejection(old_mixture, ll, n=n, method=method)
    elseif algorithm == "demean"
        return propose_new_component_demean(old_mixture, ll, n=n, n_est_c=n_est_c, method=method)
    else
        error("wrong method")
    end
end

"fit new component of distribution D with weighted samples"
function fit_new_component{D <: Distribution, T <: AbstractFloat}(d::Type{D}, x::Array{T}, w::Vector{Float64}; method="MLE")
    w = w / sum(w)
    if method == "MLE"
        return fit_mle(d, x, w)
    else
        error("wrong method")
    end
end

"equal weight variant"
function fit_new_component{D <: Distribution, T <: AbstractFloat}(d::Type{D}, x::Array{T}; method="MLE")
    n = ndims(x)==1 ? length(x) : size(x, 2)
    return fit_new_component(d, x, ones(n) / n, method=method)
end


"proposal: particles ~ true distribution"
function propose_new_component_plain(old_mixture::Distribution, ll::Function; n=10, method="MLE")
  pt = sample_importance(old_mixture, ll, n=n)
  return fit_new_component(get_component_distribution(old_mixture), pt.x, pt.w, method=method)
end

"proposal: particles ~ exp(log true - log approx)"
function propose_new_component_ratio(old_mixture::Distribution, ll::Function; n=10, method="MLE")
  pt = sample_importance(old_mixture, ll, n=n, normalized=false)
  w = pt.w ./ pdf(old_mixture, pt.x)
  return fit_new_component(get_component_distribution(old_mixture), pt.x, w, method=method)
end

"max density for univariate normal"
function max_density(d::Normal)
  return pdf(d, d.μ)
end

"max density for multivariate normal"
function max_density(d::MultivariateNormal)
    return pdf(d, d.μ)
end

"max density for a mixture model"
function max_density(d::AbstractMixtureModel)
  K = length(probs(d))
  return maximum([pdf(d, d.components[i].μ) for i in 1:K])
end

"proposal: particles ~ true distribution, but rejecting those around the max mode of the approx"
function propose_new_component_rejection(old_mixture::Distribution, ll::Function; n=10, method="MLE")
  γ = max_density(old_mixture)
  pt = sample_importance(old_mixture, ll, n=n)
  rej = 1.0 - pdf(old_mixture, pt.x) ./ γ
  rej[rej .< 0.0] = 0.0 # small numerical error may make it negative
  w = pt.w .* rej
  return fit_new_component(get_component_distribution(old_mixture), pt.x, w, method=method)
end


"proposal: `demean estimator` particle ~ (true density - approx) ∨ 0"
function propose_new_component_demean(old_mixture::Distribution, ll::Function; n=10, n_est_c=1000, method="MLE")
    mean_w_est = estimate_normalizing_factor(old_mixture, ll, n=n_est_c)
    return propose_new_component_demean(old_mixture, ll, mean_w_est; n=n, method=method)
end

"proposal: `demean estimator` particle ~ (true density - approx) ∨ 0, with mean_w as an input"
function propose_new_component_demean(old_mixture::Distribution, ll::Function, mean_w_est::AbstractFloat; n=10, method="MLE")
    w = zeros(n)
    p = length(old_mixture)
    if p==1
        x = zeros(n)
    else
        x = zeros((p, n))
    end
    i = 1
    while i <= n
        if p==1
            x[i] = rand(old_mixture)
            w[i] = exp(ll(x[i]) - logpdf(old_mixture, x[i])) - mean_w_est # demean
        else
            x[:, i] = rand(old_mixture)
            w[i] = exp(ll(x[:, i]) - logpdf(old_mixture, x[:, i])) - mean_w_est # demean
        end

        if w[i] > 0
            i += 1
        end
    end
    return fit_new_component(get_component_distribution(old_mixture), x, w, method=method)
end
