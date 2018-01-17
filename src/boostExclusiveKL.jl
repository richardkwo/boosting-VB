# boosting algorithm based on minimizing KL(q||p)
# ELBO
"approximated ELBO based on n samples"
function ELBO(approx_d::Distribution, ll::Function; n=100000)
    x = rand(approx_d, n)
    return ELBO(approx_d, ll, x)
end

"approximate ELBO using supplied samples x"
function ELBO{T <: Real}(approx_d::Distribution, ll::Function, x::Array{T})
    Z = ll(x) - logpdf(approx_d, x)
    return safemean(Z)
end

# mean, with Inf and NaN removed
function safemean{T <: Real}(x::AbstractVector{T})
    # @show mean(isfinite(x))
    x = x[isfinite.(x)]
    !isempty(x) || error("No finite value left for mean: x =", x)
    return mean(x)
end

# stochastic approximation of L as a function of α
"return approximate exclusive L as a function of α, with supplied samples"
function loss_exclusive{T <: Real}(q::Distribution, h::Distribution, ll::Function, samples_from_q::Array{T}, samples_from_h::Array{T})
    if length(q)==1
        @assert length(samples_from_q) == length(samples_from_h)
    else
        @assert size(samples_from_q) == size(samples_from_h)
    end
    f1 = _helper_dL_exclusive(q, h, ll, samples_from_h)
    f2 = _helper_dL_exclusive(q, h, ll, samples_from_q)
    return α -> α * f1(α) + (1-α) * f2(α)
end

"return approximate exclusive L as a function of α, with n samples"
function loss_exclusive(q::Distribution, h::Distribution, ll::Function; n=10000)
    samples_from_h = rand(h, n)
    samples_from_q = rand(q, n)
    return loss_exclusive(q, h, ll, samples_from_q, samples_from_h)
end

# stochastic approximations of dL/dα and d2L/dα2
"return ∫ p(x) [log ((1-α) q + α h) - log L] dx, as a function of α, with supplied samples x ~ p"
function _helper_dL_exclusive{T <: Real}(q::Distribution, h::Distribution, ll::Function, x::Array{T})
    z1 = pdf(q, x) + ϵ_lower_bound
    z2 = pdf(h, x) + ϵ_lower_bound
    z3 = logsumexp(ll(x) + log(ϵ_lower_bound))
    # @show mean(isfinite(z1)), mean(isfinite(z2)), mean(isfinite(z3))
    return α -> safemean( log.((1-α) * z1 + α * z2) - z3 )
end

"return approximate dL/dα as a function of α, with supplied samples"
function dL_exclusive{T <: Real}(q::Distribution, h::Distribution, ll::Function, samples_from_q::Array{T}, samples_from_h::Array{T})
    if length(q)==1
        @assert length(samples_from_q) == length(samples_from_h)
    else
        @assert size(samples_from_q) == size(samples_from_h)
    end
    f1 = _helper_dL_exclusive(q, h, ll, samples_from_h)
    f2 = _helper_dL_exclusive(q, h, ll, samples_from_q)
    return α -> f1(α) - f2(α)
end

"return approximate dL/dα as a function of α, with n samples"
function dL_exclusive(q::Distribution, h::Distribution, ll::Function; n=10000)
    samples_from_h = rand(h, n)
    samples_from_q = rand(q, n)
    return dL_exclusive(q, h, ll, samples_from_q, samples_from_h)
end

"return ∫ p(x) [(h - q) / ((1-α)q + αh)] dx, as a function of α, with supplied samples x ~ p"
function _helper_d2L_exclusive{T <: Real}(q::Distribution, h::Distribution, x::Array{T})
    hh = pdf(h, x) + ϵ_lower_bound
    qq = pdf(q, x) + ϵ_lower_bound
    # @show mean(isfinite(hh)), mean(isfinite(qq))
    return α -> safemean( (hh - qq) ./ ((1-α) * qq + α * hh) )
end

"return approximate d2L_exclusive/dα2 as a function of α, with supplied samples"
function d2L_exclusive{T <: Real}(q::Distribution, h::Distribution, samples_from_q::Array{T}, samples_from_h::Array{T})
    if length(q)==1
        @assert length(samples_from_q) == length(samples_from_h)
    else
        @assert size(samples_from_q) == size(samples_from_h)
    end
    f1 = _helper_d2L_exclusive(q, h, samples_from_h)
    f2 = _helper_d2L_exclusive(q, h, samples_from_q)
    return α -> f1(α) - f2(α)
end

"return approximate d2L/dα2 as a function of α, with n samples"
function d2L_exclusive(q::Distribution, h::Distribution; n=10000)
    samples_from_h = rand(h, n)
    samples_from_q = rand(q, n)
    return d2L_exclusive(q, h, samples_from_q, samples_from_h)
end

# minimization of L(α)
"search α with Newton's method"
function search_α_exclusive_newton(approx_d::Distribution, new_component::Distribution, ll::Function;
                  init_step_size=1., n=100, n_plot=1000, tol=1e-4, max_iter=1000, verbose=false, show_plot=false)
    # unicode plot
    if show_plot
        # get samples
        samples_from_q = rand(approx_d, n_plot)
        samples_from_h = rand(new_component, n_plot)
        approx_loss = loss_exclusive(approx_d, new_component, ll, samples_from_q, samples_from_h)
        α_vec = linspace(0.01, 0.99, 20)
        print(lineplot(α_vec, map(approx_loss, α_vec), title="loss"))
    end
    α = 0.
    iter = 0
    change = 1.
    while true
        iter += 1
        if iter > max_iter
            warn(@sprintf("change=%5g > %5g, max_iter=%2d reached", change, tol, max_iter))
            break
        end
        # get stochastic approximates
        samples_from_q = rand(approx_d, n)
        samples_from_h = rand(new_component, n)
        approx_dL = dL_exclusive(approx_d, new_component, ll, samples_from_q, samples_from_h)
        approx_d2L = d2L_exclusive(approx_d, new_component, samples_from_q, samples_from_h)
        # Newton's
        step_size =  init_step_size / iter
        α_new = α - step_size * approx_dL(α) / approx_d2L(α)
        α_new = min(α_new, 1.0)
        α_new = max(α_new, 0.0)
        change = abs(α_new - α)
        α = α_new
        if change < tol && iter > 200
            break
        end
    end
    if verbose
        println(@sprintf "α = %5g (change = %5g < tol = %5g) with %2d iterations." α change tol iter)
    end
    return α
end

# minimization of L(α)
"search α with SGD"
function search_α_exclusive_sgd(approx_d::Distribution, new_component::Distribution, ll::Function;
                  init_step_size=1.0, n=100, tol=1e-4, min_iter = 100, max_iter=10000, verbose=false)
    α = 0.
    iter = 0
    change = 1.
    while true
        iter += 1
        if iter > max_iter
            warn(@sprintf("change=%5g > %5g, max_iter=%2d reached", change, tol, max_iter))
            break
        end
        # get stochastic approximates
        samples_from_q = rand(approx_d, n)
        samples_from_h = rand(new_component, n)
        approx_dL = dL_exclusive(approx_d, new_component, ll, samples_from_q, samples_from_h)
        # Newton's
        step_size =  init_step_size / iter
        α_new = α - step_size * approx_dL(α)
        α_new = min(α_new, 1.0)
        α_new = max(α_new, 0.0)
        change = abs(α_new - α)
        α = α_new
        if change < tol && iter > min_iter
            break
        end
    end
    if verbose
        println(@sprintf "α = %5g (change = %5g < tol = %5g) with %2d iterations." α change tol iter)
    end
    return α
end
