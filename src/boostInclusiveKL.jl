# Note: true_density(X::Array{T, 2}) must operate column-wise on X

"""
generate Particles(x, w) ~ true_density: x ~ approx_d, w = true_density(x) ./ approx_d(x)
* ll: log of true_density
"""
function sample_importance(approx_d::Distribution, ll::Function; n=1000, normalized=true, verbose=false)
    x = rand(approx_d, n)
    w_log = ll(x) - logpdf(approx_d, x)
    idx = isfinite(w_log)
    x = ndims(x)==1 ? x[idx] : x[:, idx]
    w_log = w_log[idx]
    if normalized
        w_log = w_log .- maximum(w_log) + 500.0 # prevent overflow
        w = exp(w_log)
        w = w / sum(w)
        idx = w .> 0. # discard particles with zero weight
    else
        w = exp(w_log)
        idx = isfinite(w) & (w .> 0.)
    end
    if verbose
        info(@sprintf("importance samples: %d out of %d valid (%.4f%%), max = %g, sum = %g",
        sum(idx), n, 100.*sum(idx)/n, maximum(w), 1.0))
    end
    w = w[idx]
    x = ndims(x)==1 ? x[idx] : x[:, idx]
    @assert !isempty(x)
    return Particles(x, w)
end

"goodness of fit KL(true||approx_d) as estimated by Particles, the higher the better"
function cross_entropy(approx_d::Distribution, ll::Function; n=1000)
    pt = sample_importance(approx_d, ll; n=n)
    return -dot(pt.w, logpdf(approx_d, pt.x))
end

"goodness of fit KL(true||(1-α) old_approx_d + α new_component)"
function cross_entropy{T<:Real}(old_approx_d::Distribution, new_component::Distribution,
                                α::Union{AbstractFloat, AbstractVector{T}}, ll::Function; n=1000)
    pt = sample_importance(old_approx_d, ll, n=n)
    return cross_entropy(old_approx_d, new_component, pt, α, ll)
end

"version with supplied Particles"
function cross_entropy(old_approx_d::Distribution, new_component::Distribution, pt::Particles,
                       α::AbstractFloat, ll::Function)
    0 <= α <= 1 || error("bad α = $α")
    z1 = pdf(old_approx_d, pt.x)
    z2 = pdf(new_component, pt.x)
    return -dot(pt.w, log((1-α) * z1 + α * z2))
end

"vector version"
function cross_entropy{T<:Real}(old_approx_d::Distribution, new_component::Distribution, pt::Particles,
                                α::AbstractVector{T}, ll::Function)
    return map(b -> cross_entropy(old_approx_d, new_component, pt, b, ll), α)
end

"best possible fitness_score if approx_d = true"
function self_entropy(true_d::Distribution; n=500000)
    x = rand(true_d, n)
    return -mean(logpdf(true_d, x))
end

"""
Compute helper integrals (up to a constant) with supplied samples x ~ p.

* return: ∫ p(x) [L / ((1-α)q + αh)] dx, ∫ p(x) [L(q-h) / ((1-α)q + αh)^2] dx as
          functions of α, both up to the *same* constant exp(-const_factor).
          This is to prevent numerical issue in Newton's method.
"""
function _helpers_dL_d2L_inclusive{T <: Real}(q::Distribution, h::Distribution,
                                             ll::Function, x::Array{T}; const_factor=0.)
    d = length(q)
    z = ll(x) .- const_factor
    # @show const_factor mean(z) maximum(z) minimum(z)
    qq = pdf(q, x)
    hh = pdf(h, x)
    # @show safemean(log(0.5 * qq + 0.5 * hh))
    return α -> safemean(exp(z - log((1-α) * qq + α * hh))), α -> safemean((qq - hh) ./ ((1-α) * qq + α * hh) .* exp(z - log((1-α) * qq + α * hh)))
end

"return approximate dL/dα and d2L/dα2 as a function of α, with n samples"
function dL_d2L_inclusive(q::Distribution, h::Distribution, ll::Function; n=10000)
    samples_from_q = rand(q, n)
    samples_from_h = rand(h, n)
    return dL_d2L_inclusive(q, h, ll, samples_from_q, samples_from_h)
end

"return approximate dL/dα and d2L/dα2 as a function of α, with supplied samples"
function dL_d2L_inclusive{T <: Real}(q::Distribution, h::Distribution, ll::Function,
                      samples_from_q::Array{T}, samples_from_h::Array{T})
    if length(q)==1
        @assert length(samples_from_q) == length(samples_from_h)
    else
        @assert size(samples_from_q) == size(samples_from_h)
    end
    # the same const factor must be used!
    if length(q)==1
        zz = x -> median( ll(x[1:10]) - log(0.5 * pdf(q, x[1:10]) + 0.5 * pdf(h, x[1:10])) )
    else
        zz = x -> median( ll(x[:, 1:10]) - log(0.5 * pdf(q, x[:, 1:10]) + 0.5 * pdf(h, x[:, 1:10])) )
    end
    const_factor = maximum(zz, (samples_from_q, samples_from_h)) # don't take min!

    dL_q, d2L_q = _helpers_dL_d2L_inclusive(q, h, ll, samples_from_q, const_factor=const_factor)
    dL_h, d2L_h = _helpers_dL_d2L_inclusive(q, h, ll, samples_from_h, const_factor=const_factor)
    return α -> dL_q(α) - dL_h(α), α -> d2L_q(α) - d2L_h(α)
end


"search α, with Stochatic Newton's method, to minimize inclusive KL(true||(1-α) old_approx_d + α new_component).
problem is convex."
function search_α(approx_d::Distribution, new_component::Distribution, ll::Function;
                  n=10000, tol=1e-5, max_iter=100, verbose=false, show_plot=false)
    # get stochastic approximates
    approx_dL, approx_d2L = dL_d2L_inclusive(approx_d, new_component, ll, n=n)
    if show_plot
        α_vec = linspace(0.01, 0.99, 20)
        print(lineplot(α_vec, map(approx_dL, α_vec), title="dL/dα"))
        print(lineplot(α_vec, map(approx_d2L, α_vec), title="d2L/dα2"))
    end
    α = 0.01
    iter = 0
    change = 1.0
    while true
        iter += 1
        if iter > max_iter
          warn(@sprintf("change=%.2f > %g, max_iter=%d reached", change, tol, max_iter))
          break
        end
        # Newton
        # @show approx_dL(α), approx_d2L(α), approx_dL(α)/approx_d2L(α)
        α_new = α - approx_dL(α) / approx_d2L(α)
        α_new = min(α_new, 1-1e-3)
        α_new = max(α_new, 1e-3)
        change = abs(α_new - α)
        α = α_new
        if change < tol
            break
        end
    end
    if verbose
        println(@sprintf "α = %5g (final change = %5g < tol = %5g) with %2d iterations." α change tol iter)
    end
    return α
end
