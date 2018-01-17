
"return (1-α) old_mixture + α new_component, as a mixture"
function add_to_mixture(old_mixture::AbstractMixtureModel, new_component::Distribution, α::AbstractFloat)
    if α < 1e-4
        return old_mixture
    end
    comps = components(old_mixture)
    priors = probs(old_mixture)
    push!(comps, new_component)
    priors = priors * (1-α)
    push!(priors, α)
    return MixtureModel(comps, priors)
end

"variant: when old_mixture is not yet a mixture"
function add_to_mixture(old_mixture::Distribution, new_component::Distribution, α::AbstractFloat)
    if α < 1e-6
        return old_mixture
    end
    return MixtureModel([old_mixture, new_component], [1-α, α])
end

"return the distribution for the last component"
function get_component_distribution(d::AbstractMixtureModel)
    return typeof(components(d)[end])
end

"for non-mixture distribution"
function get_component_distribution(d::Distribution)
    return typeof(d)
end

"estimate normalization factor with mean of {true_density(x) / approx(x)}"
function estimate_normalizing_factor(approx_d::Distribution, ll::Function; n=1000)
    pt = sample_importance(approx_d, ll, n=n, normalized=false)
    return mean(pt.w)
end
