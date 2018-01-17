function stable_logpdf(approx_d::Distribution; logϵ = log(ϵ_lower_bound))
    _logpdf(x) = map(s -> logsumexp(s, logϵ), logpdf(approx_d, x))
    return _logpdf
end

function stable_logpdf(approx_d::Distribution, x; logϵ = log(ϵ_lower_bound))
    return map(s -> logsumexp(s, logϵ), logpdf(approx_d, x))
end

function stable_ll(ll::Function; logϵ = log(ϵ_lower_bound))
    _ll(x) = map(s -> logsumexp(s, logϵ), ll(x))
    return _ll
end

function stable_ll(ll::Function, x; logϵ = log(ϵ_lower_bound))
    return map(s -> logsumexp(s, logϵ), ll(x))
end

function stable_negative_log_residual(ll::Function, approx_d::Distribution; logϵ = log(ϵ_lower_bound))
    _negative_log_res(x) = -ll(x) + stable_logpdf(approx_d, x; logϵ = logϵ)
    return _negative_log_res
end
