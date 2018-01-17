"seek h by Laplacian approximation to L/q, L: log of true_density, q: current_approx.
Laplacian: μ = argmax (log L - Log q), Σ = (- Hessian of log L at μ)^(-1)
The new compoent is N(μ, Σ/2).

Univariate version."
function get_new_component_laplacian(ll::Function, current_approx::UnivariateDistribution;
                                    init_approx = false, lower = -100., upper = 100.,
                                    show_trace=false, verbose=false)
    # get init point
    opt_init = rand(current_approx)
    if init_approx
        # get the 1st component: scaled Laplacian approximation
        f(x) = -ll(x)
    else
        # maximize log L - log q
        # Julia optimization routine uses "Brent's" for univariate optimization
        # lower and upper bound must be provided
        f = stable_negative_log_residual(ll, current_approx)  # obj to minimize
    end

    opt_result = optimize(f, lower, upper, method = GoldenSection())
    # check convergence
    if verbose
        println(opt_result)
    end
    converged(opt_result) || error("Optimization not converged", opt_result)
    # get the Hessian
    η = opt_result.minimizer
    # Finite Differencing
    H = second_derivative(f, η)
    println(H)
    new_component = laplacian_fit(get_component_distribution(current_approx), η, H)
    if init_approx
        println(@sprintf("Initial approximation: N(μ=%f, σ=%f)", new_component.μ, new_component.σ))
    end
    return new_component
end

"seek h by Laplacian approximation to L/q, L: log of true_density, q: current_approx.
Laplacian: μ = argmax (log L - Log q), Σ = (- Hessian of log L at μ)^(-1)
The new compoent is N(μ, Σ/2).

Multivariate version, random initialization."
function get_new_component_laplacian(ll::Function, current_approx::MultivariateDistribution;
                                    init_approx = false, diagonal = false,
                                    max_iter=1000, show_trace=false, verbose=false)

    opt_init = rand(current_approx)
    return get_new_component_laplacian(ll, current_approx, opt_init;
                                       init_approx = init_approx,
                                       diagonal = diagonal,
                                       max_iter=max_iter,
                                       show_trace=show_trace, verbose=verbose)
end

"with supplied initialization."
function get_new_component_laplacian{T <: Real}(ll::Function,
                                                current_approx::MultivariateDistribution,
                                                x_init::AbstractVector{T};
                                                init_approx = false,
                                                diagonal = false,
                                                max_iter=1000, show_trace=false, verbose=false)
    d = length(current_approx)
    if init_approx
        # get the 1st component: scaled Laplacian approximation
        f(x) = -ll(x)
    else
        # maximize log L - log q
        f = stable_negative_log_residual(ll, current_approx)  # obj to minimize
    end
    opt_result = optimize(f, x_init, LBFGS(),
                Options(iterations = max_iter,
                                    show_trace = show_trace))
    # check convergence
    converged(opt_result) || error("Optimization not converged", opt_result)
    if verbose
        println(opt_result)
    end
    # get the Hessian
    η = opt_result.minimizer
    # Finite Differencing, H should be positive definite
    if d > 20
        println("(.. estimating hessian ..)")
    end
    H = hessian(f, η)
    H = (H + H') / 2.
    if diagonal
        H = diagm(diag(H))
    end
    if d > 20
        if diagonal
            println("(.. diagonal hessian approximated ..)")
        else
            println("(.. cholesky on hessian ..)")
        end
    end
    H = PDMat(H)
    new_component = laplacian_fit(get_component_distribution(current_approx), η, H)
    if init_approx
        println("Initial approximation:N(μ = ", new_component.μ, ", Σ = ", new_component.Σ)
    end
    return new_component
end

"""
Laplacian fit for multivariate Gaussian

* η: argmax log L - log q
* H: Hessian of (-log L + log q) at η

return h = MvNormal(η, inv(H)/2)
"""
function laplacian_fit{D <: MvNormal, T <: Real}(::Type{D}, η::Vector{T}, H::PDMat)
    @assert length(η) == size(H, 1)
    Σ = inv(H) / 2.0
    # Σ = inv(H)
    all(isfinite.(η)) || error("Invalid number encountered in Laplacian fit to Gaussian")
    if length(η) > 20
        println("(.. new Gaussian component determined ..)")
    end
    return MvNormal(η, Σ) # optimal scaling
end

"Laplacian fit for univariate Gaussian"
function laplacian_fit{D <: Normal, T <: Real}(::Type{D}, η::T, H::T)
    # σ2 = 1.0 / H / 2.0
    σ2 = 1.0 / H / 2.0
    if !(isfinite.(σ2) && σ2 > 0)
        warn("ill H = ", H)
        return Normal(η, 1e-4)
    end
    return Normal(η, sqrt(σ2)) # optimal scaling, 2nd argument is sd
end
