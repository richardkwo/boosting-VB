push!(LOAD_PATH, pwd()*"/../../src/")
push!(LOAD_PATH, pwd()*"/src/")
using Distributions
using PyPlot
using BVB

lower = -20.
upper = 20.

true_d = Cauchy(0.0, 2)
ll(x) = logpdf(true_d, x)

init_approx_d = Normal(0, 40)
x = linspace(-20,20,500)

max_iter = 30

# include("run_laplacian_gradient_boosting.jl")
include("run_laplacian_gradient_boosting_exclusive.jl")
save_trace("./cauchy/", x, init_approx_d, approx_d, true_d,
                    iter_vec, α_vec,
                    eval_vec, -elbo_vec)
show()

println("Done.")
