push!(LOAD_PATH, pwd()*"/../../src/")
using Distributions
using PyPlot
using BVB

lower = -100.
upper = 100.

max_iter = 30
init_approx_d =  Normal(0, 40)

# example 2
true_d = MixtureModel(Normal, [(-3.5, 1), (-1.0, 0.4), (1.0, 0.5), (6.0, 1)], [0.2, 0.3, 0.3, 0.2])
ll(x) = logpdf(true_d, x)
x = linspace(-20,20,500)

include("run_laplacian_gradient_boosting_exclusive.jl")
save_trace("./mog-1/", x, init_approx_d, approx_d, true_d,
                    iter_vec, α_vec,
                    eval_vec, -elbo_vec)
show()

println("Done.")
readline(STDIN)
