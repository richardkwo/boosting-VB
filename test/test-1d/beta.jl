push!(LOAD_PATH, pwd()*"/../../src/")
using Distributions
using PyPlot
using BVB

x = linspace(-0.1, 1.1, 500)
init_approx_d = Normal(0, 4)

lower = 0.
upper = 1.
max_iter = 100

# example 1
@show true_d = Beta(2.0, 2.0)
ll(x) = logpdf(true_d, x)

# include("run_laplacian_gradient_boosting.jl")
include("run_laplacian_gradient_boosting_exclusive.jl")
show()

# example 2
@show true_d = Beta(2.0, 5.0)
ll(x) = logpdf(true_d, x)

# include("run_laplacian_gradient_boosting.jl")
include("run_laplacian_gradient_boosting_exclusive.jl")
show()

# example 3
@show true_d = Beta(0.6, 0.5)
ll(x) = logpdf(true_d, x)

# include("run_laplacian_gradient_boosting.jl")
include("run_laplacian_gradient_boosting_exclusive.jl")
show()

println("Done.")
readline(STDIN)
