push!(LOAD_PATH, pwd()*"/../../src/")
using Distributions
using PyPlot
using BVB

x = linspace(-15, 15, 500)
init_approx_d = Normal(0, 20)

lower = -10.
upper = 10.
max_iter = 200

# example 1
@show true_d = TriangularDist(-10, 10, 0)
ll(x) = logpdf(true_d, x)

# include("run_laplacian_gradient_boosting.jl")
include("run_laplacian_gradient_boosting_exclusive.jl")
show()


println("Done.")
readline(STDIN)
