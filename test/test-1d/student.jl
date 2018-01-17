push!(LOAD_PATH, pwd()*"/../../src/")
using Distributions
using PyPlot
using BVB

x = linspace(-4,4,500)
max_iter = 90
lower = -10.
upper = 10.

init_approx_d = Normal(0.0, 10)

# example 1
@show true_d = TDist(1.0)
ll(x) = logpdf(true_d, x)

include("run_laplacian_gradient_boosting_exclusive.jl")
show()

println("Done.")
readline(STDIN)
