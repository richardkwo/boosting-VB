push!(LOAD_PATH, pwd()*"/../../src/")
using Distributions
using PyPlot
using BVB

x = linspace(-0.5,1.5,500)
max_iter = 200
init_approx_d = Normal(-1.0, 2)

# example 1
@show true_d = Exponential(1.0)
ll(x) = logpdf(true_d, x)

include("run_laplacian_gradient_boosting.jl")
show()

println("Done.")
readline(STDIN)
