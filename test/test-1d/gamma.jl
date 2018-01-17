push!(LOAD_PATH, pwd()*"/../../src/")
using Distributions
using PyPlot
using BVB

true_d = Gamma(2.0, 3.0)
ll(x) = logpdf(true_d, x)
x = linspace(-5,30,500)

init_approx_d = Normal(0, 20)

max_iter = 300

# include("run_laplacian_gradient_boosting.jl")
include("run_laplacian_gradient_boosting_exclusive.jl")
show()

println("Done.")
readline(STDIN)
