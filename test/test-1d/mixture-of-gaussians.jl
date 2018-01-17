push!(LOAD_PATH, pwd()*"/../../src/")
using Distributions
using PyPlot
using BVB

lower = -100.
upper = 100.

max_iter = 50

# example 0
true_d = MixtureModel(Normal, [(5.0, 2), (10.0, 2), (20.0, 1), (-2, 1)], [1.0/3, 1.0/4, 1.0/4, 1.0/6])
ll(x) = logpdf(true_d, x)

init_approx_d =  Normal(0, 40)
x = linspace(-10,30,500)


include("run_laplacian_gradient_boosting_exclusive.jl")
show()

# example 1
true_d = MixtureModel(Normal, [(-2.0, 1), (2.0, 1)], [0.5, 0.5])
ll(x) = logpdf(true_d, x)

init_approx_d = Normal(0, 20)
x = linspace(-10,10,500)

include("run_laplacian_gradient_boosting_exclusive.jl")
show()

println("Done.")

# example 2
true_d = MixtureModel(Normal, [(-3.5, 1), (-1.0, 0.4), (1.0, 0.5), (6.0, 1)], [0.2, 0.3, 0.3, 0.2])
ll(x) = logpdf(true_d, x)
x = linspace(-10,10,500)

include("run_laplacian_gradient_boosting_exclusive.jl")
show()

println("Done.")
