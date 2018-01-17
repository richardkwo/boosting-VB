push!(LOAD_PATH, pwd()*"/../../src/")
include("utils-logit.jl")

using Distributions
using PyPlot
using RDatasets
using DataFrames
using BVB
using Optim: optimize, LBFGS

df = dataset("boot", "nodal")
y = convert(Vector{Float64}, df[:R])
y = 2 * y - 1.
X = convert(Matrix{Float64}, delete!(df, :R))
@show n, p = size(X)

@show λ = 1.
ll(β) = ll_logistic(β, y, X; λ=λ)
approx_d = MvNormal(zeros(p), 100.0 * eye(p))

@show max_iter = 30
α_vec = zeros(max_iter)

iter_vec = collect(1:max_iter)
eval_interval = 2
eval_on = map(i->(i%eval_interval==1), iter_vec)
eval_vec = Vector{Int64}()
elbo_vec = Vector{Real}()

for iter in 1:max_iter
    if iter%20==0
        println(@sprintf "iter %d" iter)
    end
    # get new component by matching to functional gradient
    h = BVB.get_new_component_laplacian(ll, approx_d, diagonal=true, verbose=false)
    println("\niter $iter")
    @show h.μ
    @show h.Σ.mat
    # get rid of bad h
    all(isfinite.(h.μ)) && all(isfinite.(h.Σ.mat)) || continue
    # α = BVB.search_α(approx_d, h, ll, n=10000, verbose=true)
    α = BVB.search_α_exclusive_sgd(approx_d, h, ll, n=100, min_iter=20, verbose=true)
    approx_d = BVB.add_to_mixture(approx_d, h, α)

    # tracing
    α_vec[iter] = α
    if eval_on[iter]
        push!(eval_vec, iter)
        push!(elbo_vec, BVB.ELBO(approx_d, ll; n=10000))
        println(@sprintf("ELBO (Monte Carlo) = %g", elbo_vec[end]))
    end
end

figure()
subplot(2,2,1)
plot(eval_vec, elbo_vec)
xlabel("iter"); ylabel("ELBO")

subplot(2,2,2)
xlabel("iter"); ylabel("alpha")
vlines(iter_vec, 0, α_vec)

subplot(2,2,3)
plot(1:length(probs(approx_d)), sort(probs(approx_d), rev=true), ".-")
xlabel("component"); ylabel("weight")

writetable("bayes-logit-nodal-elbo.csv",
           DataFrame(iter = eval_vec, elbo = elbo_vec))
writetable("bayes-logit-nodal-alpha.csv",
           DataFrame(iter = 1:max_iter, elbo = α_vec))
@show writedlm("./bayes-logit-nodal.out", rand(approx_d, 4000)')

show()
println("Done.")
