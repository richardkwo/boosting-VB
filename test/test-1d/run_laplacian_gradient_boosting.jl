approx_d = init_approx_d

α_vec = zeros(max_iter)
μ_vec = zeros(max_iter)

iter_vec = collect(1:max_iter)
eval_interval = 10
eval_on = map(i->(i%eval_interval==1), 1:max_iter)
eval_vec = iter_vec[eval_on]
cross_entropy = Vector{Float64}()
goodness_new_component = Vector{Float64}()

figure()
plot(x, exp(ll(x)), label="true")
plot(x, pdf(approx_d, x), label="init approx"); legend()


for iter in 1:max_iter
    if iter%30==0
        println(@sprintf "iter %d" iter)
    end
    # get new component by matching to functional gradient
    h = BVB.get_new_component_laplacian(ll, approx_d, verbose=true, n_init=10)
    α = BVB.search_α(approx_d, h, ll, n=10000, verbose=true)
    approx_d = BVB.add_to_mixture(approx_d, h, α)

    # tracing
    α_vec[iter] = α
    μ_vec[iter] = mean(h)
    if eval_on[iter]
        push!(cross_entropy, BVB.cross_entropy(approx_d, ll, n=100000))
        push!(goodness_new_component, BVB.goodness_of_component_criteria_1(h, ll, approx_d, lower_bound=true))
    end
end

figure()
plot(x, exp(ll(x)), label="true")
plot(x, pdf(init_approx_d, x), label="init approx")
plot(x, pdf(approx_d, x), label="approximation")
legend()

figure()
subplot(3,2,1)
BVB.plot_loss(eval_vec, cross_entropy, true_d)

subplot(3,2,2)
xlabel("iter"); ylabel("alpha")
vlines(iter_vec, 0, α_vec)

subplot(3,2,3)
plot(1:length(probs(approx_d)), sort(probs(approx_d), rev=true), ".-")
xlabel("component"); ylabel("weight")

subplot(3,2,4)
plot(iter_vec, μ_vec, ".")
xlabel("iter"); ylabel("mean of h")

subplot(3,2,5)
plot(eval_vec, goodness_new_component, ".-")
xlabel("iter"); ylabel("goodness of h")

subplot(3,2,6)
plot(x, ll(x) - logpdf(approx_d,x), "-")
axhline(y=0., ls="--", color="red")
xlabel("x"); ylabel("log L - log p")

figure()
BVB.plot_mixture_sequential(approx_d, x)
