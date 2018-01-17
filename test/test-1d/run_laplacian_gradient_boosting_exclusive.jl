using DataFrames

function save_trace(path, x, init_approx_d, approx_d, true_d,
                    iter_vec, α_vec,
                    eval_vec, KL_vec)
  pdf_init_approx = pdf(init_approx_d, x)
  pdf_approx = pdf(approx_d, x)
  pdf_true = pdf(true_d, x)
  df_pdf = DataFrame(x = x,
                     pdf_init = pdf(init_approx_d, x),
                     pdf_approx = pdf(approx_d, x),
                     pdf_true = pdf(true_d, x))
  mkpath(path)
  writetable(joinpath(path, "density.csv"), df_pdf)
  df_alpha = DataFrame(iter = iter_vec,
                       alpha = α_vec)
  writetable(joinpath(path, "alpha.csv"), df_alpha)
  df_KL = DataFrame(KL_iter = eval_vec, KL = KL_vec)
  writetable(joinpath(path, "KL.csv"), df_KL)
  println("Resuls saved to ", path)
end


approx_d = init_approx_d

α_vec = zeros(max_iter)
μ_vec = zeros(max_iter)

iter_vec = collect(1:max_iter)
eval_interval = 2
eval_on = map(i->(i%eval_interval==1), 1:max_iter)
eval_vec = iter_vec[eval_on]
elbo_vec = Vector{Float64}()

for iter in 1:max_iter
    if iter%30==0
        println(@sprintf "iter %d" iter)
    end
    # get new component by matching to functional gradient
    try
        h = BVB.get_new_component_laplacian(ll, approx_d,
                                        lower=lower, upper=upper, verbose=true)
    catch x
        println(x)
        continue
    end
    println(h)
    α = BVB.search_α_exclusive_sgd(approx_d, h, ll, n=100, verbose=true)
    approx_d = BVB.add_to_mixture(approx_d, h, α)

    # tracing
    α_vec[iter] = α
    μ_vec[iter] = mean(h)
    if eval_on[iter]
        push!(elbo_vec, BVB.ELBO(approx_d, ll))
    end
end

figure()
subplot(3,2,1)
plot(x, exp(ll(x)), label="true")
plot(x, pdf(init_approx_d, x), label="init approx")
plot(x, pdf(approx_d, x), label="approximation")

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
plot(x, ll(x) - logpdf(approx_d,x), "-")
axhline(y=0., ls="--", color="red")
xlabel("x"); ylabel("log L - log p")

subplot(3, 2, 6)
plot(eval_vec, -elbo_vec, ".-")
xlabel("iter"); ylabel("KL")

figure()
BVB.plot_mixture_sequential(approx_d, x)
