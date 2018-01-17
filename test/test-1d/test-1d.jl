push!(LOAD_PATH, pwd()*"/../../src/")
using Distributions
using PyPlot
using BVB

# example 0
true_d = MixtureModel(Normal, [(5.0, 2), (10.0, 2), (20.0, 1), (-2, 1)], [1.0/3, 1.0/4, 1.0/4, 1.0/6])
# true_d = Gamma(2.0, 3.0)
ll(x) = logpdf(true_d, x)

init_approx_d =  Normal(0, 40)
x = linspace(-10,30,500)
max_iter = 50

figure()
subplot(3, 2, 1)
plot(x, exp(ll(x)), label="true")
plot(x, pdf(init_approx_d, x), label="init approx")
for (col, method) in enumerate(["inclusive", "exclusive"])
    approx_d, trace = BVB.infer(ll, init_approx_d, method=method, max_iter=max_iter,
                                eval_cross_entropy=true, eval_ELBO=true, show_plot=true)
    subplot(3, 2, 1)
    plot(x, pdf(approx_d, x), label=method); legend(loc=0)

    subplot(3, 2, 2)
    plot(trace["iter"], trace["cross_entropy"], ".-", label=method)
    xlabel("x"); ylabel("cross entropy"); legend(loc=0); grid("on")

    subplot(3, 2, 3)
    plot(trace["iter"], trace["ELBO"], ".-", label=method)
    xlabel("x"); ylabel("ELBO"); legend(loc=0); grid("on")

    subplot(3, 2, 4)
    vlines(collect(1:max_iter) + 0.5 * (col-1), 0., trace["α"], label=method, colors=(col==1 ? "blue" : "green"))
    xlabel("iter"); ylabel("alpha"); legend(loc=0); grid("on")

    subplot(3, 2, 5)
    plot(1:length(probs(approx_d)), sort(probs(approx_d), rev=true), ".-", label=method)
    xlabel("component"); ylabel("weight"); legend(loc=0); grid("on")

    subplot(3, 2, 6)
    plot(x, ll(x) - logpdf(approx_d,x), "-", label=method)
    axhline(y=0., ls="--", color="red")
    xlabel("x"); ylabel("log L - log p"); legend(loc=0); grid("on")
end

show()
