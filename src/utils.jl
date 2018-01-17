export plot_mixture_by_component, plot_mixture_sequential

using PyPlot

function plot_mixture_by_component(d::AbstractMixtureModel, x; lw=1, show_legend=false)
    @assert length(d) == 1
    K = length(probs(d))
    cmap = ColorMap("bwr")
    for (i, j) in enumerate(sort(collect(1:K), by = (x -> probs(d)[x]), rev=true))
        plot(x, probs(d)[j] * pdf(d.components[j], x), color=cmap.o(i/K*0.45),
        lw=lw,
        label=@sprintf("component %d: p=%.1f", i, probs(d)[i]))
    end
    plot(x, pdf(d, x), lw=1, ls=":", color="red", alpha=0.7, label="overall")
    if show_legend
        legend()
    end
end

function plot_mixture_sequential(d::AbstractMixtureModel, x; lw=1, rescaled=true)
    @assert length(d) == 1
    K = length(probs(d))
    cmap = ColorMap("Blues")
    subplot(2,1,1)
    for k in 1:K
        if probs(d)[k] < 1e-3
            continue
        end
        priors = probs(d)[1:k]
        w_sum = sum(priors)
        priors = priors / w_sum
        sub_d = MixtureModel(d.components[1:k], priors)
        if rescaled
            plot(x, pdf(sub_d, x), lw=lw, color=cmap.o(w_sum))
        else
            plot(x, w_sum * pdf(sub_d, x), lw=lw, color=cmap.o(w_sum))
        end
    end
    xlabel("x"); ylabel("density")
    subplot(2,1,2)
    w = zeros(K)
    for k in 1:K
        w[k] = k==1 ? probs(d)[k] : probs(d)[k] + w[k-1]
        plot(k, w[k], ".", ms=10, color=cmap.o(w[k]))
    end
    plot(collect(1:K), w, "--", color="blue", alpha=0.5)
    xlabel("component added"); ylabel("sum of p")
end

"plot fitness score vs. iterations"
function plot_loss(iter_vec, loss_vec::Vector{Float64}; color="blue")
    plot(iter_vec, loss_vec, ".-", color=color)
    xlabel("iteration"); ylabel("cross entropy"); grid()
end

"plot fitness score vs. iterations, with best possible fitness score"
function plot_loss{T<:Integer}(iter_vec::AbstractVector{T}, loss_vec::Vector{Float64}, true_d::Distribution; color="blue")
    plot_loss(iter_vec, loss_vec, color=color)
    axhline(y=self_entropy(true_d), ls="--", color="red")
end

function plot_loss(loss_vec::Vector{Float64}, true_d::Distribution; color="blue")
    plot_loss(1:length(loss_vec), loss_vec, true_d, color=color)
end



"generate a meshgrid and evaluate f(.) on each point"
function ndgrid_eval{T}(v1::AbstractVector{T}, v2::AbstractVector{T}, f::Function)
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    X = repmat(v1, 1, n)
    Y = repmat(v2, m, 1)
    Z = zeros(m, n)
    for i in 1:m
        for j in 1:n
            Z[i, j] = f([X[i,j], Y[i,j]])
        end
    end
    return (X, Y, Z)
end

"plot density in 3D"
function plot_density_3D{T}(d::Distribution, x::AbstractVector{T}=linspace(-3,3,128), y::AbstractVector{T}=linspace(-3,3,128))
    X, Y, Z = ndgrid_eval(x, y, z -> pdf(d, z))
    cmap = ColorMap("bwr")
    subplot(2,2,1, projection="3d")
    plot_surface(X, Y, Z, alpha=0.3)
    xlabel("x"); ylabel("y");
    subplot(2,2,4)
    contour(X, Y, Z); grid()
    xlabel("x"); ylabel("y")
    subplot(2,2,2)
    contour(Z, Y, X); grid()
    ylabel("y")
    subplot(2,2,3)
    contour(X, Z, Y); grid()
    xlabel("x")

end

"plot and compare 2D distributions"
function plot_compare_3D{T}(true_d::Distribution, approx_d::Distribution,
                            x::AbstractVector{T}, y::AbstractVector{T}; n=1000)
    X, Y, Z1 = ndgrid_eval(x, y, z -> pdf(true_d, z))
    X, Y, Z2 = ndgrid_eval(x, y, z -> pdf(approx_d, z))

    figure()
    plot_surface(X, Y, Z1, rstride=8, cstride=8, alpha=0.3, color="blue")
    plot_surface(X, Y, Z2, rstride=8, cstride=8, alpha=0.3, color="green")
    xlabel("x"); ylabel("y"); zlabel("density")

    figure()
    samples_true = rand(true_d, n)
    samples_approx = rand(approx_d, n)
    plot(samples_true[1,:][:], samples_true[2,:][:], color="blue", alpha=0.3, ".", label="true")
    plot(samples_approx[1,:][:], samples_approx[2,:][:], color="green", alpha=0.3, ".", label="approx")
    legend()
end

"read libsvm format"
function read_libsvm_format(filename::AbstractString, n::Int, p::Int)
    y = Vector{Int}(n)
    X = zeros(n, p)
    open(filename) do fr
        iter = 0
        for line in eachline(fr)
            iter += 1
            x = zeros(p)
            for (i,s) in enumerate(split(rstrip(line)))
                if i==1
                    y[iter] = parse(Int, s)
                else
                    j, v = split(s, ":")
                    j = parse(Int, j)
                    v = parse(Float64, v)
                    X[iter, j] = v
                end
            end
            if iter==n
                break
            end
        end
    end
    @show size(X), size(y)
    return X, y
end
