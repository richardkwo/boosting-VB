"""
BVB Inference, general purpose.
* input
    * ll: log-likelihood, should return a float number for a vector, or a vector
          for a matrix (each column corresponds to a vector of parameter)
    * init_approx_d: a Distribution, initial approximation, usually set to be a flat
                     Normal() or MvNormal()
    * method: "inclusive" (default) or "exclusive"
    * max_iter: 100 (default)
    * n: number of MC samples used to search α, for approximating integral in Newton's method
    * verbose: >=0, more display with higher verbose
    * eval_interval: for every such # of rounds, evaluate cross_entropy and ELBO as trace information
    * eval_cross_entropy: true or false (default)
    * n_eval_cross_entropy: number of samples for evaluating cross_entropy (default: 100000)
    * eval_ELBO: true or false (default)
    * n_eval_ELBO: number of samples for evaluating cross_entropy (default: 100000)
    * show_plot: show unicode plots if true (default: false)
    * opt_grtol: gradient norm tol for optimization to terminate (default: 1e-8).
                 Can be very slow if set too small.
    * opt_max_iter: max iter allowed for optimization (default: 1000)
    * opt_n_init: pick Θ with maximum L/q from such number of samples ~ q (default: 1, just randomly initialize)
    * opt_show_trace: show trace of LBFGS if true (default: false)
    * ...: extra functions of approx_d that is called every `eval_interval`, traced and reported
* return
    * approx_d: as a MixtureModel
    * trace: a Dict storing trace information
"""
function infer(ll::Function, init_approx_d::Distribution; method="exclusive",
               max_iter=100, n=100, verbose=1, eval_interval=10,
               diagonal = false,
               eval_cross_entropy=false, eval_ELBO=false, show_plot=false,
               opt_max_iter=1000, sgd_max_iter=1000, opt_show_trace=false,
               n_eval_cross_entropy=100000, n_eval_ELBO=100000, kwargs...)
    iter = 0
    p = length(init_approx_d)
    @assert method == "inclusive" || method == "exclusive"
    println("\nInferring $p parameters with $method KL for $max_iter iterations.")
    println("Using $n MC samples for searching α.")
    trace_eval_iter = Vector{Int64}()
    trace_cross_entropy = Vector{Float64}()
    trace_ELBO = Vector{Float64}()
    trace_α = Vector{Float64}()
    trace_docall = Dict{Any, Vector}()
    if !isempty(kwargs)
        println("Running with extra reporting of $kwargs")
        for (name, do_call_f) in kwargs
            isa(do_call_f, Function) || error("Wrong keyword argument: $name is not a function.")
            trace_docall[name] = Vector{Float64}()
        end
    end

    approx_d = init_approx_d

    while iter < max_iter
        iter += 1
        println("... Iter $iter ...")
        # get new component
        h = try
                BVB.get_new_component_laplacian(ll, approx_d;
                                                verbose=(verbose>2),
                                                diagonal = diagonal,
                                                max_iter=opt_max_iter,
                                                show_trace=opt_show_trace)
            catch error_msg
                warn("Error in getting h, retrying: ", error_msg)
                continue
            end
        if show_plot && p>1
            fig = scatterplot(1:p, mean(h), color=:red, name="h", title="mean")
            if isa(approx_d, MixtureModel)
                scatterplot!(fig, 1:p, mean(approx_d.components[end]), color=:blue, name="last h")
            else
                scatterplot!(fig, 1:p, mean(approx_d), color=:blue, name="last h")
            end
            print(fig)
        end
        # determine α
        if method == "inclusive"
            α = BVB.search_α(approx_d, h, ll;
                            n=n, verbose=(verbose>0), show_plot=show_plot)
        else
            α = BVB.search_α_exclusive_sgd(approx_d, h, ll;
                                              n=n, verbose=(verbose>0), max_iter=sgd_max_iter)
        end
        # add new component
        approx_d = BVB.add_to_mixture(approx_d, h, α)
        # evaluate trace information
        push!(trace_α, α)
        if show_plot
            print(barplot(collect(1:length(trace_α)), trace_α, title="trace of α"))
        end
        if eval_interval == 1 || iter % eval_interval == 1
            push!(trace_eval_iter, iter)
            if eval_cross_entropy
                if verbose > 1
                    println("Computing cross entropy...")
                end
                push!(trace_cross_entropy, BVB.cross_entropy(approx_d, ll, n=n_eval_cross_entropy))
                if verbose > 0
                    println(@sprintf "iter %4d: cross entropy = %g" iter trace_cross_entropy[end])
                end
            end
            if eval_ELBO
                if verbose > 1
                    println("Computing ELBO...")
                end
                push!(trace_ELBO, BVB.ELBO(approx_d, ll, n=n_eval_ELBO))
                if verbose > 0
                    println(@sprintf "iter %4d: ELBO = %g" iter trace_ELBO[end])
                end
            end
            # extra do_call funtions
            for (name, do_call_f) in kwargs
                s_f = do_call_f(approx_d)
                println("Iter $iter: $name => $s_f")
                push!(trace_docall[name], s_f)
            end
            if show_plot
                _lineplot_trace(trace_docall)
            end
        end
    end

    trace = Dict{Any, Any}()
    trace["method"] = method
    trace["α"] = trace_α
    trace["iter"] = trace_eval_iter
    trace["cross_entropy"] = trace_cross_entropy
    trace["ELBO"] = trace_ELBO
    trace["max_iter"] = max_iter
    trace["n"] = n
    for name in keys(trace_docall)
        trace[name] = trace_docall[name]
    end

    return approx_d, trace
end

"""
BVB Inference, for mini-batch estimated log-likelihood.
* input
    * ll_factory: each call of ll_factory() will generate an ll(...) function, as
                  stochastic approximator of full-data ll.
    * init_approx_d: a Distribution, initial approximation, usually set to be a flat
                     Normal() or MvNormal()
    * learning_rate: a function of iter (default: t -> 1.0/t). α_t = learning_rate(t) x (α_t∗)
    * method: "inclusive" or "exclusive" (default)
    * max_iter: 100 (default)
    * n: number of MC samples used to search α, for approximating integral in Newton's method
    * first_n: number of MC samples used in the first round (default to n)
    * verbose: >=0, more display with higher verbose
    * eval_interval: for every such # of rounds, evaluate cross_entropy and ELBO as trace information
    * eval_cross_entropy: true or false (default)
    * n_eval_cross_entropy: number of samples for evaluating cross_entropy (default: 100000)
    * eval_ELBO: true or false (default)
    * n_eval_ELBO: number of samples for evaluating cross_entropy (default: 100000)
    * show_plot: show unicode plots if true (default: false)
    * opt_grtol: gradient norm tol for optimization to terminate (default: 1e-8).
                 Can be very slow if set too small.
    * opt_max_iter: max iter allowed for optimization (default: 1000)
    * opt_n_init: pick Θ with maximum L/q from such number of samples ~ q (default: 1, just randomly initialize)
    * opt_show_trace: show trace of LBFGS if true (default: false)
    * ...: extra functions of approx_d that is called every `eval_interval`, traced and reported
* return
    * approx_d: as a MixtureModel
    * trace: a Dict storing trace information
"""
# TODO(guo): make this work
function minibatch_infer(ll_factory::Function, init_approx_d::Distribution, learning_rate::Function=(t->1.0/t);
               method="exclusive",
               max_iter=100, n=50000, first_n=n, verbose=1, eval_interval=10,
               eval_cross_entropy=false, eval_ELBO=false,
               n_eval_cross_entropy=100000, n_eval_ELBO=100000,
               show_plot=false,
               opt_grtol=1e-8, opt_max_iter=1000, opt_n_init=1, opt_show_trace=false,
               kwargs...)
    iter = 0
    p = length(init_approx_d)
    @assert method == "inclusive" || method == "exclusive"
    println("\nMini-batched inferring $p parameters with $method KL for $max_iter iterations.")
    println("Using $n samples for searching α.")
    trace_eval_iter = Vector{Int64}()
    trace_cross_entropy = Vector{Float64}()
    trace_ELBO = Vector{Float64}()
    trace_α = Vector{Float64}()
    trace_α_effective = Vector{Float64}()
    trace_docall = Dict{Any, Vector}()
    if !isempty(kwargs)
        println("Running with extra reporting of $kwargs")
        for (name, do_call_f) in kwargs
            isa(do_call_f, Function) || error("Wrong keyword argument: $name is not a function.")
            trace_docall[name] = Vector{Float64}()
        end
    end

    approx_d = init_approx_d

    while iter < max_iter
        iter += 1
        println("... Iter $iter ...")
        # renew stochastic approximation of ll
        ll = ll_factory()
        # get new component
        h = try
                BVB.get_new_component_laplacian(ll, approx_d, verbose=(verbose>2),
                                                n_init=opt_n_init, grtol=opt_grtol,
                                                max_iter=opt_max_iter, show_trace=opt_show_trace)
            catch error_msg
                println("Retrying: get h", error_msg)
                iter -= 1
                continue
            end
        if show_plot && p>1
            fig = scatterplot(1:p, mean(h), color=:red, name="h", title="mean")
            if isa(approx_d, MixtureModel)
                scatterplot!(fig, 1:p, mean(approx_d.components[end]), color=:blue, name="last h")
            else
                scatterplot!(fig, 1:p, mean(approx_d), color=:blue, name="last h")
            end
            print(fig)
        end
        # determine α
        if method == "inclusive"
            α = BVB.search_α(approx_d, h, ll,
                             n=(isa(approx_d, MixtureModel) ? n : first_n),
                             verbose=(verbose>0), show_plot=show_plot)
        else
            α = BVB.search_α_exclusive_newton(approx_d, h, ll,
                                              n=(isa(approx_d, MixtureModel) ? n : first_n),
                                              verbose=(verbose>0), show_plot=show_plot)
        end
        # add new component
        lr = learning_rate(1.0 * iter)
        α_effective = α * lr
        if verbose > 0
            println("effective α <- (α = $α) x (lr = $lr) = $α_effective")
        end
        approx_d = BVB.add_to_mixture(approx_d, h, α_effective)
        # evaluate trace information
        push!(trace_α, α)
        push!(trace_α_effective, α_effective)
        if show_plot
            print(barplot(collect(1:length(trace_α)), trace_α, title="trace of α"))
            print(barplot(collect(1:length(trace_α_effective)), trace_α_effective, title="trace of effective α"))
        end
        if eval_interval==1 || iter % eval_interval == 1
            push!(trace_eval_iter, iter)
            if eval_cross_entropy
                if verbose > 1
                    println("Computing cross entropy...")
                end
                push!(trace_cross_entropy, BVB.cross_entropy(approx_d, ll, n=n_eval_cross_entropy))
                if verbose > 0
                    println(@sprintf "iter %4d: cross entropy = %g" iter trace_cross_entropy[end])
                end
            end
            if eval_ELBO
                if verbose > 1
                    println("Computing ELBO...")
                end
                push!(trace_ELBO, BVB.ELBO(approx_d, ll, n=n_eval_ELBO))
                if verbose > 0
                    println(@sprintf "iter %4d: ELBO = %g" iter trace_ELBO[end])
                end
            end
            # extra do_call funtions
            for (name, do_call_f) in kwargs
                s_f = do_call_f(approx_d)
                println("Iter $iter: $name => $s_f")
                push!(trace_docall[name], s_f)
            end
            if show_plot
                _lineplot_trace(trace_docall)
            end
        end
    end

    trace = Dict{Any, Any}()
    trace["method"] = method
    trace["α"] = trace_α
    trace["iter"] = trace_eval_iter
    trace["cross_entropy"] = trace_cross_entropy
    trace["ELBO"] = trace_ELBO
    trace["max_iter"] = max_iter
    trace["n"] = n
    for name in keys(trace_docall)
        trace[name] = trace_docall[name]
    end

    return approx_d, trace
end

function _lineplot_trace(trace::Dict)
    if !isempty(trace)
        for name in keys(trace)
            info = trace[name]
            fig = lineplot(1:length(info), info, title="extra trace", name=string(name))
            print(fig)
        end
    end
end
