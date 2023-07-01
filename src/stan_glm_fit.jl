using Statistics, LinearAlgebra, Random, StatsBase, SpecialFunctions


function stan_glm_fit(x, y;
    weights = ones(length(y)),
    offset = zeros(length(y)),
    family = "gaussian",
    prior = default_prior_coef(family),
    prior_intercept = default_prior_intercept(family),
    prior_aux = exponential(autoscale = true),
    prior_smooth = exponential(autoscale = false),
    group = Dict(),
    prior_PD = false,
    algorithm = "sampling",
    mean_PPD = algorithm != "optimizing" && !prior_PD,
    adapt_delta = nothing,
    QR = false,
    sparse = false,
    importance_resampling = algorithm != "sampling",
    keep_every = algorithm != "sampling")

    if prior_ops != nothing
    prior, prior_intercept, prior_aux = support_deprecated_prior_options(prior, prior_intercept, prior_aux, prior_ops)
    prior_ops = nothing
    end

    algorithm = match_arg(algorithm)

    family = validate_family(family)
    supported_families = ["binomial", "gaussian", "Gamma", "inverse.gaussian",
            "poisson", "neg_binomial_2", "Beta regression"]
    fam = findfirst(isequal(family), supported_families)
    if fam == nothing
    supported_families_err = replace(supported_families, "Beta regression" => "mgcv::betar")
    throw(ArgumentError("'family' must be one of " * join(supported_families_err, ", ")))
    end

    supported_links = supported_glm_links(supported_families[fam])
    link = findfirst(isequal(family.link), supported_links)
    if link == nothing
        throw(ArgumentError("'link' must be one of " * join(supported_links, ", ")))
    end

    if binom_y_prop(y, family, weights)
        throw(ArgumentError("To specify 'y' as proportion of successes and 'weights' as number of trials please use stan_glm rather than calling stan_glm.fit directly."))
    end

    y = validate_glm_outcome_support(y, family)
    trials = nothing
    if is_binomial(family) && size(y, 2) == 2
        trials = [y[1] + y[2] for y in eachrow(y)]
        y = [y[1] for y in eachrow(y)]
        if length(y) == 1
            y = Array(y)
            trials = Array(trials)
        end
    end

    # useless assignments to pass R CMD check
    has_intercept = prior_df = prior_df_for_intercept = prior_df_for_aux = prior_df_for_smooth =
    prior_dist = prior_dist_for_intercept = prior_dist_for_aux = prior_dist_for_smooth =
    prior_mean = prior_mean_for_intercept = prior_mean_for_aux = prior_mean_for_smooth =
    prior_scale = prior_scale_for_intercept = prior_scale_for_aux = prior_scale_for_smooth =
    prior_autoscale = prior_autoscale_for_intercept = prior_autoscale_for_aux = 
    prior_autoscale_for_smooth = global_prior_scale = global_prior_df = slab_df = 
    slab_scale = nothing

    x_stuff = S = smooth_map = nothing
    if x isa Dict
        x_stuff = center_x(x[1], sparse)
        smooth_map = [j for j in 1:(length(x) - 1) for _ in 1:size(x[j+1], 2)]
        S = hcat(x[2:end]...)
    else
        x_stuff = center_x(x, sparse)
        S = Matrix{Float64}(undef, size(x, 1), 0)
        smooth_map = Int[]
    end
    xtemp, xbar, has_intercept = x_stuff["xtemp"], x_stuff["xbar"], x_stuff["has_intercept"]
    nvars = size(xtemp, 2)

    ok_dists = Dict("normal" => "normal", "student_t" => "t", "cauchy" => "cauchy", "hs" => "hs", "hs_plus" => "hs_plus", 
                    "laplace" => "laplace", "lasso" => "lasso", "product_normal" => "product_normal")
    ok_intercept_dists = ok_dists[1:3]
    ok_aux_dists = Dict("exponential" => "exponential", ok_dists[1:3]...)

    # prior distributions
    prior_stuff = handle_glm_prior(prior, nvars, link=family.link, default_scale=2.5, ok_dists=ok_dists)
    for (k, v) in prior_stuff
        eval(Meta.parse("$k = $v"))
    end

    m_y = 0
    if islist(prior_intercept) && prior_intercept["default"]
        if family == "gaussian" && family.link == "identity"
            m_y = mean(y) # y can be NULL if prior_PD=TRUE
        end
        prior_intercept["location"] = m_y
    end
    prior_intercept_stuff = handle_glm_prior(prior_intercept, nvars=1, default_scale=2.5, link=family.link, ok_dists=ok_intercept_dists)
    for (k, v) in prior_intercept_stuff
        eval(Meta.parse("$(k)_for_intercept = $v"))
    end

    prior_aux_stuff = handle_glm_prior(prior_aux, nvars=1, default_scale=1, link=nothing, ok_dists=ok_aux_dists)
    for (k, v) in prior_aux_stuff
        eval(Meta.parse("$(k)_for_aux = $v"))
    end

    if prior_aux == nothing
        if prior_PD
            throw(ArgumentError("'prior_aux' cannot be NULL if 'prior_PD' is TRUE."))
        end
        prior_aux_stuff["prior_scale_for_aux"] = Inf
    end
    for (i, v) in prior_aux_stuff
        eval(Meta.parse("$i = $v"))
    end

    if size(S, 2) > 0  # prior_{dist, mean, scale, df, dist_name, autoscale}_for_smooth
        prior_smooth_stuff = handle_glm_prior(prior_smooth, max(smooth_map), default_scale=1, link=nothing, ok_dists=ok_aux_dists)
        prior_smooth_stuff = Dict([(k * "_for_smooth", v) for (k, v) in prior_smooth_stuff])
        if prior_smooth == nothing
            if prior_PD
                throw(ArgumentError("'prior_smooth' cannot be NULL if 'prior_PD' is TRUE"))
            end
            prior_smooth_stuff["prior_scale_for_smooth"] = Inf
        end
        for (i, v) in prior_smooth_stuff
            eval(Meta.parse("$i = $v"))
        end
        prior_scale_for_smooth = Array(prior_scale_for_smooth)
    else
        prior_dist_for_smooth = 0
        prior_mean_for_smooth = Array{Float64}(undef, 0)
        prior_scale_for_smooth = Array{Float64}(undef, 0)
        prior_df_for_smooth = Array{Float64}(undef, 0)
    end

    famname = supported_families[fam]
    is_bernoulli = is_binomial(famname) && all(in(0:1), y) && trials == nothing
    is_nb = is_nb(famname)
    is_gaussian = is_gaussian(famname)
    is_gamma = is_gamma(famname)
    is_ig = is_ig(famname)
    is_beta = is_beta(famname)
    is_continuous = is_gaussian || is_gamma || is_ig || is_beta

    # require intercept for certain family and link combinations
    if !has_intercept
        linkname = supported_links[link]
        needs_intercept = !is_gaussian && linkname == "identity" ||
            is_gamma && linkname == "inverse" ||
            is_binomial(famname) && linkname == "log"
        if needs_intercept
            throw(ArgumentError("To use this combination of family and link, the model must have an intercept."))
        end
    end

    # allow prior_PD even if no y variable
    if y == nothing
        if !prior_PD
            throw(ArgumentError("Outcome variable must be specified if 'prior_PD' is not TRUE."))
        else
            y = fake_y_for_prior_PD(N = size(x, 1), family = family)
            if is_gaussian && 
                (prior_autoscale || prior_autoscale_for_intercept || prior_autoscale_for_aux)
                println("'y' not specified, will assume sd(y)=1 when calculating scaled prior(s).")
            end
        end
    end

    if is_gaussian
        ss = std(y)
        if prior_dist > 0 && prior_autoscale 
            prior_scale = ss * prior_scale
        end
        if prior_dist_for_intercept > 0 && prior_autoscale_for_intercept
            prior_scale_for_intercept = ss * prior_scale_for_intercept
        end
        if prior_dist_for_aux > 0 && prior_autoscale_for_aux
            prior_scale_for_aux = ss * prior_scale_for_aux
        end
    end
    if !QR && prior_dist > 0 && prior_autoscale
        min_prior_scale = 1e-12
        prior_scale = max.(min_prior_scale, prior_scale ./ 
                            [begin 
                                num_categories = length(unique(xi))
                                x_scale = if num_categories == 1
                                    1
                                else
                                    std(xi)
                                end
                                x_scale
                            end for xi in eachcol(xtemp)])
    end
    prior_scale = Array(min(floatmax(Float64), prior_scale))
    prior_scale_for_intercept = min(floatmax(Float64), prior_scale_for_intercept)

    if QR
        if size(xtemp, 2) <= 1
            throw(ArgumentError("'QR' can only be specified when there are multiple predictors."))
        end
        if sparse
            throw(ArgumentError("'QR' and 'sparse' cannot both be TRUE."))
        end
        cn = names(xtemp)
        decomposition = qr(xtemp)
        Q = Matrix(decomposition.Q)
        if prior_autoscale 
            scale_factor = sqrt(size(xtemp, 1) - 1)
        else 
            scale_factor = diag(decomposition.R)[end]
        end
        R_inv = inv(decomposition.R) * Q * scale_factor
        xtemp = Q * scale_factor
        xtemp.columns = cn
        xbar = xbar * R_inv
        if length(weights) > 0 && all(weights .== 1)
            weights = Float64[]
        end
        if length(offset) > 0 && all(offset .== 0)
            offset = Float64[]
        end

        standata = Dict(
            "N" => size(xtemp, 1),
            "K" => size(xtemp, 2),
            "xbar" => Array(xbar),
            "dense_X" => !sparse,
            "family" => stan_family_number(famname), # Placeholder function
            "link" => link,
            "has_weights" => length(weights) > 0,
            "has_offset" => length(offset) > 0,
            "has_intercept" => has_intercept,
            "prior_PD" => prior_PD,
            "compute_mean_PPD" => mean_PPD,
            "prior_dist" => prior_dist,
            "prior_mean" => prior_mean,
            "prior_scale" => prior_scale,
            "prior_df" => prior_df,
            "prior_dist_for_intercept" => prior_dist_for_intercept,
            "prior_scale_for_intercept" => [prior_scale_for_intercept],
            "prior_mean_for_intercept" => [prior_mean_for_intercept],
            "prior_df_for_intercept" => [prior_df_for_intercept],
            "global_prior_df" => global_prior_df,
            "global_prior_scale" => global_prior_scale,
            "slab_df" => slab_df,
            "slab_scale" => slab_scale,
            "z_dim" => 0,
            "link_phi" => 0,
            "betareg_z" => zeros(size(xtemp, 1), 0),
            "has_intercept_z" => 0,
            "zbar" => zeros(0),
            "prior_dist_z" => 0,
            "prior_mean_z" => Int64[],
            "prior_scale_z" => Int64[],
            "prior_df_z" => Int64[],
            "global_prior_scale_z" => 0,
            "global_prior_df_z" => 0,
            "prior_dist_for_intercept_z" => 0,
            "prior_mean_for_intercept_z" => 0,
            "prior_scale_for_intercept_z" => 0,
            "prior_df_for_intercept_z" => 0,
            "prior_dist_for_aux" => prior_dist_for_aux,
            "prior_dist_for_smooth" => prior_dist_for_smooth,
            "prior_mean_for_smooth" => prior_mean_for_smooth,
            "prior_scale_for_smooth" => prior_scale_for_smooth,
            "prior_df_for_smooth" => prior_df_for_smooth,
            "slab_df_z" => 0,
            "slab_scale_z" => 0,
            "num_normals" => (prior_dist == 7 ? [prior_df] : []),
            "num_normals_z" => Int64[],
            "clogit" => 0,
            "J" => 0,
            "strata" => Int64[]
        )
        # make a copy of user specification before modifying 'group' (used for keeping
        # track of priors)
        user_covariance = if !length(group) nothing else group["decov"]
        if length(group) > 0 && length(group["flist"]) > 0
        if length(group["strata"]) > 0
            standata["clogit"] = true
            standata["J"] = count_unique(group["strata"]) # assuming `count_unique` is a function you've defined
            standata["strata"] = vcat(convert(Array{Int64}, group["strata"][y .== 1]),
                                    convert(Array{Int64}, group["strata"][y .== 0]))
        end
        check_reTrms(group) # assuming `check_reTrms` is a function you've defined
        decov = group["decov"]
        if isnothing(group["SSfun"])
            standata["SSfun"] = 0
            standata["input"] = Float64[]
            standata["Dose"] = Float64[]
        else
            standata["SSfun"] = group["SSfun"]
            standata["input"] = group["input"]
            if group["SSfun"] == 5 
            standata["Dose"] = group["Dose"]
            else 
            standata["Dose"] = Float64[]
            end
        end
        Z = transpose(group["Zt"])
        group = pad_reTrms(Ztlist = group["Ztlist"],
                            cnms = group["cnms"],
                            flist = group["flist"]) # assuming `pad_reTrms` is a function you've defined
        Z = group["Z"]
        p = [length(x) for x in group["cnms"]]
        l = [count_unique(x) for x in group["flist"]] # assuming `count_unique` is a function you've defined
        t = length(l)
        b_nms = make_b_nms(group) # assuming `make_b_nms` is a function you've defined
        g_nms = [string(group["cnms"][i], "|", names(group["cnms"])[i]) for i in 1:t]
        standata["t"] = t
        standata["p"] = convert(Array{Int64}, p)
        standata["l"] = convert(Array{Int64}, l)
        standata["q"] = size(Z, 2)
        standata["len_theta_L"] = sum([binomial(x, 2) for x in p], p) # assuming `binomial` is a function you've defined
        if is_bernoulli # assuming `is_bernoulli` is a variable you've defined
            parts0 = extract_sparse_parts(Z[y .== 0, :]) # assuming `extract_sparse_parts` is a function you've defined
            parts1 = extract_sparse_parts(Z[y .== 1, :]) # assuming `extract_sparse_parts` is a function you've defined
            standata["num_non_zero"] = [length(parts0["w"]), length(parts1["w"])]
            standata["w0"] = convert(Array{Float64}, parts0["w"])
            standata["w1"] = convert(Array{Float64}, parts1["w"])
            standata["v0"] = convert(Array{Int64}, parts0["v"])
            standata["v1"] = convert(Array{Int64}, parts1["v"])
            standata["u0"] = convert(Array{Int64}, parts0["u"])
            standata["u1"] = convert(Array{Int64}, parts1["u"])
        else
            parts = extract_sparse_parts(Z) # assuming `extract_sparse_parts` is a function you've defined
            standata["num_non_zero"] = length(parts["w"])
            standata["w"] = parts["w"]
            standata["v"] = parts["v"]
            standata["u"] = parts["u"]
        end
        standata["shape"] = convert(Array{Float64}, maybe_broadcast(decov["shape"], t)) # assuming `maybe_broadcast` is a function you've defined
        standata["scale"] = convert(Array{Float64}, maybe_broadcast(decov["scale"], t)) # assuming `maybe_broadcast` is a function you've defined
        standata["len_concentration"] = sum(p[p .> 1])
        standata["concentration"] = 
            convert(Array{Float64}, maybe_broadcast(decov["concentration"], sum(p[p .> 1]))) # assuming `maybe_broadcast` is a function you've defined
        standata["len_regularization"] = sum(p .> 1)
        standata["regularization"] = 
            convert(Array{Float64}, maybe_broadcast(decov["regularization"], sum(p .> 1))) # assuming `maybe_broadcast` is a function you've defined
        standata["special_case"] = all([length(x) == 1 && x == "(Intercept)" for x in group["cnms"]])
        else # not multilevel
        if length(group) > 0
            standata["clogit"] = true
            standata["J"] = count_unique(group["strata"]) # assuming `count_unique` is a function you've defined
            standata["strata"] = vcat(convert(Array{Int64}, group["strata"][y .== 1]),
                                    convert(Array{Int64}, group["strata"][y .== 0]))
        end
        standata["t"] = 0
        standata["p"] = Int64[]
        standata["l"] = Int64[]
        standata["q"] = 0
        standata["len_theta_L"] = 0
        if is_bernoulli # assuming `is_bernoulli` is a variable you've defined
            standata["num_non_zero"] = [0, 0]
            standata["w0"] = standata["w1"] = Float64[]
            standata["v0"] = standata["v1"] = Int64[]
            standata["u0"] = standata["u1"] = Int64[]
        else
            standata["num_non_zero"] = 0
            standata["w"] = Float64[]
            standata["v"] = Int64[]
            standata["u"] = Int64[]
        end
        standata["special_case"] = 0
        standata["shape"] = standata["scale"] = standata["concentration"] =
            standata["regularization"] = Float64[]
        standata["len_concentration"] = 0
        standata["len_regularization"] = 0
        standata["SSfun"] = 0
        standata["input"] = Float64[]
        standata["Dose"] = Float64[]
        end

        if !is_bernoulli
            if sparse
              parts = extract_sparse_parts(xtemp)
              standata["nnz_X"] = length(parts["w"])
              standata["w_X"] = parts["w"]
              standata["v_X"] = parts["v"]
              standata["u_X"] = parts["u"]
              standata["X"] = zeros(Int64, 0, size(xtemp))
            else
              standata["X"] = reshape(xtemp, 1, size(xtemp)...)
              standata["nnz_X"] = 0
              standata["w_X"] = Float64[]
              standata["v_X"] = Int64[]
              standata["u_X"] = Int64[]
            end
            standata["y"] = y
            standata["weights"] = weights
            standata["offset_"] = offset
            standata["K_smooth"] = size(S, 2)
            standata["S"] = S
            standata["smooth_map"] = smooth_map
          end
          
          # call stan() to draw from posterior distribution
          if is_continuous
            standata["ub_y"] = Inf
            standata["lb_y"] = is_gaussian ? -Inf : 0
            standata["prior_scale_for_aux"] = isnothing(prior_scale_for_aux) ? 0 : prior_scale_for_aux
            standata["prior_df_for_aux"] = [prior_df_for_aux]
            standata["prior_mean_for_aux"] = [prior_mean_for_aux]
            standata["len_y"] = length(y)
            stanfit = stanmodels["continuous"]
          elseif isbinomial(famname)
            standata["prior_scale_for_aux"] = 
              (!length(group) || prior_scale_for_aux == Inf) ? 0 : prior_scale_for_aux
            standata["prior_mean_for_aux"] = 0
            standata["prior_df_for_aux"] = 0
            if is_bernoulli
              y0 = y .== 0
              y1 = y .== 1
              standata["N"] = [sum(y0), sum(y1)]
              if sparse
                standata["X0"] = zeros(Int64, 0, sum(y0), size(xtemp, 2))
                standata["X1"] = zeros(Int64, 0, sum(y1), size(xtemp, 2))
                parts0 = extract_sparse_parts(xtemp[y0, :])
                standata["nnz_X0"] = length(parts0["w"])
                standata["w_X0"] = parts0["w"]
                standata["v_X0"] = parts0["v"]
                standata["u_X0"] = parts0["u"]
                parts1 = extract_sparse_parts(xtemp[y1, :])
                standata["nnz_X1"] = length(parts1["w"])
                standata["w_X1"] = parts1["w"]
                standata["v_X1"] = parts1["v"]
                standata["u_X1"] = parts1["u"]
              else
                standata["X0"] = reshape(xtemp[y0, :], 1, sum(y0), size(xtemp, 2))
                standata["X1"] = reshape(xtemp[y1, :], 1, sum(y1), size(xtemp, 2))
                standata["nnz_X0"] = 0
                standata["w_X0"] = Float64[]
                standata["v_X0"] = Int64[]
                standata["u_X0"] = Int64[]
                standata["nnz_X1"] = 0
                standata["w_X1"] = Float64[]
                standata["v_X1"] = Int64[]
                standata["u_X1"] = Int64[]
              end
              if length(weights)
                # nocov start
                # this code is unused because weights are interpreted as number of 
                # trials for binomial glms
                standata["weights0"] = weights[y0]
                standata["weights1"] = weights[y1]
                # nocov end
              else
                standata["weights0"] = Float64[]
                standata["weights1"] = Float64[]
              end
              if length(offset)
                standata["offset0"] = offset[y0]
                standata["offset1"] = offset[y1]
              else
                standata["offset0"] = Float64[]
                standata["offset1"] = Float64[]
              end
              standata["K_smooth"] = size(S, 2)
              standata["S0"] = S[y0, :]
              standata["S1"] = S[y1, :]
              standata["smooth_map"] = smooth_map
              stanfit = stanmodels["bernoulli"]
            else
              standata["trials"] = trials
              stanfit = stanmodels["binomial"]
            end
          elseif ispoisson(famname)
            standata["prior_scale_for_aux"] = isnothing(prior_scale_for_aux) ? 0 : prior_scale_for_aux
            standata["prior_mean_for_aux"] = 0
            standata["prior_df_for_aux"] = 0
            stanfit = stanmodels["count"]
          elseif is_nb
            standata["prior_scale_for_aux"] = isnothing(prior_scale_for_aux) ? 0 : prior_scale_for_aux
            standata["prior_df_for_aux"] = [prior_df_for_aux]
            standata["prior_mean_for_aux"] = [prior_mean_for_aux]
            stanfit = stanmodels["count"]
          elseif is_gamma
            # nothing
          else
            error("$(famname) is not supported.")
          end
          prior_info = summarize_glm_prior(
            user_prior = prior_stuff,
            user_prior_intercept = prior_intercept_stuff,
            user_prior_aux = prior_aux_stuff,
            user_prior_covariance = user_covariance,
            has_intercept = has_intercept,
            has_predictors = nvars > 0,
            adjusted_prior_scale = prior_scale,
            adjusted_prior_intercept_scale = prior_scale_for_intercept,
            adjusted_prior_aux_scale = prior_scale_for_aux,
            family = family
        )
        
        pars = vcat(
            has_intercept ? ["alpha"] : [],
            ["beta"],
            size(S, 2) > 0 ? ["beta_smooth"] : [],
            length(group) > 0 ? ["b"] : [],
            (is_continuous | is_nb) ? ["aux"] : [],
            size(S, 2) > 0 ? ["smooth_sd"] : [],
            standata["len_theta_L"] > 0 ? ["theta_L"] : [],
            (mean_PPD && !standata["clogit"]) ? ["mean_PPD"] : []
        )
        
        if algorithm == "optimizing"
            optimizing_args = Dict{Symbol, Any}()
            optimizing_args[:draws] = get(optimizing_args, :draws, 1000)
            optimizing_args[:object] = stanfit
            optimizing_args[:data] = standata
            optimizing_args[:constrained] = true
            optimizing_args[:importance_resampling] = importance_resampling
            optimizing_args[:tol_rel_grad] = get(optimizing_args, :tol_rel_grad, 10000)
            out = optimizing(optimizing_args...)
            check_stanfit(out)
            if optimizing_args[:draws] == 0
              out["theta_tilde"] = out["par"]
              size(out["theta_tilde"]) = (1, length(out["par"]))
            end
            new_names = keys(out["par"])
            mark = occursin.(r"^beta\\[[[:digit:]]+\\]$", new_names)
            if QR
              out["par"][mark] = R_inv * out["par"][mark]
              out["theta_tilde"][:,mark] .= out["theta_tilde"][:, mark] * transpose(R_inv)
            end
            new_names[mark] = names(xtemp)
            if size(S, 2) > 0
              mark = occursin.(r"^beta_smooth\\[[[:digit:]]+\\]$", new_names)
              new_names[mark] = names(S)
            end
            new_names[new_names .== "alpha[1]"] .= "(Intercept)"
            new_names[occursin.(r"aux(\\[1\\])?$", new_names)] = 
              if is_gaussian 
                "sigma" 
              elseif is_gamma 
                "shape" 
              elseif is_ig 
                "lambda" 
              elseif is_nb 
                "reciprocal_dispersion" 
              elseif is_beta 
                "(phi)" 
              else 
                missing 
              end
            out["par"] = Dict(new_names .=> values(out["par"]))
            names(out["theta_tilde"]) = new_names
        end
        if optimizing_args["draws"] > 0 && importance_resampling
            lr = out["log_p"] - out["log_g"]
            lr[lr .== -Inf] .= -800
            p = psis(lr, r_eff = 1) # Assuming the psis function handles warnings internally.
            p["log_weights"] = p["log_weights"] - log_sum_exp(p["log_weights"])
            theta_pareto_k = map(out["theta_tilde"]) do col
                if all(isfinite, col)
                    psis(log1p(col.^2) / 2 + lr, r_eff = 1)["diagnostics"]["pareto_k"]
                else
                    NaN
                end
            end
            if p["diagnostics"]["pareto_k"] > 1
                @warn("Pareto k diagnostic value is $(round(p["diagnostics"]["pareto_k"], digits = 2)).
                        Resampling is disabled. Decreasing tol_rel_grad may help if optimization has terminated prematurely.
                        Otherwise consider using sampling.")
                importance_resampling = false
            elseif p["diagnostics"]["pareto_k"] > 0.7
                @warn("Pareto k diagnostic value is $(round(p["diagnostics"]["pareto_k"], digits = 2)).
                        Resampling is unreliable. Increasing the number of draws or decreasing tol_rel_grad may help.")
            end
            out["psis"] = Dict("pareto_k" => p["diagnostics"]["pareto_k"],
                               "n_eff" => p["diagnostics"]["n_eff"] / keep_every)
        else
            theta_pareto_k = fill(NaN, length(new_names))
            importance_resampling = false
        end
        
        if importance_resampling
            ir_idx = sample_indices(exp.(p["log_weights"]), n_draws = ceil(Int, optimizing_args["draws"] / keep_every))
            out["theta_tilde"] = out["theta_tilde"][ir_idx, :]
            out["ir_idx"] = ir_idx
            w_sir = counts(ir_idx) / length(ir_idx)
            mcse = map(out["theta_tilde"][unique(ir_idx), :]) do col
                if all(isfinite, col)
                    sqrt(sum(w_sir.^2 .* (col .- mean(col)).^2))
                else
                    NaN
                end
            end
            n_eff = round.(var(out["theta_tilde"][unique(ir_idx), :]) ./ (mcse.^2), digits=0)
        else
            out["ir_idx"] = nothing
            mcse = fill(NaN, length(theta_pareto_k))
            n_eff = fill(NaN, length(theta_pareto_k))
        end
        out["diagnostics"] = DataFrame(mcse=mcse, theta_pareto_k=theta_pareto_k, n_eff=n_eff)
        
        out["stanfit"] = sampling(stanfit, data=standata, chains=0)
        
        out
        
        if algorithm == "sampling"
            sampling_args = set_sampling_args(
                object = stanfit, 
                prior = prior, 
                user_adapt_delta = adapt_delta, 
                data = standata, 
                pars = pars, 
                show_messages = false)
            stanfit = sampling(sampling_args...)
          else
            vb_args = Dict{Symbol, Any}()
            vb_args[:output_samples] = get(vb_args, :output_samples, 1000)
            vb_args[:tol_rel_obj] = get(vb_args, :tol_rel_obj, 1e-4)
            vb_args[:keep_every] = get(vb_args, :keep_every, keep_every)
            vb_args[:object] = stanfit
            vb_args[:data] = standata
            vb_args[:pars] = pars
            vb_args[:algorithm] = algorithm
            vb_args[:importance_resampling] = importance_resampling
            stanfit = vb(vb_args...)
            if !QR && standata["K"] > 1
                recommend_QR_for_vb()
            end
          end

        check = try check_stanfit(stanfit) # Assuming there's an equivalent 'try' function in Julia
        if !isTRUE(check)
            return standata
        end
        
        if QR
            thetas = extract(stanfit, pars = "beta", inc_warmup = true, permuted = false)
            betas = [R_inv * theta for theta in thetas]
            end_ = last(size(betas))
            for chain in 1:end_, param in 1:size(betas, 1)
                # This requires a way to index into 'stanfit' in the same way as the original R code
                stanfit["sim"]["samples"][chain][has_intercept + param] =
                    (size(xtemp, 2) > 1) ? betas[param, :, chain] : betas[param, chain]
            end
        end
        
        if standata["len_theta_L"]
            # This block involves function calls and operations that might not directly translate to Julia.
            # You might need to find equivalent Julia functions or write custom ones.
        end
        
        new_names = vcat(
            ifelse(has_intercept, ["(Intercept)"], []),
            names(xtemp),
            ifelse(size(S, 2), names(S), []),
            # Additional elements are conditionally added to 'new_names' here.
            # These conditional blocks involve checking the type of the model, which is probably represented differently in Julia.
        )
        
        # This requires a way to index into 'stanfit' in the same way as the original R code
        stanfit["sim"]["fnames_oi"] = new_names
        
        return Dict("stanfit" => stanfit, "prior_info" => prior_info, "dropped_cols" => x_stuff["dropped_cols"])
        end
end


function match_arg(input, options = ["sampling", "optimizing", "meanfield", "fullrank"])
if input ∉ options
throw(ArgumentError("Invalid algorithm. Choices are: " * join(options, ", ")))
end
return input
end
