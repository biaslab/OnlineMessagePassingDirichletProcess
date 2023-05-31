### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ a1c1ef76-64d9-4f7b-b6b8-b70e180fc0ff
using RxInfer, Distributions, PlutoUI, PyPlot, PGFPlotsX, LaTeXStrings, Random, LinearAlgebra

# ╔═╡ 7c5873ce-a0d5-46f2-9d59-188ffd09cb5b
begin
	using SpecialFunctions
	@rule Categorical(:p, Marginalisation) (m_out::Categorical, q_out::PointMass) = begin
	    @logscale -SpecialFunctions.logfactorial(length(probvec(q_out)))
	    return Dirichlet(probvec(q_out) .+ one(eltype(probvec(q_out))))
	end

	struct EnforceMarginalFunctionalDependency <: ReactiveMP.AbstractNodeFunctionalDependenciesPipeline
	    edge :: Symbol
	end
	
	function ReactiveMP.message_dependencies(::EnforceMarginalFunctionalDependency, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
	    return ReactiveMP.message_dependencies(ReactiveMP.DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
	end
	
	function ReactiveMP.marginal_dependencies(enforce::EnforceMarginalFunctionalDependency, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
	    default = ReactiveMP.marginal_dependencies(ReactiveMP.DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
	    index   = ReactiveMP.findnext(i -> name(i) === enforce.edge, nodeinterfaces, 1)
	    if index === iindex 
	        return default
	    end
	    vmarginal = ReactiveMP.getmarginal(ReactiveMP.connectedvar(nodeinterfaces[index]), IncludeAll())
	    loc = ReactiveMP.FactorNodeLocalMarginal(-1, index, enforce.edge)
	    ReactiveMP.setstream!(loc, vmarginal)
	    # Find insertion position (probably might be implemented more efficiently)
	    insertafter = sum(first(el) < iindex ? 1 : 0 for el in default; init = 0)
	    return ReactiveMP.TupleTools.insertafter(default, insertafter, (loc, ))
	end

	# function for using hard switching
function ReactiveMP.functional_dependencies(::EnforceMarginalFunctionalDependency, factornode::MixtureNode{N, F}, iindex::Int) where {N, F <: FullFactorisation}
    message_dependencies = if iindex === 1
        # output depends on:
        (factornode.inputs,)
    elseif iindex === 2
        # switch depends on:
        (factornode.out, factornode.inputs)
    elseif 2 < iindex <= N + 2
        # k'th input depends on:
        (factornode.out, )
    else
        error("Bad index in functional_dependencies for SwitchNode")
    end

    marginal_dependencies = if iindex === 1
        # output depends on:
        (factornode.switch,)
    elseif iindex == 2
        #  switch depends on
        ()
    elseif 2 < iindex <= N + 2
        # k'th input depends on:
        (factornode.switch,)
    else
        error("Bad index in function_dependencies for SwitchNode")
    end
    # println(marginal_dependencies)
    return message_dependencies, marginal_dependencies
end

# create an observable that is used to compute the switch with pipeline constraints
function ReactiveMP.get_messages_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{ReactiveMP.NodeInterface, NTuple{N, ReactiveMP.IndexedNodeInterface}}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    switchinterface  = messages[1]
    inputsinterfaces = messages[2]

    msgs_names = Val{(name(switchinterface), name(inputsinterfaces[1]))}
    msgs_observable =
    combineLatest((ReactiveMP.messagein(switchinterface), combineLatest(map((input) -> ReactiveMP.messagein(input), inputsinterfaces), PushNew())), PushNew()) |>
        map_to((ReactiveMP.messagein(switchinterface), ReactiveMP.ManyOf(map((input) -> ReactiveMP.messagein(input), inputsinterfaces))))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the output with pipeline constraints
function ReactiveMP.get_messages_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{NTuple{N, ReactiveMP.IndexedNodeInterface}}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    inputsinterfaces = messages[1]

    msgs_names = Val{(name(inputsinterfaces[1]), )}
    msgs_observable =
    combineLatest(map((input) -> ReactiveMP.messagein(input), inputsinterfaces), PushNew()) |>
        map_to((ReactiveMP.ManyOf(map((input) -> ReactiveMP.messagein(input), inputsinterfaces)),))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the input with pipeline constraints
function ReactiveMP.get_messages_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{ReactiveMP.NodeInterface}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    outputinterface = messages[1]

    msgs_names = Val{(name(outputinterface), )}
    msgs_observable = combineLatestUpdates((ReactiveMP.messagein(outputinterface), ), PushNew())
    return msgs_names, msgs_observable
end

function ReactiveMP.get_marginals_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, marginals::Tuple{ReactiveMP.NodeInterface}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    switchinterface = marginals[1]

    marginal_names       = Val{(name(switchinterface), )}
    marginals_observable = combineLatestUpdates((getmarginal(ReactiveMP.connectedvar(switchinterface), IncludeAll()), ), PushNew())

    return marginal_names, marginals_observable
end

	# import Distributions: Dirichlet
	# using SpecialFunctions: loggamma
	# function Dirichlet{T}(alpha::AbstractVector{T}; check_args::Bool=true) where T
 #        alpha0 = sum(alpha)
 #        lmnB = sum(loggamma, alpha) - loggamma(alpha0)
 #        Dirichlet{T,typeof(alpha),typeof(lmnB)}(alpha, alpha0, lmnB)
 #    end
	
end

# ╔═╡ d2910ba3-6f0c-4905-b10a-f32ad8239ab6
md"""
# Validation experiments: online Dirichlet process training
"""

# ╔═╡ c30f10ee-7bb7-4b12-8c90-8c1947ce0bc9
begin
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}");
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}");
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{statistics}");
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\pgfplotsset{compat=1.5.1}");
	function plot_ellipse(means::Vector, cov::Matrix)
		@assert length(means) == 2
		@assert size(cov) == (2,2)
		rads = eigvals(cov)
		vecs = eigvecs(cov)
		xrad = rads[1]
		yrad = rads[2]
		rot = atand(vecs[2,1], vecs[1,2])
		z = " [black] (axis cs: $(means[1]), $(means[2])) ellipse [rotate=$(rot), x radius=$(xrad), y radius=$(yrad)];"
		return raw"\draw" * z
	end
end;

# ╔═╡ 2bfcbece-2b63-443d-95d9-de7479ded607
md"""
### Generate data
"""

# ╔═╡ fa56df5b-e46b-4ed7-a9a7-7ae1d8697250
md"""number of samples: $(@bind nr_samples Slider(1:2_000; default=1500, show_value=true))"""

# ╔═╡ 1ee92290-71ac-41ce-9666-241478bc04cb
begin
	means = (rand(MersenneTwister(123), 8, 2).-0.5)*40
	means_vec = [means[k,:] for k in 1:8]
	dist = MixtureModel(
		map(m -> MvNormal(m, I) ,means_vec),
		Categorical([0.1, 0.3, 0.2, 0.05, 0.15, 0.1, 0.05, 0.05])
	)
end;

# ╔═╡ b9550341-a928-4274-b9da-40ac84d2a991
data = rand(MersenneTwister(123), dist, nr_samples);

# ╔═╡ d6fbfce8-c87b-4e0e-b11c-4d9fdce25b51
begin
	plt.figure()
	plt.scatter(data[1,:], data[2,:], alpha=0.1)
	plt.xlabel(L"y_1")
	plt.ylabel(L"y_2")
	plt.xlim(-20, 20)
	plt.ylim(-20, 20)
	plt.gcf()
end

# ╔═╡ 4eee819f-f099-4d4d-b72b-434be3077f99
md"""
### Model specification
"""

# ╔═╡ 01187b6e-8bf2-45eb-b918-73b767cf94b8
md"""upper bound number of components: $(@bind nr_components Slider(1:20; default=10, show_value=true))"""

# ╔═╡ e9c3f636-0049-4b42-b0f9-cc42bee61360
@model function model_dirichlet_process()

    # specify experimental outcomes
    y = datavar(Vector{Float64})

	# updatable parameters
    α = datavar(Vector{Float64})
	μ_θ = datavar(Vector{Float64}, nr_components)
	Λ_θ = datavar(Matrix{Float64}, nr_components)

	# other variables
	θk = randomvar(nr_components)
	
    # specify initial distribution over clusters
    π ~ Dirichlet(α)
		
	# prior over model selection variable
	z ~ Categorical(π) where { pipeline = EnforceMarginalFunctionalDependency(:out) }

	# specify prior models over θ
	for k in 1:nr_components
		θk[k] ~ MvNormalMeanPrecision(μ_θ[k], Λ_θ[k])
	end

	tθ = tuple(θk...)

	# specify mixture distribution
	θ ~ Mixture(z, tθ) where { pipeline = EnforceMarginalFunctionalDependency(:switch) }

	# specify observation noise
	y ~ MvNormalMeanPrecision(θ, diagm(ones(2)))

    return y, θ, θk, z, π, α

end

# ╔═╡ 5fd5ee6b-3208-42fe-b8b2-14e95ffd08b5
md"""
### Constraint specification
"""

# ╔═╡ 07b1d74f-d203-475c-834e-ae83459d714a
@constraints function constraints_dirichlet_process()
    q(z) :: PointMass
end;

# ╔═╡ 9bc23d63-8693-4eb8-ae90-29ddc4ec4997
md"""
### Probabilistic inference
"""

# ╔═╡ 16e37d4b-523f-4d1b-ab71-b54ba81364b1
@bind alpha Slider(-5:5; default=-1, show_value=false)

# ╔═╡ 6293a919-a314-4063-a679-c70008b12b2f
md"""alpha = 1e$(alpha)"""

# ╔═╡ 5f6aaef7-1deb-4812-9888-fc52009bdc5f
base_measure = MvNormalMeanPrecision(zeros(2), 0.1*diagm(ones(2)));

# ╔═╡ 37897bf5-c864-456f-87c0-1251ad532010
begin
	function update_alpha_vector(α_prev)
		ind = findfirst(x -> isapprox(1e-10,x;rtol=0.1), α_prev)
		if isnothing(ind)
			α_new = α_prev
			@error "upper bound reached"
		elseif ind == 2 && α_prev[1] != 10.0^alpha
			α_new = α_prev
			α_new[ind-1] = 1
			α_new[ind] = 10.0^alpha
		elseif ind > 2 && α_prev[ind-1] ≈ 1+10.0^alpha
			α_new = α_prev
			α_new[ind-1] = 1
			α_new[ind] = 10.0^alpha
		else
			α_new = α_prev
		end
		return α_new
	end
	function update_alpha(dist)
		return update_alpha_vector(probvec(dist))
	end
	function broadcast_mean_precision(dist)
		tmp = mean_precision.(dist)
		return first.(tmp), last.(tmp)
	end
	@rule Mixture((:inputs, k), Marginalisation) (m_out::Any, q_switch::PointMass) = begin
	
	    # check whether mean is one-hot
	    p = mean(q_switch)
	    @assert sum(p) ≈ 1 "The selector variable connected to the Mixture node is not normalized."
	    @assert all(x -> x == 1 || x == 0, p) "The selector variable connected to the Mixture node is not one-hot encoded."
	
	    # get selected cluster
	    kmax = argmax(p)
	
	    if k == kmax
	        @logscale 0
	        return m_out
	    else
	        @logscale missing
	        return missing
	    end
	end
end;

# ╔═╡ aec5408c-aab9-4fba-9bf2-0bace3c2c29f
autoupdates_dirichlet_process = @autoupdates begin
    α = update_alpha(q(π))
	μ_θ, Λ_θ = broadcast_mean_precision(q(θk))
end;

# ╔═╡ 5f3b9e1f-2ffc-403b-95df-3e48504399bc
 function run_dirichlet_process(data)
	 alpha_start = 1e-10*ones(nr_components)
	 alpha_start[1] = 10.0^alpha
	 return rxinference(
		model         = model_dirichlet_process(),
		data          = (y = [data[:,k] for k=1:size(data,2)], ),
		constraints   = constraints_dirichlet_process(),
		autoupdates   = autoupdates_dirichlet_process,
		initmarginals = (π = Dirichlet(alpha_start; check_args=false), θk = base_measure),
		returnvars    = (:π, :θk),
		keephistory   = size(data,2),
		historyvars   = (z = KeepLast(), π = KeepLast(), θk = KeepLast()),
		autostart     = true,
		addons        = AddonLogScale()
	)
 end;

# ╔═╡ 1b2c3587-b4b6-4d5b-a54a-430fc478dcd4
md"""
### Results
"""

# ╔═╡ 2bfa1683-86c3-4b9d-b7e3-3890bb32c645
results_dirichlet_process = run_dirichlet_process(data)

# ╔═╡ 075cbfdd-698b-4a38-8ae3-557d39acb5d2
md"""sample index: $(@bind N Slider(1:nr_samples; default=nr_samples, show_value=true))"""

# ╔═╡ e2ed2836-5a4d-4762-ab59-d77277b47f39
begin
	plt.figure()
	plt.scatter(data[1,1:N], data[2,1:N], alpha=0.1, c=argmax.(mean.(results_dirichlet_process.history[:z][1:N])))
	for k in findall(x -> x >= 1, probvec(results_dirichlet_process.history[:π][N]))
		plt.scatter(mean(results_dirichlet_process.history[:θk][N][k])[1], mean(results_dirichlet_process.history[:θk][N][k])[2], marker="x", color="black")
	end
	plt.xlabel(L"y_1")
	plt.ylabel(L"y_2")
	plt.xlim(-20, 20)
	plt.ylim(-20, 20)
	plt.grid()
	plt.gcf()
end

# ╔═╡ ef365ff1-9a53-4e7e-96ce-a68baeb6b67b
begin
	data_1500 = rand(MersenneTwister(123), dist, 1500)
	results_dirichlet_process_1500 = run_dirichlet_process(data_1500)
	N1 = 5
	N2 = 25
	N3 = 250
	N4 = 1500

	C1 = isnothing(findfirst(x -> isapprox(1e-10,x;rtol=0.1), probvec(results_dirichlet_process_1500.history[:π][N1]))) ? nr_components : findfirst(x -> isapprox(1e-10,x;rtol=0.1), probvec(results_dirichlet_process_1500.history[:π][N1])) - 2
	C2 = isnothing(findfirst(x -> isapprox(1e-10,x;rtol=0.1), probvec(results_dirichlet_process_1500.history[:π][N2]))) ? nr_components : findfirst(x -> isapprox(1e-10,x;rtol=0.1), probvec(results_dirichlet_process_1500.history[:π][N2])) - 2	
	C3 = isnothing(findfirst(x -> isapprox(1e-10,x;rtol=0.1), probvec(results_dirichlet_process_1500.history[:π][N3]))) ? nr_components : findfirst(x -> isapprox(1e-10,x;rtol=0.1), probvec(results_dirichlet_process_1500.history[:π][N3])) - 2	
	C4 = isnothing(findfirst(x -> isapprox(1e-10,x;rtol=0.1), probvec(results_dirichlet_process_1500.history[:π][N4]))) ? nr_components : findfirst(x -> isapprox(1e-10,x;rtol=0.1), probvec(results_dirichlet_process_1500.history[:π][N4])) - 2	

	fig_tikz = @pgf GroupPlot(

		# group plot options
		{
			group_style = {
				group_size = "2 by 4",
				horizontal_sep = "2cm"
			},
			label_style={font="\\footnotesize"},
			ticklabel_style={font="\\scriptsize",},
	        grid = "major",
		},

		# row 1 column 1
		{
			xlabel="\$y_1\$",
			ylabel_style={align="center"},
			ylabel = "\$\\bm{N=$(N1)}\$ \\\\ \\\\ \$y_2\$",
			xmin = -20,
			ymin = -20,
			xmax = 20,
			ymax = 20,
			width = "2in",
			height = "2in",
			title = "\\textbf{Assignments and clusters}",
			axis_equal,
		},
		Plot({ 
				scatter,
				only_marks,
				opacity=0.2,
            	scatter_src="explicit"
	        },
			Table(
				{
	                meta = "label"
	            },
	            x = data_1500[1,1:N1],
	            y = data_1500[2,1:N1],
	            label = argmax.(mean.(results_dirichlet_process_1500.history[:z][1:N1]))
			)
    	),
		Plot({ 
				only_marks,
				mark_size="4pt",
				mark_color="black",
				mark="x",
				very_thick,
	        },
			Table(hcat(mean.(results_dirichlet_process_1500.history[:θk][N1][1:C1])...)')
    	),
		plot_ellipse.(mean.(results_dirichlet_process_1500.history[:θk][N1])[1:C1], map(x->x.+diagm(ones(2)), cov.(results_dirichlet_process_1500.history[:θk][N1])[1:C1])),

		# row 1 column 2
		{
			ybar,
			bar_width="10pt",
			ylabel = "\$\\alpha_k\$",
			xlabel = "\$k\$",
			ymin = 0,
			width = "4in",
			height = "2in",
			title = "\\textbf{Posterior concentration parameters}"
	    },
	    Plot({ 
				fill="blue",
	        },
	        Table(collect(1:nr_components), probvec(results_dirichlet_process_1500.history[:π][N1]))
	    ),

		# row 2 column 1
		{
			xlabel="\$y_1\$",
			ylabel_style={align="center"},
			ylabel = "\$\\bm{N=$(N2)}\$ \\\\ \\\\ \$y_2\$",
			xmin = -20,
			ymin = -20,
			xmax = 20,
			ymax = 20,
			width = "2in",
			height = "2in",
			axis_equal,
		},
		Plot({ 
				scatter,
				only_marks,
				opacity=0.2,
            	scatter_src="explicit"
	        },
			Table(
				{
	                meta = "label"
	            },
	            x = data_1500[1,1:N2],
	            y = data_1500[2,1:N2],
	            label = argmax.(mean.(results_dirichlet_process_1500.history[:z][1:N2]))
			)
    	),
		Plot({ 
				only_marks,
				mark_size="4pt",
				mark_color="black",
				mark="x",
				very_thick,
	        },
			Table(hcat(mean.(results_dirichlet_process_1500.history[:θk][N2][1:C2])...)')
    	),
		plot_ellipse.(mean.(results_dirichlet_process_1500.history[:θk][N2])[1:C2], map(x->x.+diagm(ones(2)), cov.(results_dirichlet_process_1500.history[:θk][N2])[1:C2])),

		# row 2 column 2
		{
			ybar,
			bar_width="10pt",
			ylabel = "\$\\alpha_k\$",
			xlabel = "\$k\$",
			ymin = 0,
			width = "4in",
			height = "2in",
	    },
	    Plot({ 
				fill="blue",
	        },
	        Table(collect(1:nr_components), probvec(results_dirichlet_process_1500.history[:π][N2]))
	    ),

		# row 3 column 1
		{
			xlabel="\$y_1\$",
			ylabel_style={align="center"},
			ylabel = "\$\\bm{N=$(N3)}\$ \\\\ \\\\ \$y_2\$",
			xmin = -20,
			ymin = -20,
			xmax = 20,
			ymax = 20,
			width = "2in",
			height = "2in",
			axis_equal,
		},
		Plot({ 
				scatter,
				only_marks,
				opacity=0.2,
         		scatter_src="explicit",
	        },
			Table(
				{
	                meta = "label"
	            },
	            x = data_1500[1,1:N3],
	            y = data_1500[2,1:N3],
	            label = argmax.(mean.(results_dirichlet_process_1500.history[:z][1:N3]))
			)
    	),
		Plot({ 
				only_marks,
				mark_size="4pt",
				mark_color="black",
				mark="x",
				very_thick,
	        },
			Table(hcat(mean.(results_dirichlet_process_1500.history[:θk][N3][1:C3])...)')
    	),
		plot_ellipse.(mean.(results_dirichlet_process_1500.history[:θk][N3])[1:C3], map(x->x.+diagm(ones(2)), cov.(results_dirichlet_process_1500.history[:θk][N3])[1:C3])),

		# row 3 column 2
		{
			ybar,
			bar_width="10pt",
			ylabel = "\$\\alpha_k\$",
			xlabel = "\$k\$",
			ymin = 0,
			width = "4in",
			height = "2in",
	    },
	    Plot({ 
				fill="blue",
	        },
	        Table(collect(1:nr_components), probvec(results_dirichlet_process_1500.history[:π][N3]))
	    ),

		# row 4 column 1
		{
			xlabel="\$y_1\$",
			ylabel_style={align="center"},
			ylabel = "\$\\bm{N=$(N4)}\$ \\\\ \\\\ \$y_2\$",
			xmin = -20,
			ymin = -20,
			xmax = 20,
			ymax = 20,
			width = "2in",
			height = "2in",
			axis_equal,
		},
		Plot({ 
				scatter,
				only_marks,
				opacity=0.2,
            	scatter_src="explicit"
	        },
			Table(
				{
	                meta = "label"
	            },
	            x = data_1500[1,1:N4],
	            y = data_1500[2,1:N4],
	            label = argmax.(mean.(results_dirichlet_process_1500.history[:z][1:N4]))
			)
    	),
		Plot({ 
				only_marks,
				mark_size="4pt",
				mark_color="black",
				mark="x",
				very_thick,
	        },
			Table(hcat(mean.(results_dirichlet_process_1500.history[:θk][N4][1:C4])...)')
    	),
		plot_ellipse.(mean.(results_dirichlet_process_1500.history[:θk][N4])[1:C4], map(x->x.+diagm(ones(2)), cov.(results_dirichlet_process_1500.history[:θk][N4])[1:C4])),

		# row 4 column 2
		{
			ybar,
			bar_width="10pt",
			ylabel = "\$\\alpha_k\$",
			xlabel = "\$k\$",
			ymin = 0,
			width = "4in",
			height = "2in",
	    },
	    Plot({ 
				fill="blue",
	        },
	        Table(collect(1:nr_components), probvec(results_dirichlet_process_1500.history[:π][N4]))
	    ),
			
	)
end

# ╔═╡ b6ecbc4d-6bc2-4ec5-ad5b-78ae441fbf49
begin
	pgfsave("../exports/validation_experiments_dirichlet.tikz", fig_tikz)
	pgfsave("../exports/validation_experiments_dirichlet.png", fig_tikz)
	pgfsave("../exports/validation_experiments_dirichlet.pdf", fig_tikz)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PGFPlotsX = "8314cec4-20b6-5062-9cdb-752b83310925"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RxInfer = "86711068-29c9-4ff7-b620-ae75d7495b3d"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
Distributions = "~0.25.84"
LaTeXStrings = "~1.3.0"
PGFPlotsX = "~1.5.3"
PlutoUI = "~0.7.50"
PyPlot = "~2.11.0"
RxInfer = "~2.9.0"
SpecialFunctions = "~2.2.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.1"
manifest_format = "2.0"
project_hash = "a1813872a723ed9a36b1d3bb10b594d41beb85cf"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0310e08cb19f5da31d08341c6120c047598f5b9c"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.5.0"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4d9946e51e24f5e509779e3e2c06281a733914c2"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.1.0"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4aff5fa660eb95c2e0deb6bcdabe4d9a96bc4667"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "0.8.18"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "Static"]
git-tree-sha1 = "2c144ddb46b552f72d7eafe7cc2f50746e41ea21"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.2"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "02d2316b7ffceff992f3096ae48c7829a8aa0638"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.3"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "e32a90da027ca45d84678b826fffd3110bb3fc90"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.8.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "89a9db8d28102b094992472d333674bd1a83ce2a"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.1"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefaultApplication]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "c0dfa5a35710a193d83f03124356eef3386688fc"
uuid = "3f0dd361-4fe0-5fc6-8523-80b14ec94d85"
version = "1.1.0"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "d71264a7b9a95dca3b8fff4477d94a837346c545"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.84"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainIntegrals]]
deps = ["CompositeTypes", "DomainSets", "FastGaussQuadrature", "GaussQuadrature", "HCubature", "IntervalSets", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "9de4488ef530cc8b83c6c061cf8fc47290b2f3ac"
uuid = "cc6bae93-f070-4015-88fd-838f9505a86c"
version = "0.4.1"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "aa0f95312367be88ec8459994ae31a6e308eee2d"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.6.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1dfc97ea0df8b3689c2608db8bf28b3fc4f4147b"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.5.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "ed1b56934a2f7a65035976985da71b6a65b4f2cf"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.18.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GaussQuadrature]]
deps = ["SpecialFunctions"]
git-tree-sha1 = "eb6f1f48aa994f3018cbd029a17863c6535a266d"
uuid = "d54b0c1a-921d-58e0-8e36-89d8069c0969"
version = "0.5.8"

[[deps.GraphPPL]]
deps = ["MacroTools", "TupleTools"]
git-tree-sha1 = "36d1953626dcb87e87824488167a5a27e6046424"
uuid = "b3f8163a-e979-4e85-b43e-1f63d8c8b42c"
version = "3.1.0"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "e95b36755023def6ebc3d269e6483efa8b2f7f65"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.5.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "734fd90dd2f920a2f1921d5388dcebe805b262dc"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.14"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "f6742270b551e03bd8ba64e1c754f5fab089ab63"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "0.22.16"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDTypes", "SLEEFPirates", "SnoopPrecompile", "SpecialFunctions", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "2acf6874142d05d5d1ad49e8d3786b8cd800936d"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.152"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "0ff59b4b9024ab9a736db1ad902d2b1b48441c19"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "0.9.6"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "1903afc76b7d01719d9c30d3c7d501b61db96721"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.4"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PGFPlotsX]]
deps = ["ArgCheck", "Dates", "DefaultApplication", "DocStringExtensions", "MacroTools", "OrderedCollections", "Parameters", "Requires", "Tables"]
git-tree-sha1 = "e98a6743775e107062be357560977c06850a79be"
uuid = "8314cec4-20b6-5062-9cdb-752b83310925"
version = "1.5.3"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6f4fbcd1ad45905a5dee3f4256fabb49aa2110c6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.7"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "62f417f6ad727987c755549e9cd88c46578da562"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.95.1"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "f9d953684d4d21e947cb6d642db18853d43cb027"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.ReactiveMP]]
deps = ["DataStructures", "Distributions", "DomainIntegrals", "DomainSets", "FastGaussQuadrature", "ForwardDiff", "HCubature", "LazyArrays", "LinearAlgebra", "LoopVectorization", "MacroTools", "Optim", "PositiveFactorizations", "Random", "Requires", "Rocket", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers", "TupleTools", "Unrolled"]
git-tree-sha1 = "ddd3fdc131c7c236e38715f7c2fd228d044632b7"
uuid = "a194aa59-28ba-4574-a09c-4a745416d6e3"
version = "3.7.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Rocket]]
deps = ["DataStructures", "Sockets", "Unrolled"]
git-tree-sha1 = "33e270ce5710d5315f28c205ec7d598c4fdf660d"
uuid = "df971d30-c9d6-4b37-b8ff-e965b2cb3a40"
version = "1.7.0"

[[deps.RxInfer]]
deps = ["DataStructures", "Distributions", "DomainSets", "GraphPPL", "LinearAlgebra", "MacroTools", "Optim", "ProgressMeter", "Random", "ReactiveMP", "Reexport", "Rocket", "TupleTools"]
git-tree-sha1 = "dd5436be8ed8a5fa9318b4b061ba14185c9b4c25"
uuid = "86711068-29c9-4ff7-b620-ae75d7495b3d"
version = "2.9.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "cda0aece8080e992f6370491b08ef3909d1c04e7"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.38"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d0435ba43ab5ad1cbb5f0d286ca4ba67029ed3ee"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.4"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "5589ab073f8a244d2530b36478f53806f9106002"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.2.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2d7d9e1ddadc8407ffd460e24218e37ef52dd9a3"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.16"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5aa6250a781e567388f3285fb4b0f214a501b4d5"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.2.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "c97f60dd4f2331e1a495527f80d242501d2f9865"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.1"

[[deps.TinyHugeNumbers]]
git-tree-sha1 = "d1bd5b57d45431fcbf2db38d3e17453a603e76ad"
uuid = "783c9a47-75a3-44ac-a16b-f1ab7b3acf04"
version = "1.0.0"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Unrolled]]
deps = ["MacroTools"]
git-tree-sha1 = "69eccecbb9ba198e46b4ddcb164f8dea3c685601"
uuid = "9602ed7d-8fef-5bc8-8597-8f21381861e8"
version = "0.1.4"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "952ba509a61d1ebb26381ac459c5c6e838ed43c4"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.60"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─d2910ba3-6f0c-4905-b10a-f32ad8239ab6
# ╠═a1c1ef76-64d9-4f7b-b6b8-b70e180fc0ff
# ╟─c30f10ee-7bb7-4b12-8c90-8c1947ce0bc9
# ╟─2bfcbece-2b63-443d-95d9-de7479ded607
# ╟─fa56df5b-e46b-4ed7-a9a7-7ae1d8697250
# ╟─1ee92290-71ac-41ce-9666-241478bc04cb
# ╠═b9550341-a928-4274-b9da-40ac84d2a991
# ╟─d6fbfce8-c87b-4e0e-b11c-4d9fdce25b51
# ╟─4eee819f-f099-4d4d-b72b-434be3077f99
# ╟─01187b6e-8bf2-45eb-b918-73b767cf94b8
# ╟─7c5873ce-a0d5-46f2-9d59-188ffd09cb5b
# ╠═e9c3f636-0049-4b42-b0f9-cc42bee61360
# ╟─5fd5ee6b-3208-42fe-b8b2-14e95ffd08b5
# ╠═07b1d74f-d203-475c-834e-ae83459d714a
# ╟─9bc23d63-8693-4eb8-ae90-29ddc4ec4997
# ╟─16e37d4b-523f-4d1b-ab71-b54ba81364b1
# ╟─6293a919-a314-4063-a679-c70008b12b2f
# ╠═5f6aaef7-1deb-4812-9888-fc52009bdc5f
# ╟─37897bf5-c864-456f-87c0-1251ad532010
# ╠═aec5408c-aab9-4fba-9bf2-0bace3c2c29f
# ╠═5f3b9e1f-2ffc-403b-95df-3e48504399bc
# ╟─1b2c3587-b4b6-4d5b-a54a-430fc478dcd4
# ╠═2bfa1683-86c3-4b9d-b7e3-3890bb32c645
# ╟─075cbfdd-698b-4a38-8ae3-557d39acb5d2
# ╟─e2ed2836-5a4d-4762-ab59-d77277b47f39
# ╟─ef365ff1-9a53-4e7e-96ce-a68baeb6b67b
# ╟─b6ecbc4d-6bc2-4ec5-ad5b-78ae441fbf49
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
