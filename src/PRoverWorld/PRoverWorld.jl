module PRoverWorld
using POMDPs
using POMDPModelTools
using POMDPPolicies
using Random
using Parameters
using Plots
using DiscreteValueIteration
using TabularTDLearning
using POMDPSimulators
using Latexify

struct PState
	x::Int
	y::Int
	t::Int # discrete time
	e::Bool # energy level
	visited::Vector{Bool} # visited[i] = true if target i has been visited
end

struct PObservation
	x::Int
	y::Int
	t::Int # discrete time
	e::Bool # energy level
	visited::Vector{Bool} # visited[i] = true if target i has been visited
end

@enum PAction UP DOWN LEFT RIGHT MEASURE

@with_kw struct PRoverWorldMDP <: POMDP{PState, PAction, PObservation}
	grid_size::Tuple{Int,Int} = (20, 20)   # size of the grid
	max_time::Int = 100 # max time steps
	null_xy::Tuple{Int,Int} = (-1, -1) # terminal state outside of the grid
    γ::Float64 = 0.95
	tgts::Dict{Int, Tuple{Tuple{Int, Int}, Tuple{Int, Int}, Float64}} = Dict(1 => ((10,18),(1,max_time),50),
													2 => ((4,3),(1,max_time),25),
													3 => ((18,3),(1,max_time),100)
													) # dictionary mapping target ID to ((x,y), (t0,tf), reward)
	obstacles_grid::Array{Float64,3} = zeros(Float64, (grid_size[1], grid_size[2], max_time)) # grid of obstacles
	exit_xys::Vector{Tuple{Int,Int}} = [(18, 3)] # if the rover is at any of these xys, the episode terminates
	include_measurement::Bool = true
	measure_reward::Float64 = 2.0
	# Any addition to these params should be reflected in modify_γ() below
end

POMDPs.isterminal(mdp::PRoverWorldMDP, s::PState) = ((s.x, s.y) == mdp.null_xy)
POMDPs.discount(mdp::PRoverWorldMDP) = mdp.γ

include("states.jl")
include("actions.jl")
include("rewards.jl")
include("transitions.jl")
include("observations.jl")
include("visualizations.jl")
include("policies.jl")
include("simulation.jl")


function print_details(mdp::PRoverWorldMDP)
	println("========== PRoverWorldMDP ==========")
	for f in fieldnames(typeof(mdp))
		if f == :obstacles_grid
			println("$f: omitting.")
		else
			println("$f: $(getfield(mdp, Symbol(f)))")
		end
	end
	println("=====================================")
end

end # module
