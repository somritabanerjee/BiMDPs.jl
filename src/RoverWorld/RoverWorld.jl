module RoverWorld
using POMDPs
using POMDPModelTools
using POMDPPolicies
using Random
using Parameters
using Plots
using DiscreteValueIteration
using Reel
using TabularTDLearning
using POMDPSimulators
using ColorSchemes, Colors
using Latexify

struct State
	x::Int
	y::Int
	t::Int # discrete time
	visited::Vector{Bool} # visited[i] = true if target i has been visited
end

@enum Action UP DOWN LEFT RIGHT # TODO: Add diagonals

@with_kw struct RoverWorldMDP <: MDP{State, Action}
	grid_size::Tuple{Int,Int} = (20, 20)   # size of the grid
	max_time::Int = 100 # max time steps
	null_xy::Tuple{Int,Int} = (-1, -1) # terminal state outside of the grid
	p_transition::Real = 1.0 # probability of transitioning to the correct next state
    γ::Float64 = 0.95
	reward_vals::Dict{Int, Tuple{Tuple{Int, Int}, Tuple{Int, Int}, Float64}} = Dict(1 => ((10,18),(1,max_time),50),
													2 => ((4,3),(1,max_time),25),
													3 => ((18,3),(1,max_time),100)
													) # dictionary mapping target ID to ((x,y), (t0,tf), reward)
	exit_xys::Vector{Tuple{Int,Int}} = [(18, 3)] # if the rover is at any of these xys, the episode terminates
	# Any addition to these params should be reflected in modify_γ() below
end

POMDPs.isterminal(mdp::RoverWorldMDP, s::State) = ((s.x, s.y) == mdp.null_xy)
POMDPs.discount(mdp::RoverWorldMDP) = mdp.γ

include("states.jl")
include("actions.jl")
include("rewards.jl")
include("transitions.jl")
include("visualizations.jl")
include("policies.jl")
include("simulation.jl")

function modify_γ(mdp::RoverWorldMDP; γ::Float64=mdp.γ)
	mdp_new = RoverWorldMDP(grid_size = mdp.grid_size,
								max_time = mdp.max_time,
								null_xy = mdp.null_xy, 
								p_transition = mdp.p_transition, 
								reward_vals = mdp.reward_vals, 
								γ = γ,
								exit_xys = mdp.exit_xys)
	return mdp_new
end

end # module
