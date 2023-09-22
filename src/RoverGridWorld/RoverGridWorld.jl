module RoverGridWorld
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
end

@enum Action UP DOWN LEFT RIGHT

@with_kw struct RoverGridWorldMDP <: MDP{State, Action}
	grid_size::Tuple{Int,Int} = (10, 10)   # size of the grid
	null_state::State = State(-1, -1) # terminal state outside of the grid
	p_transition::Real = 0.7 # probability of transitioning to the correct next state
    γ::Float64 = 0.95
	reward_vals::Dict{Tuple{Int, Int}, Float64} = Dict((4,3) => -10.0,
														(4,6) => -5.0,
														(9,3) => 10.0,
														(8,8) => 3.0
													)
end	

POMDPs.isterminal(mdp::RoverGridWorldMDP, s::State) = s == mdp.null_state
POMDPs.discount(mdp::RoverGridWorldMDP) = mdp.γ

include("states.jl")
include("actions.jl")
include("rewards.jl")
include("transitions.jl")
include("visualizations.jl")
include("policies.jl")
include("simulation.jl")

function modify_γ(mdp::RoverGridWorldMDP; γ::Float64=mdp.γ)
	mdp_new = RoverGridWorldMDP(grid_size=mdp.grid_size, null_state=mdp.null_state, p_transition=mdp.p_transition, reward_vals=mdp.reward_vals, γ=γ)
	return mdp_new
end

end # module
