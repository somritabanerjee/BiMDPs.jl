module MRoverWorld
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

struct MState
	x::Int
	y::Int
	t::Int # discrete time
	measured::Vector{Bool} # measured[i] = true if vicinity of target i has been measured
	visited::Vector{Bool} # visited[i] = true if target i has been visited
end

@enum MAction UP DOWN LEFT RIGHT MEASURE # TODO: Add diagonals

@with_kw struct MRoverWorldMDP <: MDP{MState, MAction}
	grid_size::Tuple{Int,Int} = (20, 20)   # size of the grid
	max_time::Int = 100 # max time steps
	null_xy::Tuple{Int,Int} = (-1, -1) # terminal state outside of the grid
	p_transition::Real = 1.0 # probability of transitioning to the correct next state
    γ::Float64 = 0.95
	tgts::Dict{Int, Tuple{Tuple{Int, Int}, Tuple{Int, Int}, Float64}} = Dict(1 => ((10,18),(1,max_time),50),
													2 => ((4,3),(1,max_time),25),
													3 => ((18,3),(1,max_time),100)
													) # dictionary mapping target ID to ((x,y), (t0,tf), reward)
	obstacles_grid::Array{Float64,3} = zeros(Float64, (grid_size[1], grid_size[2], max_time)) # grid of obstacles
	exit_xys::Vector{Tuple{Int,Int}} = [(18, 3)] # if the rover is at any of these xys, the episode terminates
	measure_reward::Float64 = 2.0
end

POMDPs.isterminal(mdp::MRoverWorldMDP, s::MState) = ((s.x, s.y) == mdp.null_xy)
POMDPs.discount(mdp::MRoverWorldMDP) = mdp.γ

include("states.jl")
include("actions.jl")
include("rewards.jl")
include("transitions.jl")
include("visualizations.jl")
include("policies.jl")
include("simulation.jl")

function print_details(mdp::MRoverWorldMDP)
	println("========== MRoverWorldMDP ==========")
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
