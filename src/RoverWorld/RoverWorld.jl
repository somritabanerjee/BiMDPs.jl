module RoverWorld
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

struct State
	x::Int
	y::Int
	t::Int # discrete time
	visited::Vector{Bool} # visited[i] = true if target i has been visited
end

@enum Action UP DOWN LEFT RIGHT MEASURE # TODO: Add diagonals

@with_kw struct RoverWorldMDP <: MDP{State, Action}
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
	include_measurement::Bool = false
	measure_reward::Float64 = 2.0
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
								tgts = mdp.tgts, 
								obstacles_grid = mdp.obstacles_grid,
								γ = γ,
								exit_xys = mdp.exit_xys,
								include_measurement = mdp.include_measurement,
								measure_reward = mdp.measure_reward)
	return mdp_new
end

function print_details(mdp::RoverWorldMDP)
	println("========== RoverWorldMDP ==========")
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
