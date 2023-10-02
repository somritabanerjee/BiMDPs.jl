module MLLRoverWorld
using POMDPs
using POMDPModelTools
using POMDPPolicies
using Random
using Parameters
using POMDPSimulators

struct MLLState
	x::Int
	y::Int
	t::Int # discrete time
	measured::Bool
end

@enum MLLAction UP DOWN LEFT RIGHT MEASURE

@with_kw struct MLLRoverWorldMDP <: MDP{MLLState, MLLAction}
	grid_size::Tuple{Int,Int} = (20, 20)   # size of the grid
	max_time::Int = 100 # max time steps
	null_xy::Tuple{Int,Int} = (-1, -1) # terminal state outside of the grid
	p_transition::Real = 1.0 # probability of transitioning to the correct next state
    γ::Float64 = 0.95
	current_tgt::Tuple{Tuple{Int, Int}, Tuple{Int, Int}, Float64} = ((10,18),(1,max_time),50) # ((x,y), (t0,tf), reward)
	obstacles_grid::Array{Float64,3} = zeros(Float64, (grid_size[1], grid_size[2], max_time)) # grid of obstacles
	exit_xys::Vector{Tuple{Int,Int}} = [(18, 3)] # if the rover is at any of these xys, the episode terminates
	init_state::MLLState = MLLState(1, 1, 1)
	measure_reward::Float64 = 2.0
end

POMDPs.isterminal(mdp::MLLRoverWorldMDP, s::MLLState) = ((s.x, s.y) == mdp.null_xy)
POMDPs.discount(mdp::MLLRoverWorldMDP) = mdp.γ

include("states.jl")
include("actions.jl")
include("rewards.jl")
include("transitions.jl")
include("visualizations.jl")
include("policies.jl")
include("simulation.jl")

function print_details(mdp::MLLRoverWorldMDP)
	println("========== MLLRoverWorldMDP ==========")
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