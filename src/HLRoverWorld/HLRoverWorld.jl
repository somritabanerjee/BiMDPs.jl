module HLRoverWorld
using POMDPs
using POMDPModelTools
using POMDPPolicies
using Random
using Parameters
using POMDPSimulators

struct HLState
    x::Int
    y::Int
    t::Int # discrete time
    visited::Vector{Bool} # visited[i] = true if target i has been visited
end

struct HLAction
    tgt::Int # target ID
end

@with_kw struct HLRoverWorldMDP <: MDP{HLState, HLAction}
    grid_size::Tuple{Int,Int} = (20, 20)   # size of the grid
    max_time::Int = 100 # max time steps
    null_xy::Tuple{Int,Int} = (-1, -1) # terminal HLState outside of the grid
    γ::Float64 = 0.95
    tgts::Dict{Int, Tuple{Tuple{Int, Int}, Tuple{Int, Int}, Float64}} = Dict() # dictionary mapping target ID to ((x,y), (t0,tf), reward)
    exit_xys::Vector{Tuple{Int,Int}} = [(18, 3)] # if the rover is at any of these xys, the episode terminates
end

POMDPs.isterminal(mdp::HLRoverWorldMDP, s::HLState) = ((s.x, s.y) == mdp.null_xy)
POMDPs.discount(mdp::HLRoverWorldMDP) = mdp.γ

include("states.jl")
include("actions.jl")
include("rewards.jl")
include("transitions.jl")
include("visualizations.jl")
include("policies.jl")
include("simulation.jl")

end # module