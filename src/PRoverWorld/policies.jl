# create policy grid showing the best action in each state
function policy_grid(policy::Policy, xmax::Int, ymax::Int; t::Int = 1, visited::Vector{Bool} = fill(false, length(policy.mdp.tgts)))
    arrows = Dict(UP => "↑",
                  DOWN => "↓",
                  LEFT => "←",
                  RIGHT => "→",
                  MEASURE => "=")

    grid = Array{String}(undef, xmax, ymax)
    for x = 1:xmax, y = 1:xmax
        s = PState(x, y, t, visited)
        grid[x,y] = arrows[action(policy, s)]
    end

    return grid
end

function values(mdp::PRoverWorldMDP, policy::Policy)
    valid_states = non_null_states(mdp)
    U = map(s->value(policy, s), valid_states)
end

struct NothingPolicy <: Policy end

# Use this to get a stationary grid of rewards
function values(mdp::PRoverWorldMDP, policy::Union{NothingPolicy, FunctionPolicy})
    valid_states = non_null_states(mdp)
    rewards = map(s->reward(mdp, s), valid_states)
end

function values(mdp::PRoverWorldMDP, policy::ValuePolicy)
    maxU = mapslices(maximum, policy.value_table, dims=2)
    return maxU[1:length(non_null_states(mdp))] # remove null_states
end

# function values(mdp::QuickMDP{GridWorld}, planner::MCTSPlanner)
#     valid_states = non_null_states(mdp)
#     U = []
#     for s in valid_states
#         u = 0
#         try
#             u = value(planner, s)
#         catch
#             # state not in tree
#         end
#         push!(U, u)
#     end
#     return U
# end

function one_based_policy!(policy)
    policy isa NothingPolicy && return
    # change the default action in the policy (all zeros) to all ones (if needed)
    if all(iszero, policy.policy)
        policy.policy[:] = ones(eltype(policy.policy), length(policy.policy))
    end
end
