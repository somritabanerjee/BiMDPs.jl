# create policy grid showing the best action in each state
function policy_grid(policy::Policy, xmax::Int, ymax::Int; t::Int = 1, visited::Vector{Bool} = fill(false, length(policy.mdp.tgts)))
    arrows = Dict(UP => "↑",
                  DOWN => "↓",
                  LEFT => "←",
                  RIGHT => "→",
                  MEASURE => "=")

    grid = Array{String}(undef, xmax, ymax)
    for x = 1:xmax, y = 1:xmax
        s = State(x, y, t, visited)
        grid[x,y] = arrows[action(policy, s)]
    end

    return grid
end

function values(mdp::RoverWorldMDP, policy::Policy)
    valid_states = non_null_states(mdp)
    U = map(s->value(policy, s), valid_states)
end

struct NothingPolicy <: Policy end

# Use this to get a stationary grid of rewards
function values(mdp::RoverWorldMDP, policy::Union{NothingPolicy, FunctionPolicy})
    valid_states = non_null_states(mdp)
    rewards = map(s->reward(mdp, s), valid_states)
end

function values(mdp::RoverWorldMDP, policy::ValuePolicy)
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

function solve_using(solver_name::String, mdp::RoverWorldMDP; vi_max_iterations=50, q_n_episodes=500, sarsa_n_episodes=500)
    if solver_name == "vi"
        solver = ValueIterationSolver(max_iterations=vi_max_iterations)
        policy = solve(solver, mdp)
    elseif solver_name == "qlearning"
        solver = QLearningSolver(n_episodes=q_n_episodes,
                                learning_rate=0.8,
                                exploration_policy=EpsGreedyPolicy(mdp, 0.5),
                                verbose=false)
        policy = solve(solver, mdp)
    elseif solver_name == "sarsa"
        solver = SARSASolver(n_episodes=sarsa_n_episodes,
                                learning_rate=0.8,
                                exploration_policy=EpsGreedyPolicy(mdp, 0.5),
                                verbose=false)
        policy = solve(solver, mdp)
    else
        error("Unknown solver $solver_name")
    end
end
