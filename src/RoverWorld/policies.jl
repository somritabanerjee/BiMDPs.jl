# create policy grid showing the best action in each state
function policy_grid(policy::Policy, xmax::Int, ymax::Int; t::Int = 1, visited::Vector{Bool} = fill(false, length(policy.mdp.tgts)))
    arrows = Dict(UP => "↑",
                  DOWN => "↓",
                  LEFT => "←",
                  RIGHT => "→")

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

function optimality_vs_compute(mdp::RoverWorldMDP, solvers_iters_Nsim::Vector{Tuple{String,Vector{Int},Int}}; verbose = true)
    results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}() # dictionary mapping solver_name to (comp_time, mean_reward, stddev_reward)
    for (solver_name, max_iters_vec, N_sim) in solvers_iters_Nsim
        comp_times = Vector{Float64}(undef, length(max_iters_vec))
        mean_rewards = Vector{Float64}(undef, length(max_iters_vec))
        stddev_rewards = Vector{Float64}(undef, length(max_iters_vec))
        for (i, max_iters) in enumerate(max_iters_vec)
            if solver_name == "vi"
                solver = ValueIterationSolver(max_iterations=max_iters)
            elseif solver_name == "qlearning"
                solver = QLearningSolver(n_episodes=max_iters,
                                        learning_rate=0.8,
                                        exploration_policy=EpsGreedyPolicy(mdp, 0.5),
                                        verbose=false)
            elseif solver_name == "sarsa"
                solver = SARSASolver(n_episodes=max_iters,
                                        learning_rate=0.8,
                                        exploration_policy=EpsGreedyPolicy(mdp, 0.5),
                                        verbose=false)
            end
            policy, comp_time = @timed solve(solver, mdp)
            verbose && println("Comp_time of $solver_name after $max_iters iterations: ", comp_time)
            comp_times[i] = comp_time

            mean_std(X) = (μ=mean(X), σ=std(X), r=X)
            rollsim = RolloutSimulator(max_steps=mdp.max_time)
            stats_sim = mean_std([simulate(rollsim, mdp, policy) for _ in 1:N_sim])
            verbose && println("Reward of $solver_name after $N_sim simulations: μ = $(stats_sim.μ), σ = $(stats_sim.σ)")
            mean_rewards[i] = stats_sim.μ
            stddev_rewards[i] = stats_sim.σ
        end
        results[solver_name] = (comp_times, mean_rewards, stddev_rewards)
    end
    return results
end