using POMDPs
using POMDPSimulators
using POMDPPolicies
using Statistics

function optimality_vs_compute(mdp::RoverWorld.RoverWorldMDP, solvers_iters_Nsim::Vector{Tuple{String,Vector{Int},Int}}; verbose = true)
    results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}() # dictionary mapping solver_name to (comp_time, mean_reward, stddev_reward)
    for (solver_name, max_iters_vec, N_sim) in solvers_iters_Nsim
        comp_times = Vector{Float64}(undef, length(max_iters_vec))
        mean_rewards = Vector{Float64}(undef, length(max_iters_vec))
        stddev_rewards = Vector{Float64}(undef, length(max_iters_vec))
        for (i, max_iters) in enumerate(max_iters_vec)
            if solver_name == "vi"
                solver = ValueIterationSolver(max_iterations=max_iters)
                mdp_to_solve = mdp
            elseif solver_name == "qlearning"
                solver = QLearningSolver(n_episodes=max_iters,
                                        learning_rate=0.8,
                                        exploration_policy=EpsGreedyPolicy(mdp, 0.5),
                                        verbose=false)
                mdp_to_solve = mdp
            elseif solver_name == "sarsa"
                solver = SARSASolver(n_episodes=max_iters,
                                        learning_rate=0.8,
                                        exploration_policy=EpsGreedyPolicy(mdp, 0.5),
                                        verbose=false)
                mdp_to_solve = mdp
            elseif solver_name == "bl_vi"
                mdp_to_solve = HighLevelMDP(mdp)
                solver = ValueIterationSolver(max_iterations=max_iters)
            else
                error("Unknown solver: $solver_name")
            end
            policy, comp_time = @timed solve(solver, mdp_to_solve)
            verbose && println("Comp_time of $solver_name after $max_iters iterations: ", comp_time)
            comp_times[i] = comp_time

            mean_std(X) = (μ=mean(X), σ=std(X), r=X)
            rollsim = RolloutSimulator(max_steps=mdp_to_solve.max_time)
            stats_sim = mean_std([simulate(rollsim, mdp_to_solve, policy) for _ in 1:N_sim])
            verbose && println("Reward of $solver_name after $N_sim simulations: μ = $(stats_sim.μ), σ = $(stats_sim.σ)")
            mean_rewards[i] = stats_sim.μ
            stddev_rewards[i] = stats_sim.σ
        end
        results[solver_name] = (comp_times, mean_rewards, stddev_rewards)
    end
    return results
end