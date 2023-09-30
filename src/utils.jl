using POMDPs
using POMDPSimulators
using POMDPPolicies
using Statistics
using DiscreteValueIteration
using TabularTDLearning
using Random

function optimality_vs_compute(mdp::RoverWorld.RoverWorldMDP, solvers_iters_Nsim::Vector{Tuple{String,Vector{Int},Int}}; verbose = false)
    results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}() # dictionary mapping solver_name to (comp_time, mean_reward, stddev_reward)
    for (solver_name, max_iters_vec, N_sim) in solvers_iters_Nsim
        comp_times = Vector{Float64}(undef, length(max_iters_vec))
        mean_rewards = Vector{Float64}(undef, length(max_iters_vec))
        stddev_rewards = Vector{Float64}(undef, length(max_iters_vec))
        for (i, max_iters) in enumerate(max_iters_vec)
            if solver_name == "bl_vi"
                sim_comp_times = zeros(N_sim)
                rewards = zeros(N_sim)
                for j in 1:N_sim
                    comp_time, discounted_reward, sar_history = solve_using_bilevel_mdp(mdp, max_iters=max_iters)
                    sim_comp_times[j] = comp_time
                    rewards[j] = discounted_reward
                end
                comp_times[i] = mean(sim_comp_times)
                stats_sim = mean_std(rewards)
                println("Reward of $solver_name after $N_sim simulations: μ = $(stats_sim.μ), σ = $(stats_sim.σ)")
                mean_rewards[i] = stats_sim.μ
                stddev_rewards[i] = stats_sim.σ
            else
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
                else
                    error("Unknown solver: $solver_name")
                end
                policy, comp_time = @timed solve(solver, mdp)
                verbose && println("Comp_time of $solver_name after $max_iters iterations: ", comp_time)
                comp_times[i] = comp_time

                rollsim = RolloutSimulator(max_steps=mdp.max_time)
                stats_sim = mean_std([simulate(rollsim, mdp, policy) for _ in 1:N_sim])
                println("Reward of $solver_name after $N_sim simulations: μ = $(stats_sim.μ), σ = $(stats_sim.σ)")
                mean_rewards[i] = stats_sim.μ
                stddev_rewards[i] = stats_sim.σ
            end
        end
        results[solver_name] = (comp_times, mean_rewards, stddev_rewards)
    end
    return results
end

mean_std(X) = (μ=mean(X), σ=std(X), r=X)

function test_LL()
    ll_mdp = LLRoverWorld.LLRoverWorldMDP(grid_size = (20,20),
                    max_time = 10,
                    null_xy = (-1,-1),
                    p_transition = 1.0,
                    γ = 0.95,
                    current_tgt = ((5,1),(1,10),50),
                    obstacles = [((3,1), (1,10), -5)],
                    exit_xys = [],
                    init_state = LLRoverWorld.LLState(2, 6, 1)
                    )
    ll_solver = ValueIterationSolver(max_iterations=200)
    LLRoverWorld.test_state_indexing(ll_mdp)
    println("passed")
    LLRoverWorld.print_details(ll_mdp)
    ll_policy, ll_comp_time = @timed solve(ll_solver, ll_mdp)
    for (ll_s, ll_a, ll_r) in stepthrough(ll_mdp, ll_policy, ll_mdp.init_state, "s,a,r", max_steps=ll_mdp.max_time)
        println("LL: in state $ll_s, taking action $ll_a, received reward $ll_r")
    end
end