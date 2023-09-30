using POMDPs
using POMDPSimulators
using POMDPPolicies
using Statistics
using DiscreteValueIteration
using TabularTDLearning

function optimality_vs_compute(mdp::RoverWorld.RoverWorldMDP, solvers_iters_Nsim::Vector{Tuple{String,Vector{Int},Int}}; verbose = true)
    results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}() # dictionary mapping solver_name to (comp_time, mean_reward, stddev_reward)
    for (solver_name, max_iters_vec, N_sim) in solvers_iters_Nsim
        comp_times = Vector{Float64}(undef, length(max_iters_vec))
        mean_rewards = Vector{Float64}(undef, length(max_iters_vec))
        stddev_rewards = Vector{Float64}(undef, length(max_iters_vec))
        for (i, max_iters) in enumerate(max_iters_vec)
            if solver_name == "bl_vi"
                hl_mdp = HighLevelMDP(mdp)
                hl_solver = ValueIterationSolver(max_iterations=max_iters)
                hl_policy, comp_time = @timed solve(hl_solver, hl_mdp)
                verbose && println("Comp_time of HL part of $solver_name after $max_iters iterations: ", comp_time)
                println("Comp_time of HL part of $solver_name after $max_iters iterations: ", comp_time)
                comp_times[i] = comp_time
                ll_comp_times = zeros(N_sim)
                rewards = zeros(N_sim)
                for j in 1:N_sim
                    verbose && println("Simulation $j")
                    for (s, a, r) in stepthrough(hl_mdp, hl_policy, "s,a,r", max_steps=hl_mdp.max_time)
                        verbose && println("HL: in state $s, taking action $a, received reward $r")
                        verbose && println("Creating LL MDP from state $s to do action $(hl_mdp.tgts[a.tgt])")
                        ll_mdp = LowLevelMDP(mdp, s, a)
                        verbose && println("ll_mdp tgt: $(ll_mdp.current_tgt)")
                        verbose && println("ll_mdp obstacles: $(ll_mdp.obstacles)")
                        verbose && println("ll_mdp exit_xys: $(ll_mdp.exit_xys)")
                        ll_solver = ValueIterationSolver(max_iterations=max_iters)
                        ll_policy, ll_comp_time = @timed solve(ll_solver, ll_mdp)
                        verbose && println("Comp_time of LL part of $solver_name after $max_iters iterations: ", ll_comp_time)
                        ll_comp_times[j] += ll_comp_time
                        for (ll_s, ll_a, ll_r) in stepthrough(ll_mdp, ll_policy, ll_mdp.init_state, "s,a,r", max_steps=ll_mdp.max_time)
                            verbose && println("LL: in state $ll_s, taking action $ll_a, received reward $ll_r")
                            rewards[j] += ll_r
                        end
                        # rewards[j] += r
                    end
                    println("Rewards for simulation $j: $(rewards[j])")
                end
                comp_times[i] += mean(ll_comp_times)
                println("Adding a mean time for LL comp: $(mean(ll_comp_times))")
                stats_sim = mean_std(rewards)
                mean_rewards[i] = stats_sim.μ
                println("Reward of $solver_name after $N_sim simulations: μ = $(stats_sim.μ), σ = $(stats_sim.σ)")
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
                verbose && println("Reward of $solver_name after $N_sim simulations: μ = $(stats_sim.μ), σ = $(stats_sim.σ)")
                mean_rewards[i] = stats_sim.μ
                stddev_rewards[i] = stats_sim.σ
            end
        end
        results[solver_name] = (comp_times, mean_rewards, stddev_rewards)
    end
    return results
end

mean_std(X) = (μ=mean(X), σ=std(X), r=X)
