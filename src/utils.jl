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
                hl_mdp = HighLevelMDP(mdp)
                hl_solver = ValueIterationSolver(max_iterations=max_iters)
                hl_policy, comp_time = @timed solve(hl_solver, hl_mdp)
                verbose && println("Comp_time of HL part of $solver_name after $max_iters iterations: ", comp_time)
                comp_times[i] = comp_time
                ll_comp_times = zeros(N_sim)
                rewards = zeros(N_sim)
                for j in 1:N_sim
                    verbose && println("Simulation $j")
                    states_vec = []
                    actions_vec = []
                    rewards_vec = [] # TODO remove
                    # Do one step of HL MDP
                    rng = Random.seed!(1)
                    hl_s = HLRoverWorld.rand_initial_state(rng, hl_mdp)
                    verbose && println("HL: initial state $hl_s")
                    while HLRoverWorld.inbounds(hl_mdp, hl_s)
                        verbose && println("  in bounds")
                        hl_a = nothing; hl_r = nothing;
                        for (hl_s_step, hl_a_step, hl_r_step) in stepthrough(hl_mdp, hl_policy, hl_s, "s,a,r", max_steps=1)
                            hl_s = hl_s_step; hl_a = hl_a_step; hl_r = hl_r_step;
                            verbose && println("HL: targeting action $hl_a action $(hl_mdp.tgts[hl_a.tgt])")
                        end
                        # Create a low level mdp using this HL action
                        ll_mdp = LowLevelMDP(mdp, hl_s, hl_a)
                        ll_solver = ValueIterationSolver(max_iterations=max_iters)
                        ll_policy, ll_comp_time = @timed solve(ll_solver, ll_mdp)
                        ll_comp_times[j] += ll_comp_time
                        for (ll_s, ll_a, ll_r) in stepthrough(ll_mdp, ll_policy, ll_mdp.init_state, "s,a,r", max_steps=ll_mdp.max_time)
                            verbose && println("LL: state $ll_s")
                            verbose && ll_r > 0 && println("   reward $ll_r")
                            push!(states_vec, ll_s)
                            push!(actions_vec, ll_a)
                            push!(rewards_vec, ll_r)
                            rewards[j] = ll_r + mdp.γ * rewards[j]
                        end
                        last_ll_s = states_vec[end]
                        new_visited = copy(hl_s.visited)
                        if last_ll_s.x == hl_mdp.tgts[hl_a.tgt][1][1] && last_ll_s.y == hl_mdp.tgts[hl_a.tgt][1][2] && (hl_mdp.tgts[hl_a.tgt][2][1] <= last_ll_s.t <= hl_mdp.tgts[hl_a.tgt][2][2])
                            new_visited[hl_a.tgt] = true
                            verbose && println("  did visit target")
                            new_hl_s = HLRoverWorld.HLState(last_ll_s.x, last_ll_s.y, last_ll_s.t, new_visited)
                            verbose && println("HL new state: $new_hl_s")
                            hl_s = new_hl_s
                        else
                            verbose && println("  hit terminal state")
                            break
                        end                        
                    end
                    verbose && println("Rewards for simulation $j: $(rewards[j])")
                end
                comp_times[i] += mean(ll_comp_times)
                verbose && println("Adding a mean time for LL comp: $(mean(ll_comp_times))")
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
