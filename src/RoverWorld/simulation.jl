function print_stepthrough(mdp::RoverWorldMDP, policy; init_state = State(1,1,1,fill(false, length(mdp.reward_vals))))
    for (s,a,r) in stepthrough(mdp, policy, init_state, "s,a,r", max_steps=100)
        @info "In state ($(s.x), $(s.y)), taking action $a, receiving reward $r"
    end
end

function collect_stepthrough(mdp::RoverWorldMDP, policy; init_state = State(1,1,1,fill(false, length(mdp.reward_vals))))
    steps = collect(stepthrough(mdp, policy, init_state, "s,a,r", max_steps=100))
    return steps
end

using Statistics
function run_simulation(mdp::RoverWorldMDP, policy, q_learning_policy, sarsa_policy; N_sim = 10000, max_steps = mdp.max_time)
	mean_std(X) = (μ=mean(X), σ=std(X), r=X)

    rollsim = RolloutSimulator(max_steps=max_steps);

    ## Debug steps
    #=
    temp = simulate(rollsim, mdp, policy)
    println("reward of 1 sim using VI policy:",temp)

    for i in 1:N_sim
        temp = simulate(rollsim, mdp, policy)
        if temp != 100
            error("reward of $i th sim using VI policy:",temp)
        end
    end
    =#

	stats_vi = mean_std([simulate(rollsim, mdp, policy) for _ in 1:N_sim])
	stats_ql = mean_std([simulate(rollsim, mdp, q_learning_policy) for _ in 1:N_sim])
	stats_sarsa = mean_std([simulate(rollsim, mdp, sarsa_policy) for _ in 1:N_sim])

	results = (value_iteration=stats_vi, q_learning=stats_ql, sarsa=stats_sarsa)
end