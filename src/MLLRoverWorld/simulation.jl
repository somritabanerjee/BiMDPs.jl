function collect_stepthrough(mdp::MLLRoverWorldMDP, policy; init_state = MLLState(1,1,1))
    steps = collect(stepthrough(mdp, policy, init_state, "s,a,r", max_steps=mdp.max_time))
    return steps
end