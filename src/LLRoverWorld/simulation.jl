function collect_stepthrough(mdp::LLRoverWorldMDP, policy; init_state = LLState(1,1,1))
    steps = collect(stepthrough(mdp, policy, init_state, "s,a,r", max_steps=mdp.max_time))
    return steps
end