function collect_stepthrough(mdp::HLRoverWorldMDP, policy; init_state = HLState(1,1,1,fill(false, length(mdp.tgts))))
    steps = collect(stepthrough(mdp, policy, init_state, "s,a,r", max_steps=mdp.max_time))
    return steps
end