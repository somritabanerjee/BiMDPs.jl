function collect_stepthrough(mdp::MRoverWorldMDP, policy; init_state = MState(1,1,1,fill(false, length(mdp.tgts)),fill(false, length(mdp.tgts))))
    steps = collect(stepthrough(mdp, policy, init_state, "s,a,r", max_steps=100))
    return steps
end