function collect_stepthrough(mdp::PRoverWorldMDP, policy; init_state = PState(1,1,1,true,fill(false, length(mdp.tgts))))
    steps = collect(stepthrough(mdp, policy, init_state, "s,a,r", max_steps=100))
    return steps
end