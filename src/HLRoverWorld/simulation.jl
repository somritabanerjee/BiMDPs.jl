function collect_stepthrough(mdp::HLRoverWorldMDP, policy; init_state = HLState(1,1,1,fill(false, length(mdp.tgts))))
    if !inbounds(mdp, init_state)
        println("Initial state $init_state is out of bounds.")
        return
    end
    steps = collect(stepthrough(mdp, policy, init_state, "s,a,r", max_steps=mdp.max_time))
    return steps
end