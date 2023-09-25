using DiscreteValueIteration

function convert_to_bilevel(mdp::RoverWorld.RoverWorldMDP; hl_iters = 50, ll_iters = 50)
    hl_mdp = HighLevelMDP(mdp)
    # HLRoverWorld.test_state_indexing(hl_mdp)
    hl_solver = ValueIterationSolver(max_iterations=hl_iters)
    hl_policy = solve(hl_solver, hl_mdp)
    return hl_mdp, hl_policy
    # ll_mdp = LowLevelMDP(mdp, hl_mdp, policy)
    # ll_solver = ValueIterationSolver(max_iterations=ll_iters)
    # ll_policy = solve(ll_solver, ll_mdp)
    # return ll_policy
end


function HighLevelMDP(mdp::RoverWorld.RoverWorldMDP)
    return HLRoverWorld.HLRoverWorldMDP(grid_size = mdp.grid_size,
                            max_time = mdp.max_time,
                            null_xy = mdp.null_xy,
                            γ = mdp.γ,
                            tgts = mdp.tgts,
                            exit_xys = mdp.exit_xys)
end