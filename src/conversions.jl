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

function LowLevelMDP(mdp::RoverWorld.RoverWorldMDP, hl_s::HLRoverWorld.HLState, hl_a::HLRoverWorld.HLAction)
    return LLRoverWorld.LLRoverWorldMDP(grid_size = mdp.grid_size,
                            max_time = mdp.max_time,
                            null_xy = mdp.null_xy,
                            p_transition = mdp.p_transition,
                            γ = mdp.γ,
                            current_tgt = mdp.tgts[hl_a.tgt],
                            obstacles = mdp.obstacles,
                            exit_xys = mdp.exit_xys,
                            init_state = LLRoverWorld.LLState(hl_s.x, hl_s.y, hl_s.t))
end

function test_LL()
    ll_mdp = LLRoverWorld.LLRoverWorldMDP(grid_size = (20,20),
                    max_time = 10,
                    null_xy = (-1,-1),
                    p_transition = 1.0,
                    γ = 0.95,
                    current_tgt = ((5,1),(1,10),50),
                    obstacles = [((3,1), (1,10), -5)],
                    exit_xys = [],
                    init_state = LLRoverWorld.LLState(2, 6, 1)
                    )
    ll_solver = ValueIterationSolver(max_iterations=200)
    # hl_mdp = HLRoverWorld.HLRoverWorldMDP()
    # a_hl = POMDPs.actions(hl_mdp)
    # println("a_hl: $a_hl")
    # ll_mdp = LLRoverWorld.LLRoverWorldMDP()
    # a_ll = POMDPs.actions(ll_mdp)
    # println("a_ll: $a_ll")
    LLRoverWorld.test_state_indexing(ll_mdp)
    println("passed")

    LLRoverWorld.print_details(ll_mdp)
    ll_policy, ll_comp_time = @timed solve(ll_solver, ll_mdp)
    for (ll_s, ll_a, ll_r) in stepthrough(ll_mdp, ll_policy, ll_mdp.init_state, "s,a,r", max_steps=ll_mdp.max_time)
        println("LL: in state $ll_s, taking action $ll_a, received reward $ll_r")
        # rewards[j] += ll_r
    end
    # return ll_mdp, ll_policy, ll_comp_time
end