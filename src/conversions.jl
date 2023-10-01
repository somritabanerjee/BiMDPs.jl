using DiscreteValueIteration

function solve_using_bilevel_mdp(mdp::RoverWorld.RoverWorldMDP; max_iters::Int64=100, init_state::Union{RoverWorld.State, Nothing} = nothing)
    sar_history = Vector{Tuple{Union{HLRoverWorld.HLState, LLRoverWorld.LLState}, Union{HLRoverWorld.HLAction, LLRoverWorld.LLAction}, Union{Float64, Nothing}}}()
    comp_time = 0.0
    disc_reward = 0.0
    # Create a HL MDP
    hl_mdp = HighLevelMDP(mdp)
    hl_solver = ValueIterationSolver(max_iterations=max_iters)
    hl_policy, hl_comp_time = @timed solve(hl_solver, hl_mdp)
    comp_time += hl_comp_time
    if isnothing(init_state)
        rng = Random.seed!(1)
        hl_s = HLRoverWorld.rand_starting_state(rng, hl_mdp)
    else
        hl_s = HLState_from_RoverState(init_state)
    end
    finished = false
    # While state is in bounds
    while !isnothing(hl_s) && HLRoverWorld.inbounds(hl_mdp, hl_s)
        hl_a = nothing;
        # Do a step of HL MDP
        for (hl_s_step, hl_a_step, _) in stepthrough(hl_mdp, hl_policy, hl_s, "s,a,r", max_steps=1)
            hl_s = hl_s_step; hl_a = hl_a_step;
        end
        push!(sar_history, (hl_s, hl_a, nothing))
        # Create a low level mdp using this HL action
        ll_mdp = LowLevelMDP(mdp, hl_s, hl_a)
        ll_solver = ValueIterationSolver(max_iterations=max_iters)
        ll_policy, ll_comp_time = @timed solve(ll_solver, ll_mdp)
        comp_time += ll_comp_time
        for (ll_s, ll_a, ll_r) in stepthrough(ll_mdp, ll_policy, ll_mdp.init_state, "s,a,r", max_steps=ll_mdp.max_time)
            push!(sar_history, (ll_s, ll_a, ll_r))
            disc_reward = ll_r + mdp.γ * disc_reward
        end
        # Update HL state
        last_ll_s = last(sar_history)[1]
        hl_s = HLState_from_LLState(last_ll_s, hl_s, hl_a, hl_mdp)
    end
    return comp_time, disc_reward, sar_history
end

function solve_using_finegrained_mdp(mdp::RoverWorld.RoverWorldMDP; max_iters::Int64=100, init_state::Union{RoverWorld.State, Nothing} = nothing)
    sar_history = Vector{Tuple{RoverWorld.State, RoverWorld.Action, Float64}}()
    disc_reward = 0.0
    solver = ValueIterationSolver(max_iterations=max_iters)
    policy, comp_time = @timed solve(solver, mdp)
    hr = HistoryRecorder()
    if isnothing(init_state)
        rng = Random.seed!(1)
        init_state = RoverWorld.rand_starting_state(rng, mdp)
    end
    history = simulate(hr, mdp, policy, init_state)
    sar_history = [(h[:s], h[:a], h[:r]) for h in history]
    disc_reward = discounted_reward(history)
    return comp_time, disc_reward, sar_history
end

function HLState_from_LLState(ll_s::LLRoverWorld.LLState, prev_hl_s::HLRoverWorld.HLState, prev_hl_a::HLRoverWorld.HLAction, hl_mdp::HLRoverWorld.HLRoverWorldMDP)
    new_visited = copy(prev_hl_s.visited)
    tgt_attempted = hl_mdp.tgts[prev_hl_a.tgt]
    ((tgt_x, tgt_y), (tgt_t0, tgt_tf), val) = tgt_attempted
    if ll_s.x == tgt_x && ll_s.y == tgt_y && (tgt_t0 <= ll_s.t <= tgt_tf)
        # Visited a target
        new_visited[prev_hl_a.tgt] = true
        return HLRoverWorld.HLState(ll_s.x, ll_s.y, ll_s.t, new_visited)
    else
        # Hit a terminal state
        return nothing
    end
end

function HLState_from_RoverState(s::RoverWorld.State)
    return HLRoverWorld.HLState(s.x, s.y, s.t, s.visited)
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