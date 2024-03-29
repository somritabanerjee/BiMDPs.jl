using DiscreteValueIteration
using BeliefGridValueIteration
using BasicPOMCP
using PointBasedValueIteration
using QMDP
using BeliefUpdaters

function solve_using_bilevel_mdp(mdp::RoverWorld.RoverWorldMDP; max_iters::Int64=100, init_state::Union{RoverWorld.State, Nothing} = nothing, hl_mdp = nothing, hl_policy = nothing, hl_comp_time = nothing, verbose = false)
    sar_history = Vector{Tuple{Union{HLRoverWorld.HLState, LLRoverWorld.LLState}, Union{HLRoverWorld.HLAction, LLRoverWorld.LLAction}, Union{Float64, Nothing}}}()
    comp_time = 0.0
    disc_reward = 0.0
    disc = 1.0
    if isnothing(hl_mdp)
        # Create a HL MDP
        hl_mdp = HighLevelMDP(mdp)
        hl_solver = ValueIterationSolver(max_iterations=max_iters)
        hl_policy, hl_comp_time = @timed solve(hl_solver, hl_mdp)
        verbose && println("Done solving HL MDP")
    else 
        verbose && println("Using provided HL MDP")
    end
    comp_time += hl_comp_time
    if isnothing(init_state)
        rng = Random.default_rng()
        hl_s = HLRoverWorld.rand_starting_state(rng, hl_mdp)
    else
        hl_s = HLState_from_RoverState(init_state)
    end
    finished = false
    # While state is in bounds
    while !isnothing(hl_s) && HLRoverWorld.inbounds(hl_mdp, hl_s)
        verbose && println("HL state: $hl_s")
        hl_a = nothing;
        # Do a step of HL MDP
        for (hl_s_step, hl_a_step, _) in stepthrough(hl_mdp, hl_policy, hl_s, "s,a,r", max_steps=1)
            hl_s = hl_s_step; hl_a = hl_a_step;
        end
        verbose && println("HL action: $hl_a")
        push!(sar_history, (hl_s, hl_a, nothing))
        # Create a low level mdp using this HL action
        ll_mdp = LowLevelMDP(mdp, hl_s, hl_a)
        ll_solver = ValueIterationSolver(max_iterations=max_iters)
        ll_policy, ll_comp_time = @timed solve(ll_solver, ll_mdp)
        verbose && println("Done solving LL MDP")
        comp_time += ll_comp_time
        for (ll_s, ll_a, ll_r) in stepthrough(ll_mdp, ll_policy, ll_mdp.init_state, "s,a,r", max_steps=ll_mdp.max_time)
            push!(sar_history, (ll_s, ll_a, ll_r))
            disc *= mdp.γ
            disc_reward += disc * ll_r
            verbose && println("    LL: in state $ll_s, taking action $ll_a with reward $ll_r")
        end
        verbose && println("Done stepthrough LL MDP")
        # Update HL state
        last_ll_s = last(sar_history)[1]
        verbose && println("last LL state: $last_ll_s")
        new_hl_s = HLState_from_LLState(last_ll_s, hl_s, hl_a, hl_mdp)
        verbose && println("NEW HL state: $new_hl_s")
        if new_hl_s == hl_s
            break
        end
        hl_s = new_hl_s
    end
    return comp_time, disc_reward, sar_history
end

function solve_using_bilevel_mdp(mdp::MRoverWorld.MRoverWorldMDP; max_iters::Int64=100, init_state::Union{MRoverWorld.MState, Nothing} = nothing, verbose = false)
    sar_history = Vector{Tuple{Union{HLRoverWorld.HLState, MLLRoverWorld.MLLState}, Union{HLRoverWorld.HLAction, MLLRoverWorld.MLLAction}, Union{Float64, Nothing}}}()
    comp_time = 0.0
    disc_reward = 0.0
    disc = 1.0
    # Create a HL MDP
    hl_mdp = HighLevelMDP(mdp)
    hl_solver = ValueIterationSolver(max_iterations=max_iters)
    hl_policy, hl_comp_time = @timed solve(hl_solver, hl_mdp)
    verbose && println("Done solving HL MDP")
    comp_time += hl_comp_time
    if isnothing(init_state)
        rng = Random.default_rng()
        hl_s = HLRoverWorld.rand_starting_state(rng, hl_mdp)
    else
        hl_s = HLState_from_RoverState(init_state)
    end
    finished = false
    # While state is in bounds
    while !isnothing(hl_s) && HLRoverWorld.inbounds(hl_mdp, hl_s)
        verbose && println("HL state: $hl_s")
        hl_a = nothing;
        # Do a step of HL MDP
        for (hl_s_step, hl_a_step, _) in stepthrough(hl_mdp, hl_policy, hl_s, "s,a,r", max_steps=1)
            hl_s = hl_s_step; hl_a = hl_a_step;
        end
        verbose && println("HL action: $hl_a")
        push!(sar_history, (hl_s, hl_a, nothing))
        # Create a low level mdp using this HL action
        ll_mdp = MLowLevelMDP(mdp, hl_s, hl_a)
        MLLRoverWorld.test_state_indexing(ll_mdp)
        ll_solver = ValueIterationSolver(max_iterations=max_iters)
        ll_policy, ll_comp_time = @timed solve(ll_solver, ll_mdp)
        verbose && println("Done solving LL MDP")
        comp_time += ll_comp_time
        for (ll_s, ll_a, ll_r) in stepthrough(ll_mdp, ll_policy, ll_mdp.init_state, "s,a,r", max_steps=ll_mdp.max_time)
            push!(sar_history, (ll_s, ll_a, ll_r))
            disc *= mdp.γ
            disc_reward += disc * ll_r
            verbose && println("    LL: in state $ll_s, taking action $ll_a with reward $ll_r")
        end
        verbose && println("Done stepthrough LL MDP")
        # Update HL state
        last_ll_s = last(sar_history)[1]
        verbose && println("last LL state: $last_ll_s")
        new_hl_s = HLState_from_MLLState(last_ll_s, hl_s, hl_a, hl_mdp)
        verbose && println("NEW HL state: $new_hl_s")
        if new_hl_s == hl_s
            break
        end
        hl_s = new_hl_s
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
        rng = Random.default_rng()
        init_state = RoverWorld.rand_starting_state(rng, mdp)
    end
    history = simulate(hr, mdp, policy, init_state)
    sar_history = [(h[:s], h[:a], h[:r]) for h in history]
    disc_reward = discounted_reward(history)
    return comp_time, disc_reward, sar_history
end

function solve_using_finegrained_mdp(mdp::MRoverWorld.MRoverWorldMDP; max_iters::Int64=100, init_state::Union{MRoverWorld.MState, Nothing} = nothing)
    sar_history = Vector{Tuple{MRoverWorld.MState, MRoverWorld.MAction, Float64}}()
    disc_reward = 0.0
    solver = ValueIterationSolver(max_iterations=max_iters)
    policy, comp_time = @timed solve(solver, mdp)
    hr = HistoryRecorder()
    if isnothing(init_state)
        rng = Random.default_rng()
        init_state = MRoverWorld.rand_starting_state(rng, mdp)
    end
    history = simulate(hr, mdp, policy, init_state)
    sar_history = [(h[:s], h[:a], h[:r]) for h in history]
    disc_reward = discounted_reward(history)
    return comp_time, disc_reward, sar_history
end

function solve_using_finegrained_mdp(mdp::PRoverWorld.PRoverWorldMDP; max_iters::Int64=100, init_state::Union{PRoverWorld.PState, Nothing} = nothing)
    sar_history = Vector{Tuple{PRoverWorld.PState, PRoverWorld.PAction, Float64}}()
    disc_reward = 0.0
    
    # solver = BeliefGridValueIterationSolver(max_iterations=2, m = 2)
    # policy, comp_time = @timed solve(solver, mdp)
    # println("Solve time: $comp_time seconds")
    # hr = HistoryRecorder()
    # if isnothing(init_state)
    #     rng = Random.default_rng()
    #     init_state = PRoverWorld.rand_starting_state(rng, mdp)
    # end
    # history = simulate(hr, mdp, policy, init_state)
    # sar_history = [(h[:s], h[:a], h[:r]) for h in history]
    # disc_reward = discounted_reward(history)
    # return comp_time, disc_reward, sar_history

    # solver = PBVISolver(max_iterations = 2)
    solver = QMDPSolver(max_iterations=10,
                    belres=1e-3,
                    verbose=true
                   ) 
    policy, comp_time = @timed solve(solver, mdp)
    println("Solve time: $comp_time seconds")

    # policy = FunctionPolicy(b->PRoverWorld.UP)
    # filt = DiscreteUpdater(mdp)
    # println("About to step")
    # for (b,a,o,r) in stepthrough(mdp, policy, filt, "b,a,o,r", max_steps=2)
    #     println("State was $s,")
    #     println("action $a was taken,")
    #     println("reward $r was received,")
    #     # println("and observation $o was received.\n")
    # end

    # policy = RandomPolicy(mdp)
    for (s,a,r) in stepthrough(mdp, policy, "s,a,r", max_steps = 2)
        println("State was $s,")
        println("action $a was taken,")
        println("reward $r was received,")
    end

    # solver = POMCPSolver()
    # planner, comp_time = @timed solve(solver, mdp)
    # println("Solve time: $comp_time seconds")
    # for (s, a, o, r) in stepthrough(mdp, planner, "s,a,o,r", max_steps=10)
    #     println("State was $s,")
    #     println("action $a was taken,")
    #     println("reward $r was received,")
    #     println("and observation $o was received.\n")
    # end
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

function HLState_from_MLLState(ll_s::MLLRoverWorld.MLLState, prev_hl_s::HLRoverWorld.HLState, prev_hl_a::HLRoverWorld.HLAction, hl_mdp::HLRoverWorld.HLRoverWorldMDP)
    new_visited = copy(prev_hl_s.visited)
    attempt = prev_hl_a.tgt > 0 ? :tgt : :exit
    if attempt == :tgt
        tgt_attempted = hl_mdp.tgts[prev_hl_a.tgt]
        ((tgt_x, tgt_y), (tgt_t0, tgt_tf), val) = tgt_attempted
        if ll_s.x == tgt_x && ll_s.y == tgt_y && (tgt_t0 <= ll_s.t <= tgt_tf) && ll_s.measured
            # Visited a target
            new_visited[prev_hl_a.tgt] = true
            return HLRoverWorld.HLState(ll_s.x, ll_s.y, ll_s.t, new_visited)
        else
            # Hit a terminal state
            return nothing
        end
    elseif attempt == :exit
        if ll_s.x == hl_mdp.exit_xys[-prev_hl_a.tgt][1] && ll_s.y == hl_mdp.exit_xys[-prev_hl_a.tgt][2]
            return HLRoverWorld.HLState(ll_s.x, ll_s.y, ll_s.t, new_visited)
        else
            # Hit a terminal state
            return nothing
        end
    end
end

function HLState_from_RoverState(s::Union{RoverWorld.State, MRoverWorld.MState})
    return HLRoverWorld.HLState(s.x, s.y, s.t, s.visited)
end


function HighLevelMDP(mdp::Union{RoverWorld.RoverWorldMDP, MRoverWorld.MRoverWorldMDP})
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
                            obstacles_grid = mdp.obstacles_grid,
                            exit_xys = mdp.exit_xys,
                            include_measurement = mdp.include_measurement,
                            measure_reward = mdp.measure_reward,
                            init_state = LLRoverWorld.LLState(hl_s.x, hl_s.y, hl_s.t))
end

function MLowLevelMDP(mdp::MRoverWorld.MRoverWorldMDP, hl_s::HLRoverWorld.HLState, hl_a::HLRoverWorld.HLAction)
    return MLLRoverWorld.MLLRoverWorldMDP(grid_size = mdp.grid_size,
                            max_time = mdp.max_time,
                            null_xy = mdp.null_xy,
                            p_transition = mdp.p_transition,
                            γ = mdp.γ,
                            current_tgt = hl_a.tgt > 0 ? mdp.tgts[hl_a.tgt] : (mdp.exit_xys[-hl_a.tgt], (1, mdp.max_time), 0.0),
                            obstacles_grid = mdp.obstacles_grid,
                            exit_xys = mdp.exit_xys,
                            measure_reward = mdp.measure_reward,
                            init_state = MLLRoverWorld.MLLState(hl_s.x, hl_s.y, hl_s.t, false))
end