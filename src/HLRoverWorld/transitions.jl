function POMDPs.transition(mdp::HLRoverWorldMDP, s::HLState, a::HLAction)
    verbose = false
    verbose && println("Current HLState: $s")
    verbose && println("Current action: $a")
    if ((s.x, s.y) in mdp.exit_xys || s.t >= mdp.max_time) # transition to terminal HLState
        verbose && println("Going to terminal HLState")
        return Deterministic(HLState(mdp.null_xy[1], mdp.null_xy[2], min(s.t + 1, mdp.max_time), s.visited))
    end

    # Check if we're at a target, to update the visited list
    visited = copy(s.visited)
    for (tgt_id, ((x, y), (t0, tf), val)) in mdp.tgts
        if (s.x, s.y) == (x, y) && t0 <= s.t <= tf
            verbose && println("At target $tgt_id at $s")
            visited[tgt_id] = true
        end
    end
    s_up = HLState(s.x, s.y, s.t, visited)

    next_states = HLState[]
    probabilities = Float64[]
    if a.tgt>0 # going to a target
        verbose && println("Going to target ID $(a.tgt) : $(mdp.tgts[a.tgt])")
        sp = go_to_target(mdp, s_up, a.tgt)
        push!(next_states, sp)
        push!(probabilities, 1.0)
    else # going to an exit xy
        verbose && println("Going to exit xy $(a.tgt) : $(mdp.exit_xys[-a.tgt])")
        sp = go_to_exit_xy(mdp, s_up, -a.tgt)
        push!(next_states, sp)
        push!(probabilities, 1.0)
    end

    verbose && println("Next states: $next_states with probabilities $probabilities")
    return SparseCat(next_states, probabilities)
end

function go_to_target(mdp::HLRoverWorldMDP, s::HLState, tgt_id::Int)
    ((tgt_x, tgt_y), (t0, tf), r) = mdp.tgts[tgt_id]
    time_to_target = manhattan_distance((s.x, s.y), (tgt_x, tgt_y))
    if time_to_target == 0
        return HLState(tgt_x, tgt_y, s.t + 1, s.visited)
    elseif 1 ≤ s.t + time_to_target + 1 ≤ mdp.max_time
        return HLState(tgt_x, tgt_y, s.t + time_to_target + 1, s.visited)
    else
        return HLState(s.x, s.y, mdp.max_time, s.visited)
    end
end

function go_to_exit_xy(mdp::HLRoverWorldMDP, s::HLState, exit_id::Int)
    exit_xy = mdp.exit_xys[exit_id]
    time_to_exit = manhattan_distance((s.x, s.y), exit_xy)
    if 1 ≤ s.t + time_to_exit ≤ mdp.max_time
        return HLState(exit_xy[1], exit_xy[2], s.t + time_to_exit, s.visited)
    else
        return HLState(s.x, s.y, mdp.max_time, s.visited)
    end
end

function manhattan_distance(p1::Tuple{Int,Int}, p2::Tuple{Int,Int})
    return abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
end

inbounds(mdp::HLRoverWorldMDP, s::HLState) = 1 ≤ s.x ≤ mdp.grid_size[1] && 1 ≤ s.y ≤ mdp.grid_size[2] && 1 ≤ s.t ≤ mdp.max_time && s.visited in visited_list(mdp)

