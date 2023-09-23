function POMDPs.transition(mdp::RoverWorldMDP, s::State, a::Action)
    verbose = false
    verbose && println("Current state: $s")
    verbose && println("Current action: $a")
    if ((s.x, s.y) in mdp.exit_xys || s.t == mdp.max_time) # transition to terminal state
        verbose && println("Going to terminal state")
        return Deterministic(State(mdp.null_xy[1], mdp.null_xy[2], min(s.t + 1, mdp.max_time), s.visited))
    end

    # Check if we're at a target, to update the visited list
    visited = copy(s.visited)
    for (tgt_id, ((x, y), (t0, tf), val)) in mdp.tgts
        if (s.x, s.y) == (x, y) && t0 <= s.t <= tf
            verbose && println("At target $tgt_id at $s")
            visited[tgt_id] = true
        end
    end
    s_up = State(s.x, s.y, s.t, visited)

    next_states = State[]
    probabilities = Float64[]
    p_transition = mdp.p_transition

    (p_transition != 1.0) && error("Only deterministic transitions are supported")

    destination = s_up + MOVEMENTS[a]
    if inbounds(mdp, destination)
        push!(next_states, destination)
        push!(probabilities, 1.0)
    else
        push!(next_states, State(s_up.x, s_up.y, s_up.t + 1, s_up.visited))
        push!(probabilities, 1.0)
    end

    verbose && println("Next states: $next_states with probabilities $probabilities")
    return SparseCat(next_states, probabilities)
end


inbounds(mdp::RoverWorldMDP, s::State) = 1 ≤ s.x ≤ mdp.grid_size[1] && 1 ≤ s.y ≤ mdp.grid_size[2] && 1 ≤ s.t ≤ mdp.max_time && s.visited in visited_list(mdp)

