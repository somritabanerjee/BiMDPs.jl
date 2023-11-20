function POMDPs.transition(mdp::PRoverWorldMDP, s::PState, a::PAction)
    verbose = false
    verbose && println("Current state: $s")
    verbose && println("Current action: $a")
    if ((s.x, s.y) in mdp.exit_xys || s.t == mdp.max_time) # transition to terminal state
        verbose && println("Going to terminal state")
        return Deterministic(PState(mdp.null_xy[1], mdp.null_xy[2], min(s.t + 1, mdp.max_time), s.e, s.visited))
    end

    # Check if we're at a target, to update the visited list
    new_visited = copy(s.visited)
    for (tgt_id, ((x, y), (t0, tf), val)) in mdp.tgts
        if (s.x, s.y) == (x, y) && t0 <= s.t <= tf && s.e # must also have enough energy
            verbose && println("At target $tgt_id at $s")
            new_visited[tgt_id] = true
        end
    end

    new_energy = s.e
    if (a == MEASURE) # If we're doing a measurement, deplete the energy level
        new_energy = false
    elseif !(s.e) # If we're not measuring, other actions restore the energy level
        new_energy = true
    end

    s_up = PState(s.x, s.y, s.t, new_energy, new_visited)

    next_states = PState[]
    probabilities = Float64[]

    destination = s_up + MOVEMENTS[a]
    if inbounds(mdp, destination)
        push!(next_states, destination)
        push!(probabilities, 1.0)
    else
        push!(next_states, PState(s_up.x, s_up.y, s_up.t + 1, s_up.e, s_up.visited))
        push!(probabilities, 1.0)
    end

    verbose && println("Next states: $next_states with probabilities $probabilities")
    sum(probabilities) != 1.0 && println("Warning: probabilities don't sum to 1.0: $(sum(probabilities))")
    return SparseCat(next_states, probabilities)
end


inbounds(mdp::PRoverWorldMDP, s::PState) = 1 ≤ s.x ≤ mdp.grid_size[1] && 1 ≤ s.y ≤ mdp.grid_size[2] && 1 ≤ s.t ≤ mdp.max_time && s.e in [false, true] && s.visited in visited_list(mdp)

