function POMDPs.transition(mdp::LLRoverWorldMDP, s::LLState, a::LLAction)
    verbose = false
    verbose && println("Current state: $s")
    verbose && println("Current action: $a")
    ((x, y), (t0,tf), val) = mdp.current_tgt
    if ((s.x, s.y) in mdp.exit_xys || s.t == mdp.max_time || ((s.x, s.y) == (x, y) && t0 <= s.t)) # transition to terminal state
        verbose && println("Going to terminal state")
        return Deterministic(LLState(mdp.null_xy[1], mdp.null_xy[2], min(s.t + 1, mdp.max_time)))
    end

    next_states = LLState[]
    probabilities = Float64[]
    p_transition = mdp.p_transition

    (p_transition != 1.0) && error("Only deterministic transitions are supported")

    destination = s + MOVEMENTS[a]
    if inbounds(mdp, destination)
        push!(next_states, destination)
        push!(probabilities, 1.0)
    else
        push!(next_states, LLState(s.x, s.y, s.t + 1))
        push!(probabilities, 1.0)
    end

    verbose && println("Next states: $next_states with probabilities $probabilities")
    return SparseCat(next_states, probabilities)
end


inbounds(mdp::LLRoverWorldMDP, s::LLState) = 1 ≤ s.x ≤ mdp.grid_size[1] && 1 ≤ s.y ≤ mdp.grid_size[2] && 1 ≤ s.t ≤ mdp.max_time

