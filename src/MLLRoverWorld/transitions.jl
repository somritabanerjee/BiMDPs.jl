function POMDPs.transition(mdp::MLLRoverWorldMDP, s::MLLState, a::MLLAction)
    verbose = false
    verbose && println("Current state: $s")
    verbose && println("Current action: $a")
    ((x, y), (t0,tf), val) = mdp.current_tgt
    if ((s.x, s.y) in mdp.exit_xys || s.t == mdp.max_time || ((s.x, s.y) == (x, y) && t0 <= s.t && s.measured)) # transition to terminal state
        verbose && println("Going to terminal state")
        return Deterministic(MLLState(mdp.null_xy[1], mdp.null_xy[2], min(s.t + 1, mdp.max_time), s.measured))
    end

    # If we're doing a measurement near the target, update the measured parameter
    measured = copy(s.measured)
    if a == MEASURE
        if 0.0 < euclidean_distance((s.x, s.y), (x, y)) < 2.0
            verbose && println("Measuring target $tgt_id at $s")
            measured = true
        end
    end
    s_up = MLLState(s.x, s.y, s.t, measured)

    next_states = MLLState[]
    probabilities = Float64[]
    p_transition = mdp.p_transition

    (p_transition != 1.0) && error("Only deterministic transitions are supported")

    destination = s_up + MOVEMENTS[a]
    if inbounds(mdp, destination)
        push!(next_states, destination)
        push!(probabilities, 1.0)
    else
        push!(next_states, MLLState(s_up.x, s_up.y, s_up.t + 1, s_up.measured))
        push!(probabilities, 1.0)
    end

    verbose && println("Next states: $next_states with probabilities $probabilities")
    return SparseCat(next_states, probabilities)
end

function euclidean_distance(p1::Tuple{Int,Int}, p2::Tuple{Int,Int})
    return sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
end

inbounds(mdp::MLLRoverWorldMDP, s::MLLState) = 1 ≤ s.x ≤ mdp.grid_size[1] && 1 ≤ s.y ≤ mdp.grid_size[2] && 1 ≤ s.t ≤ mdp.max_time && s.measured in [true, false]

