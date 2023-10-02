function POMDPs.transition(mdp::MRoverWorldMDP, s::MState, a::MAction)
    verbose = false
    verbose && println("Current state: $s")
    verbose && println("Current action: $a")
    if ((s.x, s.y) in mdp.exit_xys || s.t == mdp.max_time) # transition to terminal state
        verbose && println("Going to terminal state")
        return Deterministic(MState(mdp.null_xy[1], mdp.null_xy[2], min(s.t + 1, mdp.max_time), s.measured, s.visited))
    end

    # If we're doing a measurement near a target, update the measured list
    measured = copy(s.measured)
    if a == MEASURE
        for (tgt_id, ((x, y), (t0, tf), val)) in mdp.tgts
            if 0.0 < euclidean_distance((s.x, s.y), (x, y)) < 2.0
                verbose && println("Measuring target $tgt_id at $s")
                measured[tgt_id] = true
            end
        end
    end

    # Check if we're at a target, to update the visited list
    visited = copy(s.visited)
    for (tgt_id, ((x, y), (t0, tf), val)) in mdp.tgts
        # Must have measured target before visiting it
        if (s.x, s.y) == (x, y) && t0 <= s.t <= tf && measured[tgt_id]
            verbose && println("At target $tgt_id at $s")
            visited[tgt_id] = true
        end
    end
    s_up = MState(s.x, s.y, s.t, measured, visited)

    next_states = MState[]
    probabilities = Float64[]
    p_transition = mdp.p_transition

    (p_transition != 1.0) && error("Only deterministic transitions are supported")

    destination = s_up + MOVEMENTS[a]
    if inbounds(mdp, destination)
        push!(next_states, destination)
        push!(probabilities, 1.0)
    else
        push!(next_states, MState(s_up.x, s_up.y, s_up.t + 1, s_up.measured, s_up.visited))
        push!(probabilities, 1.0)
    end

    verbose && println("Next states: $next_states with probabilities $probabilities")
    return SparseCat(next_states, probabilities)
end
function manhattan_distance(p1::Tuple{Int,Int}, p2::Tuple{Int,Int})
    return abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
end

function euclidean_distance(p1::Tuple{Int,Int}, p2::Tuple{Int,Int})
    return sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
end

inbounds(mdp::MRoverWorldMDP, s::MState) = 1 ≤ s.x ≤ mdp.grid_size[1] && 1 ≤ s.y ≤ mdp.grid_size[2] && 1 ≤ s.t ≤ mdp.max_time && s.measured in permutations_list(mdp) && s.visited in permutations_list(mdp)

