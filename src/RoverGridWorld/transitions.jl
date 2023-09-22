function POMDPs.transition(mdp::RoverGridWorldMDP, s::State, a::Action)
    if (s == mdp.terminal_state) || (mdp.terminal_state == mdp.null_state && reward(mdp, s) != 0)
        return Deterministic(mdp.null_state)
    end

    next_states = Vector{State}(undef, Nₐ + 1)
    probabilities = zeros(Nₐ + 1)
    p_transition = mdp.p_transition

    if p_transition == 1.0
        # deterministic transitions
        destination = s + MOVEMENTS[a]
        next_states[2] = destination
        if inbounds(mdp, destination)
            probabilities[2] = 1.0
        end
    else
        # stochastic transitions
        for (i, a′) in enumerate(𝒜)
            prob = (a′ == a) ? p_transition : (1 - p_transition) / (Nₐ - 1)
            destination = s + MOVEMENTS[a′]
            next_states[i+1] = destination
            if inbounds(mdp, destination)
                probabilities[i+1] += prob
            end
        end
    end
    # handle out-of-bounds transitions
    next_states[1] = s
    probabilities[1] = 1 - sum(probabilities)
    return SparseCat(next_states, probabilities)
end


inbounds(mdp::RoverGridWorldMDP, s::State) = 1 ≤ s.x ≤ mdp.grid_size[1] && 1 ≤ s.y ≤ mdp.grid_size[2]

