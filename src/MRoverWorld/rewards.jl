function POMDPs.reward(mdp::MRoverWorldMDP, s::MState, a::MAction)
    r = 0
    is_tgt = false; is_obstacle = false;
    for (tgt_id, ((x, y), (t0,tf), val)) in mdp.tgts
        # Must measure target before visiting it
        if !s.visited[tgt_id] && s.measured[tgt_id] && (s.x, s.y) == (x, y) && t0 <= s.t <= tf
            r += val
            is_tgt = true
        end
    end
    if mdp.obstacles_grid[s.x, s.y, s.t] != 0.0
        r += mdp.obstacles_grid[s.x, s.y, s.t]
        is_obstacle = true
    end
    if a == MEASURE && !is_tgt && !is_obstacle
        for (tgt_id, ((x, y), (t0,tf), val)) in mdp.tgts
            if 0.0 < euclidean_distance((s.x, s.y), (x, y)) < 2.0
                r += mdp.measure_reward
            end
        end
    end
    return r
end

function POMDPs.reward(mdp::MRoverWorldMDP, s::MState, a::MAction, sp::MState)
    verbose = false
    pot_reward = POMDPs.reward(mdp, s, a)
    if pot_reward != 0.0
        if a == MEASURE
            verbose && println("s: $s a: $a r: $(pot_reward)")
        else
            verbose && println("s: $s r: $(pot_reward)")
        end
    end
    return pot_reward
end

reward(mdp::MRoverWorldMDP, s::MState) = POMDPs.reward(mdp, s, UP)

function get_rewards(mdp::MRoverWorldMDP, policy::Policy)
    valid_states = non_null_states(mdp)
    U = map(s->reward(mdp, s), valid_states)
end