function POMDPs.reward(mdp::RoverWorldMDP, s::State, a::Action)
    r = 0
    is_tgt = false; is_obstacle = false;
    for (tgt_id, ((x, y), (t0,tf), val)) in mdp.tgts
        if !s.visited[tgt_id] && (s.x, s.y) == (x, y) && t0 <= s.t <= tf
            r += val
            is_tgt = true
        end
    end
    for ((x, y), (t0,tf), penalty) in mdp.obstacles
        if (s.x, s.y) == (x, y) && t0 <= s.t <= tf
            r += penalty
            is_obstacle = true
        end
    end
    if a == MEASURE && !is_tgt && !is_obstacle
        r += mdp.measure_reward
    end
    return r
end

function POMDPs.reward(mdp::RoverWorldMDP, s::State, a::Action, sp::State)
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

reward(mdp::RoverWorldMDP, s::State) = POMDPs.reward(mdp, s, UP)

function get_rewards(mdp::RoverWorldMDP, policy::Policy)
    valid_states = non_null_states(mdp)
    U = map(s->reward(mdp, s), valid_states)
end