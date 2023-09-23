function POMDPs.reward(mdp::RoverWorldMDP, s::State)
    r = 0
    for (tgt_id, ((x, y), (t0,tf), val)) in mdp.tgts
        if !s.visited[tgt_id] && (s.x, s.y) == (x, y) && t0 <= s.t <= tf
            r += val
        end
    end
    for ((x, y), (t0,tf), penalty) in mdp.obstacles
        if (s.x, s.y) == (x, y) && t0 <= s.t <= tf
            r += penalty
        end
    end
    return r
end

function POMDPs.reward(mdp::RoverWorldMDP, s::State, a::Action, sp::State)
    verbose = false
    pot_reward = POMDPs.reward(mdp, s)
    if pot_reward != 0.0
        verbose && println("s: $s r: $(pot_reward)")
    end
    return pot_reward
end

reward(mdp::RoverWorldMDP, s::State) = POMDPs.reward(mdp, s)

function get_rewards(mdp::RoverWorldMDP, policy::Policy)
    valid_states = non_null_states(mdp)
    U = map(s->reward(mdp, s), valid_states)
end