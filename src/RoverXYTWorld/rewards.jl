function POMDPs.reward(mdp::RoverXYTWorldMDP, s::State)
    for (tgt_id, ((x, y), val)) in mdp.reward_vals
        if !s.visited[tgt_id] && s.x == x && s.y == y
            return val
        end
    end
    return 0
end

function POMDPs.reward(mdp::RoverXYTWorldMDP, s::State, a::Action, sp::State)
    verbose = false
    pot_reward = POMDPs.reward(mdp, s)
    if pot_reward > 0.0
        verbose && println("s: $s r: $(pot_reward)")
    end
    return pot_reward
end

reward(mdp::RoverXYTWorldMDP, s::State) = POMDPs.reward(mdp, s)

function get_rewards(mdp::RoverXYTWorldMDP, policy::Policy)
    valid_states = non_null_states(mdp)
    U = map(s->reward(mdp, s), valid_states)
end