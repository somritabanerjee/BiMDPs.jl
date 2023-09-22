function POMDPs.reward(mdp::RoverGridWorldMDP, s::State)
    for (key, val) in mdp.reward_vals
        if s.x == key[1] && s.y == key[2]
            return val
        end
    end
    return 0
end

function POMDPs.reward(mdp::RoverGridWorldMDP, s::State, a::Action, sp::State)
    return POMDPs.reward(mdp, s)
end

reward(mdp::RoverGridWorldMDP, s::State) = POMDPs.reward(mdp, s)

function get_rewards(mdp::RoverGridWorldMDP, policy::Policy)
    null_state = mdp.null_state
    valid_states = setdiff(states(mdp), [null_state])
    U = map(s->reward(mdp, s), valid_states)
end