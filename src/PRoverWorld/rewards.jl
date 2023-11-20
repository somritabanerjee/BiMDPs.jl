function POMDPs.reward(mdp::PRoverWorldMDP, s::PState, a::PAction)
    r = 0
    is_tgt = false; is_tgt_rewarded = false; is_obstacle = false;
    for (tgt_id, ((x, y), (t0,tf), val)) in mdp.tgts
        if !s.visited[tgt_id] && (s.x, s.y) == (x, y) && t0 <= s.t <= tf 
            is_tgt = true
            if s.e # must also have enough energy
                r += val
                is_tgt_rewarded = true
            end
        end
    end
    if mdp.obstacles_grid[s.x, s.y, s.t] != 0.0
        r += mdp.obstacles_grid[s.x, s.y, s.t]
        is_obstacle = true
    end
    if a == MEASURE && !is_tgt && !is_obstacle && s.e # need energy to make a measurement
        r += mdp.measure_reward
    end
    return r
end

function POMDPs.reward(mdp::PRoverWorldMDP, s::PState, a::PAction, sp::PState)
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

reward(mdp::PRoverWorldMDP, s::PState) = POMDPs.reward(mdp, s, UP)

function get_rewards(mdp::PRoverWorldMDP, policy::Policy)
    valid_states = non_null_states(mdp)
    U = map(s->reward(mdp, s), valid_states)
end