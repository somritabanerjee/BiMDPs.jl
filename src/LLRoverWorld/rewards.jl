function POMDPs.reward(mdp::LLRoverWorldMDP, s::LLState)
    r = 0
    ((x, y), (t0,tf), val) = mdp.current_tgt
    if (s.x, s.y) == (x, y) && t0 <= s.t <= tf
        r += val
    end
    for ((x, y), (t0,tf), penalty) in mdp.obstacles
        if (s.x, s.y) == (x, y) && t0 <= s.t <= tf
            r += penalty
        end
    end
    return r
end

function POMDPs.reward(mdp::LLRoverWorldMDP, s::LLState, a::LLAction, sp::LLState)
    verbose = false
    pot_reward = POMDPs.reward(mdp, s)
    if pot_reward != 0.0
        verbose && println("s: $s r: $(pot_reward)")
    end
    return pot_reward
end

reward(mdp::LLRoverWorldMDP, s::LLState) = POMDPs.reward(mdp, s)
