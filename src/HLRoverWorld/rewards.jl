function POMDPs.reward(mdp::HLRoverWorldMDP, s::HLState)
    r = 0
    for (tgt_id, ((x, y), (t0,tf), val)) in mdp.tgts
        if !s.visited[tgt_id] && (s.x, s.y) == (x, y) && t0 <= s.t <= tf
            r += val
        end
    end
    return r
end

function POMDPs.reward(mdp::HLRoverWorldMDP, s::HLState, a::HLAction, sp::HLState)
    verbose = false
    pot_reward = POMDPs.reward(mdp, s)
    if pot_reward != 0.0
        verbose && println("s: $s r: $(pot_reward)")
    end
    return pot_reward
end

reward(mdp::HLRoverWorldMDP, s::HLState) = POMDPs.reward(mdp, s)
