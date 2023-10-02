function POMDPs.reward(mdp::LLRoverWorldMDP, s::LLState, a::LLAction)
    r = 0
    is_tgt = false; is_obstacle = false;
    ((x, y), (t0,tf), val) = mdp.current_tgt
    if (s.x, s.y) == (x, y) && t0 <= s.t <= tf
        r += val
        is_tgt = true
    end
    if mdp.obstacles_grid[s.x, s.y, s.t] != 0.0
        r += mdp.obstacles_grid[s.x, s.y, s.t]
        is_obstacle = true
    end
    if a == MEASURE && !is_tgt && !is_obstacle
        r += mdp.measure_reward
    end
    return r
end

function POMDPs.reward(mdp::LLRoverWorldMDP, s::LLState, a::LLAction, sp::LLState)
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

reward(mdp::LLRoverWorldMDP, s::LLState) = POMDPs.reward(mdp, s, UP)
