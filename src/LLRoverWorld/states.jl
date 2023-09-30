Base.:(==)(s1::LLState, s2::LLState) = (s1.x == s2.x) && (s1.y == s2.y) && (s1.t == s2.t)
Base.:+(s::LLState, tup::Tuple{Int,Int}) = LLState(s.x + tup[1], s.y + tup[2], s.t + 1)

function POMDPs.states(mdp::LLRoverWorldMDP)
    states = [[LLState(x, y, t) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time]..., 
                [LLState(mdp.null_xy[1], mdp.null_xy[2], t) for t in 1:mdp.max_time]...] # Add null states
    return states
end

function non_null_states(mdp::LLRoverWorldMDP)
    return [LLState(x, y, t) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time]
end



function POMDPs.stateindex(mdp::LLRoverWorldMDP, s::LLState)
    num_x = mdp.grid_size[1]
    num_y = mdp.grid_size[2]
    num_t = mdp.max_time
    if !((s.x, s.y) == mdp.null_xy)
        return (s.x) + 
                (s.y - 1) * num_x + 
                (s.t - 1) * num_x * num_y 
    else
        return num_x * num_y * num_t +
                (s.t)
    end
end

# Distribution over states
struct GWUniform
    size::Tuple{Int, Int, Int} # (num_x, num_y, num_t, visited_tgt_options)
end
# Base.rand(rng::AbstractRNG, d::GWUniform) = LLState(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), rand(rng, 1:d.size[3]), rand(boolean_permutations(d.size[4])))

# Only sample from possible *start* states
Base.rand(rng::AbstractRNG, d::GWUniform) = LLState(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), 1)

function POMDPs.pdf(d::GWUniform, s::LLState)
    if inbounds(mdp, s)
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (LLState(x, y, t) for x in 1:d.size[1], y in 1:d.size[2], t in 1:d.size[3])
POMDPs.initialstate(mdp::LLRoverWorldMDP) = GWUniform((mdp.grid_size[1], mdp.grid_size[2], mdp.max_time))

function rand_state(rng::AbstractRNG, mdp::LLRoverWorldMDP)
    return LLState(rand(rng, 1:mdp.grid_size[1]), rand(rng, 1:mdp.grid_size[2]), rand(rng, 1:mdp.max_time))
end

function test_state_indexing(mdp::LLRoverWorldMDP)
    for (i, s) in enumerate(POMDPs.states(mdp))
        # println("s: $s")
        # println("i: $i")
        # println("stateindex: $(POMDPs.stateindex(mdp, s))")
        @assert i == POMDPs.stateindex(mdp, s)
    end
end