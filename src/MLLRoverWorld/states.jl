Base.:(==)(s1::MLLState, s2::MLLState) = (s1.x == s2.x) && (s1.y == s2.y) && (s1.t == s2.t) && (s1.measured == s2.measured)
Base.:+(s::MLLState, tup::Tuple{Int,Int}) = MLLState(s.x + tup[1], s.y + tup[2], s.t + 1, s.measured)

function POMDPs.states(mdp::MLLRoverWorldMDP)
    states = [[MLLState(x, y, t, m) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time, m in [false, true]]..., 
                [MLLState(mdp.null_xy[1], mdp.null_xy[2], t, m) for t in 1:mdp.max_time, m in [false, true]]...] # Add null states
    return states
end

function non_null_states(mdp::MLLRoverWorldMDP)
    return [MLLState(x, y, t, m) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time, m in [false, true]]
end



function POMDPs.stateindex(mdp::MLLRoverWorldMDP, s::MLLState)
    num_x = mdp.grid_size[1]
    num_y = mdp.grid_size[2]
    num_t = mdp.max_time
    num_m = 2
    if !((s.x, s.y) == mdp.null_xy)
        return (s.x) + 
                (s.y - 1) * num_x + 
                (s.t - 1) * num_x * num_y +
                (s.measured ? 1 : 0) * num_x * num_y * num_t
    else
        return num_x * num_y * num_t * 2 +
                (s.t) +
                (s.measured ? 1 : 0) * num_t
    end
end

# Distribution over states
struct GWUniform
    size::Tuple{Int, Int, Int, Int} # (num_x, num_y, num_t, visited_tgt_options)
end
# Base.rand(rng::AbstractRNG, d::GWUniform) = MLLState(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), rand(rng, 1:d.size[3]), rand(boolean_permutations(d.size[4])))

# Only sample from possible *start* states
Base.rand(rng::AbstractRNG, d::GWUniform) = MLLState(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), 1, rand(Bool))

function POMDPs.pdf(d::GWUniform, s::MLLState)
    if inbounds(mdp, s)
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (MLLState(x, y, t, m) for x in 1:d.size[1], y in 1:d.size[2], t in 1:d.size[3], m in [false, true])
POMDPs.initialstate(mdp::MLLRoverWorldMDP) = GWUniform((mdp.grid_size[1], mdp.grid_size[2], mdp.max_time, 2))

function rand_state(rng::AbstractRNG, mdp::MLLRoverWorldMDP)
    return MLLState(rand(rng, 1:mdp.grid_size[1]), rand(rng, 1:mdp.grid_size[2]), rand(rng, 1:mdp.max_time), rand(Bool))
end

function test_state_indexing(mdp::MLLRoverWorldMDP)
    for (i, s) in enumerate(POMDPs.states(mdp))
        # println("s: $s")
        # println("i: $i")
        # println("stateindex: $(POMDPs.stateindex(mdp, s))")
        @assert i == POMDPs.stateindex(mdp, s)
    end
end