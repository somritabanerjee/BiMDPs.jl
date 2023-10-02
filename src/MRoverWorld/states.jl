Base.:(==)(s1::MState, s2::MState) = (s1.x == s2.x) && (s1.y == s2.y) && (s1.t == s2.t) && (s1.measured == s2.measured) && (s1.visited == s2.visited)
Base.:+(s::MState, tup::Tuple{Int,Int}) = MState(s.x + tup[1], s.y + tup[2], s.t + 1, s.measured, s.visited)

function POMDPs.states(mdp::MRoverWorldMDP)
    visited_options = permutations_list(mdp)
    measured_options = permutations_list(mdp)
    states = [[MState(x, y, t, measured, visited) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time, measured in measured_options, visited in visited_options]..., 
                [MState(mdp.null_xy[1], mdp.null_xy[2], t, measured, visited) for t in 1:mdp.max_time, measured in measured_options, visited in visited_options]...] # Add null states
    return states
end

function permutations_list(mdp::MRoverWorldMDP)
    num_tgts = length(mdp.tgts)
    return boolean_permutations(num_tgts)
end

function non_null_states(mdp::MRoverWorldMDP)
    visited_options = permutations_list(mdp)
    measured_options = permutations_list(mdp)
    return [MState(x, y, t, measured, visited) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time, measured in measured_options, visited in visited_options]
end



function POMDPs.stateindex(mdp::MRoverWorldMDP, s::MState)
    num_x = mdp.grid_size[1]
    num_y = mdp.grid_size[2]
    num_t = mdp.max_time
    num_permutations = 2^length(mdp.tgts)
    if !((s.x, s.y) == mdp.null_xy)
        return (s.x) + 
                (s.y - 1) * num_x + 
                (s.t - 1) * num_x * num_y + 
                (findfirst(isequal(s.measured), permutations_list(mdp))-1) * num_x * num_y * num_t +
                (findfirst(isequal(s.visited), permutations_list(mdp))-1) * num_x * num_y * num_t * num_permutations
    else
        return num_x * num_y * num_t * num_permutations * num_permutations +
                (s.t) + 
                (findfirst(isequal(s.measured), permutations_list(mdp))-1) * num_t +
                (findfirst(isequal(s.visited), permutations_list(mdp))-1) * num_t * num_permutations
    end
end

# Distribution over states
struct GWUniform
    size::Tuple{Int, Int, Int, Int, Int} # (num_x, num_y, num_t, measured_tgt_options, visited_tgt_options)
end
# Base.rand(rng::AbstractRNG, d::GWUniform) = MState(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), rand(rng, 1:d.size[3]), rand(boolean_permutations(d.size[4])))

# Only sample from possible *start* states
Base.rand(rng::AbstractRNG, d::GWUniform) = MState(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), 1, fill(false,d.size[4]), fill(false,d.size[5]))

function POMDPs.pdf(d::GWUniform, s::MState)
    if inbounds(mdp, s)
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (MState(x, y, t, m, v) for x in 1:d.size[1], y in 1:d.size[2], t in 1:d.size[3], m in boolean_permutations(d.size[4]), v in boolean_permutations(d.size[5]))
POMDPs.initialstate(mdp::MRoverWorldMDP) = GWUniform((mdp.grid_size[1], mdp.grid_size[2], mdp.max_time, length(mdp.tgts), length(mdp.tgts)))

function boolean_permutations(length::Int)
    return [[((i >> (length - j)) & 1) == 1 for j in 1:length] for i in 0:(2^length - 1)]
end

function rand_state(rng::AbstractRNG, mdp::MRoverWorldMDP)
    return MState(rand(rng, 1:mdp.grid_size[1]), rand(rng, 1:mdp.grid_size[2]), rand(rng, 1:mdp.max_time), rand(permutations_list(mdp)), rand(permutations_list(mdp)))
end

function rand_starting_state(rng::AbstractRNG, mdp::MRoverWorldMDP)
    return MState(rand(rng, 1:mdp.grid_size[1]), rand(rng, 1:mdp.grid_size[2]), 1, fill(false, length(mdp.tgts)), fill(false, length(mdp.tgts)))
end

function test_state_indexing(mdp::MRoverWorldMDP)
    for (i, s) in enumerate(POMDPs.states(mdp))
        # println("s: $s")
        # println("i: $i")
        # println("stateindex: $(POMDPs.stateindex(mdp, s))")
        @assert i == POMDPs.stateindex(mdp, s)
    end
end