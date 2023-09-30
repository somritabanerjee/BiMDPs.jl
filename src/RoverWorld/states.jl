Base.:(==)(s1::State, s2::State) = (s1.x == s2.x) && (s1.y == s2.y) && (s1.t == s2.t) && (s1.visited == s2.visited)
Base.:+(s::State, tup::Tuple{Int,Int}) = State(s.x + tup[1], s.y + tup[2], s.t + 1, s.visited)

function POMDPs.states(mdp::RoverWorldMDP)
    visited_options = visited_list(mdp)
    states = [[State(x, y, t, visited) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time, visited in visited_options]..., 
                [State(mdp.null_xy[1], mdp.null_xy[2], t, visited) for t in 1:mdp.max_time, visited in visited_options]...] # Add null states
    return states
end

function visited_list(mdp::RoverWorldMDP)
    num_tgts = length(mdp.tgts)
    return boolean_permutations(num_tgts)
end

function non_null_states(mdp::RoverWorldMDP)
    visited_options = visited_list(mdp)
    return [State(x, y, t, visited) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time, visited in visited_options]
end



function POMDPs.stateindex(mdp::RoverWorldMDP, s::State)
    num_x = mdp.grid_size[1]
    num_y = mdp.grid_size[2]
    num_t = mdp.max_time
    num_visited = 2^length(mdp.tgts)
    if !((s.x, s.y) == mdp.null_xy)
        return (s.x) + 
                (s.y - 1) * num_x + 
                (s.t - 1) * num_x * num_y + 
                (findfirst(isequal(s.visited), visited_list(mdp))-1) * num_x * num_y * num_t
    else
        return num_x * num_y * num_t * num_visited +
                (s.t) + 
                (findfirst(isequal(s.visited), visited_list(mdp))-1) * num_t
    end
end

# Distribution over states
struct GWUniform
    size::Tuple{Int, Int, Int, Int} # (num_x, num_y, num_t, visited_tgt_options)
end
# Base.rand(rng::AbstractRNG, d::GWUniform) = State(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), rand(rng, 1:d.size[3]), rand(boolean_permutations(d.size[4])))

# Only sample from possible *start* states
Base.rand(rng::AbstractRNG, d::GWUniform) = State(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), 1, fill(false,d.size[4]))

function POMDPs.pdf(d::GWUniform, s::State)
    if inbounds(mdp, s)
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (State(x, y, t, v) for x in 1:d.size[1], y in 1:d.size[2], t in 1:d.size[3], v in boolean_permutations(d.size[4]))
POMDPs.initialstate(mdp::RoverWorldMDP) = GWUniform((mdp.grid_size[1], mdp.grid_size[2], mdp.max_time, length(mdp.tgts)))

function boolean_permutations(length::Int)
    return [[((i >> (length - j)) & 1) == 1 for j in 1:length] for i in 0:(2^length - 1)]
end

function rand_state(rng::AbstractRNG, mdp::RoverWorldMDP)
    return State(rand(rng, 1:mdp.grid_size[1]), rand(rng, 1:mdp.grid_size[2]), rand(rng, 1:mdp.max_time), rand(visited_list(mdp)))
end

function rand_starting_state(rng::AbstractRNG, mdp::RoverWorldMDP)
    return State(rand(rng, 1:mdp.grid_size[1]), rand(rng, 1:mdp.grid_size[2]), 1, fill(false, length(mdp.tgts)))
end

function test_state_indexing(mdp::RoverWorldMDP)
    for (i, s) in enumerate(POMDPs.states(mdp))
        # println("s: $s")
        # println("i: $i")
        # println("stateindex: $(POMDPs.stateindex(mdp, s))")
        @assert i == POMDPs.stateindex(mdp, s)
    end
end