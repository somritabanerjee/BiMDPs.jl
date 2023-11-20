Base.:(==)(s1::PState, s2::PState) = (s1.x == s2.x) && (s1.y == s2.y) && (s1.t == s2.t) && (s1.e == s2.e) && (s1.visited == s2.visited)
Base.:+(s::PState, tup::Tuple{Int,Int}) = PState(s.x + tup[1], s.y + tup[2], s.t + 1, s.e, s.visited)

function POMDPs.states(mdp::PRoverWorldMDP)
    println("called states")
    visited_options = visited_list(mdp)
    states = [[PState(x, y, t, e, visited) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time, e in [false, true], visited in visited_options]..., 
                [PState(mdp.null_xy[1], mdp.null_xy[2], t, e, visited) for t in 1:mdp.max_time, e in [false, true], visited in visited_options]...] # Add null states
    return states
end

# function POMDPs.ordered_states(mdp::PRoverWorldMDP)
#     return POMDPs.states(mdp)
# end

# function POMDPs.initialstate_distribution(mdp::PRoverWorldMDP)
#     println("called initialstate_distribution")
#     return Deterministic(PState(1, 1, 1, true, fill(false, length(mdp.tgts))))
# end

function visited_list(mdp::PRoverWorldMDP)
    num_tgts = length(mdp.tgts)
    return boolean_permutations(num_tgts)
end

function non_null_states(mdp::PRoverWorldMDP)
    visited_options = visited_list(mdp)
    return [PState(x, y, t, visited) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2], t in 1:mdp.max_time, e in [false, true], visited in visited_options]
end



function POMDPs.stateindex(mdp::PRoverWorldMDP, s::PState)
    num_x = mdp.grid_size[1]
    num_y = mdp.grid_size[2]
    num_t = mdp.max_time
    num_e = length([false, true])
    num_visited = 2^length(mdp.tgts)
    if !((s.x, s.y) == mdp.null_xy)
        # println("s: $s")
        return (s.x) + 
                (s.y - 1) * num_x + 
                (s.t - 1) * num_x * num_y + 
                (s.e ? 1 : 0) * num_x * num_y * num_t +
                (findfirst(isequal(s.visited), visited_list(mdp))-1) * num_x * num_y * num_t * num_e
    else
        return num_x * num_y * num_t * num_e * num_visited +
                (s.t) + 
                (s.e ? 1 : 0) * num_t +
                (findfirst(isequal(s.visited), visited_list(mdp))-1) * num_t * num_e
    end
end

# Distribution over states
struct GWUniform
    size::Tuple{Int, Int, Int, Int, Int} # (num_x, num_y, num_t, num_e, visited_tgt_options)
end

# Only sample from possible *start* states
# Base.rand(rng::AbstractRNG, d::GWUniform) = PState(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), 1, true, fill(false,d.size[5]))
# sample from all states
Base.rand(rng::AbstractRNG, d::GWUniform) = PState(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]), rand(rng, 1:d.size[3]), rand(rng, [false, true]), rand(boolean_permutations(d.size[5])))

function POMDPs.pdf(d::GWUniform, s::PState)
    return 1.0
    if s.x>0 && s.y>0
        # println("returning $(1/prod(d.size))")
        return 1/(d.size[1] * d.size[2] * d.size[3] * d.size[4] * (2^d.size[5]))
    else
        println("returning 0")
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (PState(x, y, t, e, v) for x in 1:d.size[1], y in 1:d.size[2], t in 1:d.size[3], e in [false, true], v in boolean_permutations(d.size[5]))
POMDPs.initialstate(mdp::PRoverWorldMDP) = GWUniform((mdp.grid_size[1], mdp.grid_size[2], mdp.max_time, length([true, false]), length(mdp.tgts)))

function boolean_permutations(length::Int)
    return [[((i >> (length - j)) & 1) == 1 for j in 1:length] for i in 0:(2^length - 1)]
end

function rand_state(rng::AbstractRNG, mdp::PRoverWorldMDP)
    return PState(rand(rng, 1:mdp.grid_size[1]), rand(rng, 1:mdp.grid_size[2]), rand(rng, 1:mdp.max_time), rand(rng, [false, true]), rand(visited_list(mdp)))
end

function rand_starting_state(rng::AbstractRNG, mdp::PRoverWorldMDP)
    return PState(rand(rng, 1:mdp.grid_size[1]), rand(rng, 1:mdp.grid_size[2]), 1, true, fill(false, length(mdp.tgts)))
end

function test_state_indexing(mdp::PRoverWorldMDP)
    for (i, s) in enumerate(POMDPs.states(mdp))
        # println("s: $s")
        # println("i: $i")
        # println("stateindex: $(POMDPs.stateindex(mdp, s))")
        @assert i == POMDPs.stateindex(mdp, s)
    end
    println("Num states: $(length(POMDPs.states(mdp)))")
    d = POMDPs.initialstate(mdp)
    println("d: $(d)")
    println("All bools: $(boolean_permutations(d.size[5]))")
    println("Support: ", POMDPs.support(d))
    rstate = rand_state(Random.default_rng(), mdp)
    println("rand_state: ", rstate)
    println("pdf: ", POMDPs.pdf(d, rstate))
    prob = 0.0
    for s in POMDPs.support(d)
        prob += POMDPs.pdf(d, s)
    end
    println("Sum of pdfs: ", prob)
    prob = 0.0
    for (i, s) in enumerate(POMDPs.states(mdp))
        prob += POMDPs.pdf(d, s)
    end
    println("Sum of pdfs: ", prob)
        
end