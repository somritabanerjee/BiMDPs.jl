Base.:(==)(s1::State, s2::State) = (s1.x == s2.x) && (s1.y == s2.y)
Base.:+(s1::State, s2::State) = State(s1.x + s2.x, s1.y + s2.y)
POMDPs.states(mdp::RoverGridWorldMDP) = [[State(x, y) for x in 1:mdp.grid_size[1], y in 1:mdp.grid_size[2]]..., mdp.null_state]
# POMDPs.initialstate(mdp::RoverGridWorldMDP) = [State(1,1)]
function POMDPs.stateindex(mdp::RoverGridWorldMDP, s::State)
    if s == mdp.null_state
        return length(states(mdp))
    else
        return (s.x - 1) * mdp.grid_size[2] + s.y
    end
end

# Distribution over states
struct GWUniform
    size::Tuple{Int, Int}
end
Base.rand(rng::AbstractRNG, d::GWUniform) = State(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]))
function POMDPs.pdf(d::GWUniform, s::State)
    if s[0]>0 && s[1]>0
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (State(x, y) for x in 1:d.size[1], y in 1:d.size[2])
POMDPs.initialstate(mdp::RoverGridWorldMDP) = GWUniform(mdp.grid_size)