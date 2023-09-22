const MOVEMENTS = Dict(UP    => State(0,1),
                        DOWN  => State(0,-1),
                        LEFT  => State(-1,0),
                        RIGHT => State(1,0));
const 𝒜 = [UP, DOWN, LEFT, RIGHT]
const INDEX = Dict(UP => 1, DOWN => 2, LEFT => 3, RIGHT => 4)
const Nₐ = length(𝒜)

function POMDPs.actions(mdp::RoverGridWorldMDP)
    return 𝒜
end
function POMDPs.actionindex(mdp::RoverGridWorldMDP, a::Action)
    return INDEX[a]
end