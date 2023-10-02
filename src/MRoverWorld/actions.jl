const MOVEMENTS = Dict(UP    => (0,1),
                        DOWN  => (0,-1),
                        LEFT  => (-1,0),
                        RIGHT => (1,0),
                        MEASURE => (0,0));
const 𝒜 = [UP, DOWN, LEFT, RIGHT, MEASURE]
const INDEX = Dict(UP => 1, DOWN => 2, LEFT => 3, RIGHT => 4, MEASURE => 5)

function POMDPs.actions(mdp::MRoverWorldMDP)
    return 𝒜
end
function POMDPs.actionindex(mdp::MRoverWorldMDP, a::MAction)
    return INDEX[a]
end