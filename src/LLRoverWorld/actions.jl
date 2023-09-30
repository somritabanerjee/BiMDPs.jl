const MOVEMENTS = Dict(UP    => (0,1),
                        DOWN  => (0,-1),
                        LEFT  => (-1,0),
                        RIGHT => (1,0));
const 𝒜 = [UP, DOWN, LEFT, RIGHT]
const INDEX = Dict(UP => 1, DOWN => 2, LEFT => 3, RIGHT => 4)


function POMDPs.actions(mdp::LLRoverWorldMDP)
    return 𝒜
end
function POMDPs.actionindex(mdp::LLRoverWorldMDP, a::LLAction)
    return INDEX[a]
end