
function POMDPs.actions(mdp::HLRoverWorldMDP)
    return [[HLAction(i) for i in 1:length(mdp.tgts)]..., [HLAction(-i) for i in 1:length(mdp.exit_xys)]...]
end
function POMDPs.actionindex(mdp::HLRoverWorldMDP, a::HLAction)
    return (a.tgt > 0) ? a.tgt : (length(mdp.tgts) + abs(a.tgt))
end