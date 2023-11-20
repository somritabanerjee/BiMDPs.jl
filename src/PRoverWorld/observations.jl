# function POMDPs.observation(pomdp::PRoverWorldMDP, a::PAction, s::PState)
#     println("calling observation with state $s")
#     # TODO: add stochasticity
#     o = PObservation(s.x, s.y, s.t, s.e, s.visited)
#     next_obs = [o]
#     probabilities = [1.0]
#     sum(probabilities) != 1.0 && println("Warning: probabilities don't sum to 1.0: $(sum(probabilities))")
#     println("Returning $(next_obs)")
#     return Deterministic(next_obs)
# end

function POMDPs.observation(pomdp::PRoverWorldMDP, a::PAction, s::PState)
    if !inbounds(pomdp, s)
        o = PObservation(s.x, s.y, s.t, s.e, s.visited)
        return Deterministic(o)
    end
    println("calling observation with state $s")
    # TODO: add stochasticity
    o1 = PObservation(s.x, s.y, s.t, s.e, s.visited)
    o2 = PObservation(s.x, s.y, s.t, !(s.e), s.visited)
    next_obs = [o1, o2]
    probabilities = [0.8, 0.2]
    sum(probabilities) != 1.0 && println("Warning: probabilities don't sum to 1.0: $(sum(probabilities))")
    println("Returning $(next_obs)")
    return SparseCat(next_obs, probabilities)
end

function POMDPs.observations(mdp::PRoverWorldMDP)
    println("calling observations")
    return [PObservation(s.x, s.y, s.t, s.e, s.visited) for s in POMDPs.states(mdp)]
end
function POMDPs.obsindex(mdp::PRoverWorldMDP, o::PObservation)
    if o.x <= 0
        error("Invalid obsindex call for o: $o")
    end
    println("calling obsindex")
    eq_state = PState(o.x, o.y, o.t, o.e, o.visited)
    return POMDPs.stateindex(mdp, eq_state)
end

