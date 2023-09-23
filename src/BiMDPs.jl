module BiMDPs

include("RoverGridWorld/RoverGridWorld.jl")
export
    RoverGridWorld

include("RoverXYTWorld/RoverXYTWorld.jl")
export
    RoverXYTWorld

include("RoverWorld/RoverWorld.jl")
export
    RoverWorld

case_dictionary = Dict("case001" => RoverWorld.RoverWorldMDP(
                                            grid_size = (10,10),
                                            max_time = 20,
                                            tgts = Dict(1=>((9,2),(1,18),50),
                                                        2=>((9,8),(15,20),50)),
                                            obstacles = [],
                                            exit_xys = []
                        ),
                        "case002" => RoverWorld.RoverWorldMDP(
                                            grid_size = (10,10),
                                            max_time = 20,
                                            tgts = Dict(1=>((9,2),(1,18),50),
                                                        2=>((9,8),(15,20),50)),
                                            obstacles = [((6,6), (1,20), -5)],
                                            exit_xys = []
                        )
)
export case_dictionary            

end # module
