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
                        ),
                        "case003" => RoverWorld.RoverWorldMDP(
                                            grid_size = (10,10),
                                            max_time = 20,
                                            tgts = Dict(1=>((9,2),(1,18),50),
                                                        2=>((9,8),(15,19),50)),
                                            obstacles = [((6,6), (1,19), -5),
                                                        [((x,y), (20,20), -5) for x in 1:10, y in 1:9]...,
                                                        [((x,y), (20,20), -5) for x in 1:9, y in 10:10]...],
                                            exit_xys = []
                        ),
                        "case004" => RoverWorld.RoverWorldMDP(
                                            grid_size = (10,10),
                                            max_time = 20,
                                            tgts = Dict(1=>((9,2),(1,18),50),
                                                        2=>((9,8),(15,18),50)),
                                            obstacles = [((6,6), (1,20), -5),
                                                        ([[((x, y), (10+i,10+i), -5) for x in 10-i:10, y in 1:i] for i in 1:9]...)...,
                                                        [((x,y), (20,20), -5) for x in 1:10, y in 1:9]...,
                                                        [((x,y), (20,20), -5) for x in 1:9, y in 10:10]...],
                                            exit_xys = []
                        )
)
export case_dictionary            

end # module
