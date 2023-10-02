module BiMDPs

include("RoverGridWorld/RoverGridWorld.jl")
export RoverGridWorld

include("RoverXYTWorld/RoverXYTWorld.jl")
export RoverXYTWorld

include("RoverWorld/RoverWorld.jl")
export RoverWorld

include("MRoverWorld/MRoverWorld.jl")
export MRoverWorld

include("utils.jl")
export optimality_vs_compute

# case_dictionary = Dict("case001" => RoverWorld.RoverWorldMDP(
#                                             grid_size = (10,10),
#                                             max_time = 20,
#                                             tgts = Dict(1=>((9,2),(1,18),50),
#                                                         2=>((9,8),(15,20),50)),
#                                             obstacles = [],
#                                             exit_xys = []
#                         ),
#                         "case002" => RoverWorld.RoverWorldMDP(
#                                             grid_size = (10,10),
#                                             max_time = 20,
#                                             tgts = Dict(1=>((9,2),(1,18),50),
#                                                         2=>((9,8),(15,20),50)),
#                                             obstacles = [((6,6), (1,20), -5)],
#                                             exit_xys = []
#                         ),
#                         "case003" => RoverWorld.RoverWorldMDP(
#                                             grid_size = (10,10),
#                                             max_time = 20,
#                                             tgts = Dict(1=>((9,2),(1,18),50),
#                                                         2=>((9,8),(15,19),50)),
#                                             obstacles = [((6,6), (1,19), -5),
#                                                         [((x,y), (20,20), -5) for x in 1:10, y in 1:9]...,
#                                                         [((x,y), (20,20), -5) for x in 1:9, y in 10:10]...],
#                                             exit_xys = []
#                         ),
#                         "case004" => RoverWorld.RoverWorldMDP(
#                                             grid_size = (10,10),
#                                             max_time = 20,
#                                             tgts = Dict(1=>((9,2),(1,18),50),
#                                                         2=>((9,8),(15,18),50)),
#                                             obstacles = [((6,6), (1,20), -5),
#                                                         ([[((x, y), (10+i,10+i), -5) for x in 10-i:10, y in 1:i] for i in 1:9]...)...,
#                                                         [((x,y), (20,20), -5) for x in 1:10, y in 1:9]...,
#                                                         [((x,y), (20,20), -5) for x in 1:9, y in 10:10]...],
#                                             exit_xys = []
#                         ),
#                         "case005" => RoverWorld.RoverWorldMDP(
#                                             grid_size = (10,10),
#                                             max_time = 20,
#                                             tgts = Dict(1=>((9,2),(1,11),50),
#                                                         2=>((9,8),(1,17),50),
#                                                         3=>((10,10),(1,20),5)),
#                                             obstacles = [[((x,y), (1,20), -20) for x in 5:6, y in 5:6]...,
#                                                         ([[((x, y), (10+i,10+i), -5) for x in 10-i:10, y in 1:i] for i in 1:9]...)...,
#                                                         [((x,y), (20,20), -5) for x in 1:10, y in 1:9]...,
#                                                         [((x,y), (20,20), -5) for x in 1:9, y in 10:10]...],
#                                             exit_xys = [(10,10)]
#                         ),
#                         "case006" => RoverWorld.RoverWorldMDP(
#                                             grid_size = (10,10),
#                                             max_time = 20,
#                                             tgts = Dict(1=>((9,2),(1,11),50),
#                                                         2=>((9,8),(1,17),50),
#                                                         3=>((10,10),(1,20),5)),
#                                             obstacles = [[((x,y), (1,20), -20) for x in 5:6, y in 5:6]...,
#                                                         ([[((x, y), (10+i,10+i), -5) for x in 10-i:10, y in 1:i] for i in 1:9]...)...,
#                                                         [((x,y), (20,20), -5) for x in 1:10, y in 1:9]...,
#                                                         [((x,y), (20,20), -5) for x in 1:9, y in 10:10]...],
#                                             exit_xys = [(10,10)],
#                                             include_measurement = true,
#                                             measure_reward = 2.0
#                         ),
#                         "case007" => RoverWorld.RoverWorldMDP(
#                                             grid_size = (20,20),
#                                             max_time = 40,
#                                             tgts = Dict(1=>((9,2),(1,30),50),
#                                                         2=>((9,8),(1,39),50),
#                                                         3=>((10,10),(1,40),5)),
#                                             obstacles = [[((x,y), (1,40), -20) for x in 5:6, y in 5:6]...,
#                                                         [((x,y), (1,40), -20) for x in 14:15, y in 17:18]...,
#                                                         ([[((x, y), (10+i,10+i), -5) for x in 20-i:20, y in 1:i] for i in 10:19]...)...,
#                                                         [((x,y), (40,40), -5) for x in 1:20, y in 1:19]...,
#                                                         [((x,y), (40,40), -5) for x in 1:19, y in 20:20]...],
#                                             exit_xys = [(20,20)],
#                                             include_measurement = true,
#                                             measure_reward = 2.0
#                         ),
#                         "case008" => RoverWorld.RoverWorldMDP(
#                                             grid_size = (5,5),
#                                             max_time = 10,
#                                             tgts = Dict(1=>((2,2),(1,10),50),
#                                                         2=>((9,8),(1,39),50),
#                                                         3=>((10,10),(1,40),5)),
#                                             obstacles = [[((x,y), (1,40), -20) for x in 5:6, y in 5:6]...,
#                                                         [((x,y), (1,40), -20) for x in 14:15, y in 17:18]...,
#                                                         ([[((x, y), (10+i,10+i), -5) for x in 20-i:20, y in 1:i] for i in 10:19]...)...,
#                                                         [((x,y), (40,40), -5) for x in 1:20, y in 1:19]...,
#                                                         [((x,y), (40,40), -5) for x in 1:19, y in 20:20]...],
#                                             exit_xys = [(20,20)],
#                                             include_measurement = true,
#                                             measure_reward = 2.0
#                         ),
#                         "case009" => create_rover_world((10,10), 20, 
#                                             tgts=[((9,2), 50.0), ((9,8), 50.0)], 
#                                             shadow=:true,
#                                             shadow_value=-5,
#                                             permanent_obstacles=[((6,6), -10.0)],
#                                             exit_xys = [(10,10)],
#                                             include_measurement = true,
#                                             measure_reward = 2.0
#                         )
# )
case_dictionary = Dict("case009" => create_rover_world((10,10), 
                                            20, 
                                            tgts=[((9,2), 50.0), ((9,8), 50.0)], 
                                            shadow=:true,
                                            shadow_value=-5,
                                            permanent_obstacles=[((6,6), -10.0)],
                                            exit_xys = [(10,10)],
                                            include_measurement = true,
                                            measure_reward = 2.0
                        ),
                        "case010" => create_rover_world((10,10), 
                                            20, 
                                            tgts=[((9,2), 50.0), ((9,8), 50.0), ((10,10), 5.0)], 
                                            shadow=:true,
                                            shadow_value=-5,
                                            permanent_obstacles=[((5,5), -20.0), ((5,6), -20.0), ((6,5), -20.0), ((6,6), -20.0)],
                                            exit_xys = [(10,10)],
                                            include_measurement = false,
                                            measure_reward = 0.0
                        ),
                        "case011" => create_rover_world((10,10), 
                                            25, 
                                            tgts=[((9,2), 50.0), ((9,8), 50.0)], 
                                            shadow=:true,
                                            shadow_value=-5,
                                            permanent_obstacles=[((5,5), -20.0), ((5,6), -20.0), ((6,5), -20.0), ((6,6), -20.0)],
                                            exit_xys = [(10,10)],
                                            force_measurement = true
                        )
);
export case_dictionary  

include("HLRoverWorld/HLRoverWorld.jl")
export HLRoverWorld

include("LLRoverWorld/LLRoverWorld.jl")
export LLRoverWorld

include("MLLRoverWorld/MLLRoverWorld.jl")
export MLLRoverWorld

include("conversions.jl")
export solve_using_bilevel_mdp, solve_using_finegrained_mdp

include("visualizations.jl")

end # module
