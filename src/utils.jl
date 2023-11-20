using POMDPs
using POMDPSimulators
using POMDPPolicies
using Statistics
using DiscreteValueIteration
using TabularTDLearning
using Random
using Dates
using JLD2

function optimality_vs_compute(mdp::Union{RoverWorld.RoverWorldMDP, MRoverWorld.MRoverWorldMDP}, solvers_iters_Nsim::Vector{Tuple{String,Vector{Int},Int}}; verbose = false, dir="temp")
    results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}() # dictionary mapping solver_name to (comp_time, mean_reward, stddev_reward)
    for (solver_name, max_iters_vec, N_sim) in solvers_iters_Nsim
        comp_times = Vector{Float64}(undef, length(max_iters_vec))
        mean_rewards = Vector{Float64}(undef, length(max_iters_vec))
        stddev_rewards = Vector{Float64}(undef, length(max_iters_vec))
        for (i, max_iters) in enumerate(max_iters_vec)
            if solver_name == "bl_vi"
                sim_comp_times = zeros(N_sim)
                rewards = zeros(N_sim)
                for j in 1:N_sim
                    comp_time, discounted_reward, sar_history = solve_using_bilevel_mdp(mdp, max_iters=max_iters)
                    sim_comp_times[j] = comp_time
                    rewards[j] = discounted_reward
                end
                comp_times[i] = mean(sim_comp_times)
                stats_sim = mean_std(rewards)
                println("Reward of $solver_name after $N_sim simulations: μ = $(stats_sim.μ), σ = $(stats_sim.σ)")
                mean_rewards[i] = stats_sim.μ
                stddev_rewards[i] = stats_sim.σ
            else
                if solver_name == "vi"
                    solver = ValueIterationSolver(max_iterations=max_iters)
                elseif solver_name == "qlearning"
                    solver = QLearningSolver(n_episodes=max_iters,
                                            learning_rate=0.8,
                                            exploration_policy=EpsGreedyPolicy(mdp, 0.5),
                                            verbose=false)
                elseif solver_name == "sarsa"
                    solver = SARSASolver(n_episodes=max_iters,
                                            learning_rate=0.8,
                                            exploration_policy=EpsGreedyPolicy(mdp, 0.5),
                                            verbose=false)
                else
                    error("Unknown solver: $solver_name")
                end
                policy, comp_time = @timed solve(solver, mdp)
                verbose && println("Comp_time of $solver_name after $max_iters iterations: ", comp_time)
                comp_times[i] = comp_time

                rollsim = RolloutSimulator(max_steps=mdp.max_time)
                stats_sim = mean_std([simulate(rollsim, mdp, policy) for _ in 1:N_sim])
                println("Reward of $solver_name after $N_sim simulations: μ = $(stats_sim.μ), σ = $(stats_sim.σ)")
                mean_rewards[i] = stats_sim.μ
                stddev_rewards[i] = stats_sim.σ
            end
        end
        results[solver_name] = (comp_times, mean_rewards, stddev_rewards)
    end
    fname = dir*"/"*"data-"*Dates.format(now(),"yyyy-mm-dd_HH_MM")*".jld2"
    save(fname, "data", results)
    return results
end

mean_std(X) = (μ=mean(X), σ=std(X), r=X)

function test_LL()
    ll_mdp = LLRoverWorld.LLRoverWorldMDP(grid_size = (20,20),
                    max_time = 10,
                    null_xy = (-1,-1),
                    p_transition = 1.0,
                    γ = 0.95,
                    current_tgt = ((5,1),(1,10),50),
                    obstacles_grid = zeros(Float64, (20,20,10)),
                    exit_xys = [],
                    init_state = LLRoverWorld.LLState(2, 6, 1)
                    )
    ll_solver = ValueIterationSolver(max_iterations=200)
    LLRoverWorld.test_state_indexing(ll_mdp)
    println("passed")
    LLRoverWorld.print_details(ll_mdp)
    ll_policy, ll_comp_time = @timed solve(ll_solver, ll_mdp)
    for (ll_s, ll_a, ll_r) in stepthrough(ll_mdp, ll_policy, ll_mdp.init_state, "s,a,r", max_steps=ll_mdp.max_time)
        println("LL: in state $ll_s, taking action $ll_a, received reward $ll_r")
    end
end

function create_rover_world(grid_size::Tuple{Int64,Int64}, 
                            max_time::Int64; 
                            tgts::Vector{Tuple{Tuple{Int64,Int64}, Float64}} = [((2,2), 50.0), ((9,8), 50.0)], 
                            shadow = :true, 
                            shadow_value = -5,
                            permanent_obstacles::Vector{Tuple{Tuple{Int64,Int64}, Float64}} = [((10,10), -20.0)],                            
                            exit_xys=[(num_rows, num_cols)],
                            include_measurement = true,
                            measure_reward = 2.0,
                            force_measurement = false,
                            pomdp = false
                            )
    if shadow == :true
        obstacles_grid = create_grid_obstacles_shadow(grid_size[1], grid_size[2], max_time; exit_xys=exit_xys)
    else
        obstacles_grid = fill(false,(grid_size[1],grid_size[2],max_time))
    end
    obstacles_grid = obstacles_grid .* shadow_value
    for ((x,y),v) in permanent_obstacles
        obstacles_grid[x, y, :] .= v
    end
    tgts_dict = Dict()
    for (i, ((x,y),v)) in enumerate(tgts)
        shadow_time = findfirst(val!=0 for val in obstacles_grid[x,y,:])
        if shadow_time == nothing
            shadow_time = max_time+1
        end
        tgts_dict[i] = ((x,y),(1,shadow_time-1),v)
    end
    if force_measurement
        mrgw = MRoverWorld.MRoverWorldMDP(
            grid_size = grid_size,
            max_time = max_time,
            tgts = tgts_dict,
            obstacles_grid = obstacles_grid,
            exit_xys = exit_xys,
            measure_reward = measure_reward
        )
        MRoverWorld.test_state_indexing(mrgw)
        return mrgw
    else
        if pomdp
            rgw = PRoverWorld.PRoverWorldMDP(
                grid_size = grid_size,
                max_time = max_time,
                tgts = tgts_dict,
                obstacles_grid = obstacles_grid,
                exit_xys = exit_xys,
                include_measurement = include_measurement,
                measure_reward = measure_reward
            )
            PRoverWorld.test_state_indexing(rgw)
            return rgw
        else
            rgw = RoverWorld.RoverWorldMDP(
                grid_size = grid_size,
                max_time = max_time,
                tgts = tgts_dict,
                obstacles_grid = obstacles_grid,
                exit_xys = exit_xys,
                include_measurement = include_measurement,
                measure_reward = measure_reward
            )
            RoverWorld.test_state_indexing(rgw)
            return rgw
        end
    end
end

function create_grid_obstacles_shadow(num_rows, num_cols, max_time; start_from::Symbol=:bottom_right, exit_xys=[(num_rows, num_cols)])
    num_rows != num_cols && error("Not implemented. Currently, num_rows must equal num_cols")
    obstacles_grid = fill(false,(num_rows,num_cols,max_time))
    shadow_end_time = max_time
    # Shadow progresses one row/col at a time
    shadow_start_time = max_time - num_rows + 1
    for i in 1:num_rows
        if start_from == :bottom_right
            x_range = num_rows-i+1 : num_rows
            y_range = 1:i
        else
            error("Not implemented for direction $start_from")
        end
        cur_time = shadow_start_time + i - 1
        t_range = cur_time:shadow_end_time
        for (x,y,t) in Iterators.product(x_range, y_range,t_range)
            obstacles_grid[x, y, t] = true
        end
    end
    # Exception for exit xys
    for (x,y) in exit_xys
        obstacles_grid[x, y, :] .= false
    end
    
    return obstacles_grid
end

function manhattan_distance(p1::Tuple{Int,Int}, p2::Tuple{Int,Int})
    return abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
end

function euclidean_distance(p1::Tuple{Int,Int}, p2::Tuple{Int,Int})
    return sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
end

# test with vary_grid_size(verbose=true, N_sim=1, dir=dir)
function vary_grid_size(;verbose = false, N_sim = 10, dir="temp")
    # grid_sizes = [(10,10)]
    # grid_sizes = [(10,10), (11,11), (13,13)]
    grid_sizes = [(10,10), (20,20), (30,30), (40,40), (50,50)]
    blvi_comp_times = zeros(length(grid_sizes))
    blvi_mean_rewards = zeros(length(grid_sizes))
    blvi_std_rewards = zeros(length(grid_sizes))
    vi_comp_times = zeros(length(grid_sizes))
    vi_mean_rewards = zeros(length(grid_sizes))
    vi_std_rewards = zeros(length(grid_sizes))
    max_iterations_blvi = 5000000
    max_iterations_vi   = 50000
    for (i,gs) in enumerate(grid_sizes)
        horizon = gs[1] + gs[2]
        rgw = create_rover_world(gs, 
                            horizon, 
                            tgts=[((9,2), 50.0), ((9,8), 50.0), ((10,10), 5.0)], 
                            shadow=:true,
                            shadow_value=-5,
                            permanent_obstacles=[((5,5), -20.0), ((5,6), -20.0), ((6,5), -20.0), ((6,6), -20.0)],
                            exit_xys = [(10,10)],
                            include_measurement = false,
                            measure_reward = 0.0
                            )
        if N_sim == 1
            # Init state
            init_states = [RoverWorld.State(1, 1, 1, fill(false, length(rgw.tgts)))]
        else
            rng = Random.default_rng()
            init_states = [RoverWorld.rand_starting_state(rng, rgw) for _ in 1:N_sim]
        end
        # Create the HL MDP once to save time
        hl_mdp = HighLevelMDP(rgw)
        hl_solver = ValueIterationSolver(max_iterations=max_iterations_blvi)
        hl_policy, hl_comp_time = @timed solve(hl_solver, hl_mdp)
        verbose && println("Done solving HL MDP in $hl_comp_time seconds")
        # Create VI solver once to save time
        vi_solver = ValueIterationSolver(max_iterations=max_iterations_vi)
        vi_policy, vi_comp_time = @timed solve(vi_solver, rgw)
        verbose && println("Done solving VI MDP in $vi_comp_time seconds")
        rollsim = RolloutSimulator(max_steps=rgw.max_time)
        bct = zeros(N_sim)
        blvi_rewards = zeros(N_sim)
        vi_rewards = zeros(N_sim)
        for n in 1:N_sim
            s0 = init_states[n]
            # Solve + Simulate using BLVI
            blvi_comp_time, blvi_discounted_reward, blvi_sar_history = solve_using_bilevel_mdp(rgw, max_iters=max_iterations_blvi, verbose=false, init_state=s0, hl_mdp=hl_mdp, hl_policy=hl_policy, hl_comp_time=hl_comp_time)
            verbose && println("Done solving BLVI MDP in $blvi_comp_time seconds")
            bct[n] = blvi_comp_time
            blvi_rewards[n] = blvi_discounted_reward
            # Simulate using VI
            vi_reward = simulate(rollsim, rgw, vi_policy, s0)
            vi_rewards[n] = vi_reward
            verbose && println("BLVI reward: $blvi_discounted_reward, VI reward: $vi_reward")
        end

        # Save results
        blvi_comp_times[i] = mean(bct)
        vi_comp_times[i] = vi_comp_time
        blvi_mean_rewards[i] = mean(blvi_rewards)
        vi_mean_rewards[i] = mean(vi_rewards)
        blvi_std_rewards[i] = std(blvi_rewards)
        vi_std_rewards[i] = std(vi_rewards)
    end
    results = Dict("bl_vi" => (blvi_comp_times, blvi_mean_rewards, blvi_std_rewards), "vi" => (vi_comp_times, vi_mean_rewards, vi_std_rewards))
    fname = dir*"/"*"varygridsize-data-"*Dates.format(now(),"yyyy-mm-dd_HH_MM")*".jld2"
    save(fname, "data", results)
    return results
end