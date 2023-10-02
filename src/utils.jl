using POMDPs
using POMDPSimulators
using POMDPPolicies
using Statistics
using DiscreteValueIteration
using TabularTDLearning
using Random

function optimality_vs_compute(mdp::RoverWorld.RoverWorldMDP, solvers_iters_Nsim::Vector{Tuple{String,Vector{Int},Int}}; verbose = false)
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
                            measure_reward = 2.0
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