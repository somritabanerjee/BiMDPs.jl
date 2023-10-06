using Plots
# using Images
using ColorSchemes, Colors
using Reel


# plot the U values (maximum Q over the actions)
# x = row, y = column, z = U-value
function plot_grid_world(mdp::Union{RoverWorld.RoverWorldMDP, MRoverWorld.MRoverWorldMDP},
    policy::Policy=NothingPolicy(),
    iter=0,
    discount=NaN;
    outline=true,
    show_policy=true,
    show_rewards=false,
    outline_state::Union{RoverWorld.State, MRoverWorld.MState, Nothing}=nothing,
    timestep=nothing,
    cum_reward=nothing,
    fig_title="Grid World Policy Plot",
    augment_title=true)

    gr()

    if mdp isa RoverWorld.RoverWorldMDP
        NothingPolicy = RoverWorld.NothingPolicy
        get_rewards = RoverWorld.get_rewards
        one_based_policy! = RoverWorld.one_based_policy!
        visited_list = RoverWorld.visited_list
        State = RoverWorld.State
        reward = RoverWorld.reward
    elseif mdp isa MRoverWorld.MRoverWorldMDP
        NothingPolicy = RoverWorld.NothingPolicy
        get_rewards = MRoverWorld.get_rewards
        one_based_policy! = RoverWorld.one_based_policy!
        visited_list = MRoverWorld.permutations_list
        State = MRoverWorld.MState
        reward = MRoverWorld.reward
    end        

    if policy isa NothingPolicy
        # override when the policy is empty
        show_policy = false
    end

    if iter == 0
        # solver has not been run yet, so we just plot the raw rewards
        # overwrite policy at time=0 to be emp
        U = get_rewards(mdp, policy)
        one_based_policy!(policy); # handles the case when iterations = 0
    else
        # otherwise, use the Value Function to get the values (i.e., utility)
        U = values(mdp, policy)
    end

    # reshape to grid
    (xmax, ymax) = mdp.grid_size
    if mdp isa RoverWorld.RoverWorldMDP
        Uxy = reshape(U, xmax, ymax, mdp.max_time, 2^length(mdp.tgts))
    elseif mdp isa MRoverWorld.MRoverWorldMDP
        Uxy = reshape(U, xmax, ymax, mdp.max_time, 2^length(mdp.tgts), 2^length(mdp.tgts))
    end
    t = isnothing(outline_state) ? (isnothing(timestep) ? 1 : timestep) : outline_state.t
    visited = isnothing(outline_state) ? fill(false, length(mdp.tgts)) : outline_state.visited
    visited_idx = findfirst(isequal(visited), visited_list(mdp))
    if mdp isa RoverWorld.RoverWorldMDP
        Uxy_slice = Uxy[:,:,t,visited_idx]
    elseif mdp isa MRoverWorld.MRoverWorldMDP
        measured = isnothing(outline_state) ? fill(true, length(mdp.tgts)) : outline_state.measured
        measured_idx = findfirst(isequal(measured), visited_list(mdp))
        Uxy_slice = Uxy[:,:,t,measured_idx,visited_idx]
    end

    # plot values (i.e the U matrix)
    max_val = maximum(abs, Uxy)
    fig = heatmap(Uxy_slice',
                legend=:none,
                aspect_ratio=:equal,
                framestyle=:box,
                tickdirection=:out,
                color=cmap.colors,
                clims=(-max_val, max_val),
                grid = false,
                minorgrid = true,
                minorticks = 2,
                minorgridalpha = 1.0)
    xlims!(0.5, xmax+0.5)
    ylims!(0.5, ymax+0.5)
    xticks!(1:xmax)
    yticks!(1:ymax)

    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

    if show_rewards
        if mdp isa RoverWorld.RoverWorldMDP
            state_coll = [State(x, y, t, visited) for x in 1:xmax, y in 1:ymax]
        elseif mdp isa MRoverWorld.MRoverWorldMDP
            state_coll = [State(x, y, t, measured, visited) for x in 1:xmax, y in 1:ymax]
        end
        for s in state_coll 
            r = reward(mdp, s)
            annotate!([(s.x, s.y, (floor(Int,r), :white, :center, 12, "Computer Modern"))])
        end
    end

    for x in 1:xmax, y in 1:ymax
        # display policy on the plot as arrows
        if show_policy
            grid = policy_grid(policy, xmax, ymax; t=t, visited=visited)
            annotate!([(x, y, (grid[x,y], :center, 12, "Computer Modern"))])
        end
        if outline
            rect = rectangle(1, 1, x - 0.5, y - 0.5)
            plot!(rect, fillalpha=0, linecolor=:gray)
        end
    end

    if !isnothing(outline_state)
        color = ((outline_state.x, outline_state.y) in mdp.exit_xys) ? "yellow" : "blue"
        rect = rectangle(1, 1, outline_state.x - 0.5, outline_state.y - 0.5)
        plot!(rect, fillalpha=0, linecolor=color)
    end

    if augment_title
        fig_title = fig_title * 
                        (isnan(discount)            ? "" : " (iter=$iter, γ=$discount)") * 
                        (isnothing(outline_state)   ? "" : " (t=$t)") * 
                        (isnothing(timestep)        ? "" : " (t=$t)") *
                        (isnothing(cum_reward)      ? "" : " (reward=$cum_reward)")
    end

    title!(fig_title)

    return fig
end

cmap = ColorScheme([Colors.RGB(180/255, 0.0, 0.0), Colors.RGB(1, 1, 1), Colors.RGB(0.0, 100/255, 0.0)], "custom", "threetone, red, white, and green")

function create_value_iteration_gif(mdp::RoverWorld.RoverWorldMDP; dir="gifs")
	frames = Frames(MIME("image/png"), fps=2)
	push!(frames, plot_grid_world(mdp, NothingPolicy(), 0, mdp.γ; outline=true))
	last_frame = nothing
    num_iter = 21
	for iter in 0:num_iter
		local_solver = ValueIterationSolver(max_iterations=iter)
		local_policy = solve(local_solver, mdp)
		one_based_policy!(local_policy)
		last_frame = plot_grid_world(mdp, local_policy, iter, mdp.γ; outline=false)
		push!(frames, last_frame)
	end
	[push!(frames, last_frame) for _ in 1:10] # duplicate last frame
	!isdir(dir) && mkdir(dir) # create directory
	write(dir*"/"*"gridworld_vi.gif", frames)
end

function create_discount_gif(; dir="gifs")
	frames_γ = Frames(MIME("image/png"), fps=2)
	last_frame_γ = nothing
	for γ_iter in 0:0.05:1
		local_solver = ValueIterationSolver(max_iterations=30)

		# local MDP to play around with γ
		vi_γ_mdp = RoverWorldMDP(γ = γ_iter)
		local_π = solve(local_solver, vi_γ_mdp)
		one_based_policy!(local_π)
		last_frame_γ = plot_grid_world(vi_γ_mdp, local_π, 30, vi_γ_mdp.γ; outline=false)
		push!(frames_γ, last_frame_γ)
	end
	[push!(frames_γ, last_frame_γ) for _ in 1:10] # duplicate last frame
	!isdir(dir) && mkdir(dir) # create directory
	write(dir*"/"*"gridworld_vi_γ.gif", frames_γ)
end

function create_simulated_episode_gif(mdp, policy, steps; dir="", fname = "gridworld_episode")
	sim_frames = Frames(MIME("image/png"), fps=2)
    frame_i = nothing
	for i in 1:length(steps)
		frame_i = plot_grid_world(mdp, policy;
			outline_state=steps[i].s, 
            outline=false, 
            cum_reward = sum(st.r for st in steps[1:i]), 
            fig_title="Simulated Episode")
		push!(sim_frames, frame_i)
	end
    [push!(sim_frames, frame_i) for _ in 1:4] # duplicate last frame
	!isdir(dir) && mkdir(dir) # create directory
	write(dir*"/"*fname*".gif", sim_frames)
end

function create_reward_field_evolution_gif(mdp; dir="", fname = "reward_evolution")
	sim_frames = Frames(MIME("image/png"), fps=2)
    frame_i = nothing
	for i in 1:1:mdp.max_time
		frame_i = plot_grid_world(mdp, RoverWorld.NothingPolicy();
			timestep = i, outline=false, 
            show_rewards = true, fig_title="Reward Field Evolution")
		push!(sim_frames, frame_i)
	end
    [push!(sim_frames, frame_i) for _ in 1:4] # duplicate last frame
	!isdir(dir) && mkdir(dir) # create directory
	write(dir*"/"*fname*".gif", sim_frames)
end

function create_reward_field_evolution_imgs(mdp; dir="", subdir = "reward_evolution")
    if mdp isa RoverWorld.RoverWorldMDP
        get_rewards = RoverWorld.get_rewards
    elseif mdp isa MRoverWorld.MRoverWorldMDP
        get_rewards = MRoverWorld.get_rewards
    end       
    U = get_rewards(mdp, RoverWorld.NothingPolicy())
    !isdir(dir) && mkdir(dir) # create directory
    !isdir(dir*"/"*subdir) && mkdir(dir*"/"*subdir) # create subdirectory
    start = 1
    for i in 1:1:mdp.max_time
        slice_i = (mdp isa RoverWorld.RoverWorldMDP) ? U[:,:,i,1] : U[:,:,i,1,1]
        slice_ip = (i==mdp.max_time) ? nothing : ((mdp isa RoverWorld.RoverWorldMDP) ? U[:,:,i+1,1] : U[:,:,i+1,1,1])
        if (i<mdp.max_time) && (slice_i == slice_ip)
            continue
        end
        if start == i
            title = "t = $i"
        else
            title = "t = $start to $i"
        end
        println("Creating figure for $title")
        fname = "Reward Field at $title"
        fig = plot_grid_world(mdp, RoverWorld.NothingPolicy();
			timestep = i, outline=false, 
            show_rewards = true, fig_title=fname, augment_title=false)
        savefig(fig, dir*"/"*subdir*"/"*fname)
        savefig(fig, dir*"/"*subdir*"/"*fname*".pdf")
        start = i+1
    end
end

using RollingFunctions
using LaTeXStrings
function plot_simulation_results(results; dir="results", fname="simulation_results")
    N_sim = length(results.value_iteration.r)
    println("N_sim = $N_sim")
    window = floor(Int, min(500, N_sim/3))
    println("window = $window")
    
    rolling_mean_vi = rolling(mean, results.value_iteration.r, window)
    rolling_mean_ql = rolling(mean, results.q_learning.r, window)
    rolling_mean_sarsa = rolling(mean, results.sarsa.r, window)
    rolling_error_vi = log.(rolling(std, results.value_iteration.r, window)/3)
    rolling_error_ql = log.(rolling(std, results.q_learning.r, window)/3)
    rolling_error_sarsa = log.(rolling(std, results.sarsa.r, window)/3)
    
    num_simulations = rolling(minimum, 1:N_sim, window)

    fig = plot(num_simulations, rolling_mean_vi,
		ribbon=rolling_error_vi, fillalpha=0.2,
        color="blue", label="Value iteration", legend=(0.8, 0.65))
    plot!(num_simulations, rolling_mean_ql,
		ribbon=rolling_error_ql, fillalpha=0.2,
        color="red", label="Q-learning")
    plot!(num_simulations, rolling_mean_sarsa,
		ribbon=rolling_error_sarsa, fillalpha=0.2,
        color="black", label="SARSA")

    xlabel!("number of simulations")
    ylabel!("mean reward")
    title!("Rolling Mean")
    !isdir(dir) && mkdir(dir) # create directory
	savefig(fig, dir*"/"*fname)
    return fig
end

function plot_optimality_vs_compute(results; dir="", fname="optimality_vs_compute")
    fig = plot()
    ordering = "bl_vi", "vi", "qlearning", "sarsa"
    for (i, solver_name) in enumerate(ordering)
        !(solver_name in keys(results)) && continue
        (comp_times, mean_rewards, stddev_rewards) = results[solver_name]
        errors = log.(stddev_rewards ./ 3)
        errors = [e>0 ? e : 0.0 for e in errors]
        lbl = (solver_name == "bl_vi") ? "bl_vi (ours)" : solver_name
        plot!(comp_times, mean_rewards, ribbon=errors, fillalpha = 0.2, label=lbl, color = i, legend = :best)
    end
    xlabel!("Computation time (s)")
    ylabel!("Mean discounted reward")
    title!("Optimality vs. Compute Time")
    # xlims!((0,200))
    !isdir(dir) && mkdir(dir) # create directory
    savefig(fig, dir*"/"*fname)
    savefig(fig, dir*"/"*fname*".pdf")
    return fig
end

function plot_optimality_compute_vs_gridsize(results; dir="varygrid", fname="vary_gridsize")
    fig = plot()
    ordering = "bl_vi", "vi"
    num_grid_sizes = length(results[ordering[1]][1])
    grid_sizes = ["10x10\n5 tgts", "20x20\n10tgts", "30x30\n15tgts", "40x40\n15tgts", "50x50\n15tgts"]
    p = twinx()
    max_reward = 0
    max_time = 0
    for (i, solver_name) in enumerate(ordering)
        !(solver_name in keys(results)) && continue
        (comp_times, mean_rewards, stddev_rewards) = results[solver_name]
        errors = log.(stddev_rewards ./ 3)
        errors = [e>0 ? e : 0.0 for e in errors]
        lbl = (solver_name == "bl_vi") ? "bl_vi (ours)" : solver_name
        plot!(1:length(mean_rewards), mean_rewards, ribbon=errors, fillalpha = 0.2, label=lbl, color = i, legend = :left, ylabel = "Mean discounted reward", grid = :off)
        plot!(p, 1:length(mean_rewards), comp_times, linestyle=:dash, label=lbl, color = i, legend = :right, ylabel = "Computation time (s)", box=:on)
        max_reward = (maximum(mean_rewards) > max_reward) ? maximum(mean_rewards) : max_reward
        max_time = (maximum(comp_times) > max_time) ? maximum(comp_times) : max_time
    end
    xlabel!("Problem complexity")
    xticks!((1:num_grid_sizes), grid_sizes[1:num_grid_sizes])
    ylims!((0, max_reward))
    ylims!(p, (0, max_time))
    title!("Rewards and Computation Time vs. Problem Complexity")
    !isdir(dir) && mkdir(dir) # create directory
    savefig(fig, dir*"/"*fname)
    savefig(fig, dir*"/"*fname*".pdf")
    return fig
end


function plot_bilevel_simulated_episode(mdp::RoverWorld.RoverWorldMDP, 
                                        sar_history::Vector{Tuple{Union{HLRoverWorld.HLState, LLRoverWorld.LLState}, Union{HLRoverWorld.HLAction, LLRoverWorld.LLAction}, Union{Float64, Nothing}}}; 
                                        dir="", 
                                        fname = "bilevel_mdp_episode", 
                                        fig_title="Bi-level MDP Episode")
    gr()
    (xmax, ymax) = mdp.grid_size
    fig = plot([], framestyle=:box, aspect_ratio=:equal, xlims=(0.5, xmax+0.5), ylims=(0.5, ymax+0.5), xticks=1:xmax, yticks=1:ymax, grid=false, minorgrid=true, minorticks = 2, minorgridalpha = 0.1, label="", legend=true)
    
    ## Plot start point
    start_point = sar_history[1][1]
    scatter!([(start_point.x, start_point.y)], color=start.color, markershape=start.markershape, markersize=start.markersize, markeralpha=start.markeralpha, label=start.label)
    
    ## Plot high level targets and path leading to them
    hl_num = 0; hl_locs = Vector{Int64}();
    for tstep in 1:length(sar_history)
        (s, a, r) = sar_history[tstep]
        if s isa HLRoverWorld.HLState && tstep > 1
            hl_num += 1
            push!(hl_locs, tstep)
            color_seq = hl_num
            scatter!([(s.x, s.y)], color=color_seq, markersize=targets.markersize, markeralpha=targets.markeralpha, label="High-level tgt $hl_num")
            prev = hl_num == 1 ? 0 : hl_locs[hl_num-1]
            LL_history = [(j, (s, a, r)) for (j, (s, a, r)) in enumerate(sar_history) if (prev<j<tstep && s isa LLRoverWorld.LLState)]
            plot!([s.x for (j, (s, a, r)) in LL_history], [s.y for (j, (s, a, r)) in LL_history], 
                color = color_seq,
                marker = (llp.markershape, llp.markersize, llp.markeralpha),
                label="")
        end
        if tstep == length(sar_history)
            prev = hl_num > 0 ? hl_locs[hl_num] : 0
            LL_history = [(j, (s, a, r)) for (j, (s, a, r)) in enumerate(sar_history) if (prev<j<=tstep && s isa LLRoverWorld.LLState)]
            plot!([s.x for (j, (s, a, r)) in LL_history], [s.y for (j, (s, a, r)) in LL_history], 
                color = defpath.color,
                marker = (llp.markershape, llp.markersize, llp.markeralpha),
                label="")
        end
    end

    ## Plot measurements
    labeled_measurements = false
    for tstep in 1:length(sar_history)
        (s, a, r) = sar_history[tstep]
        if (a == LLRoverWorld.MEASURE)
            scatter!([(s.x, s.y)], color=meas.color, markershape=meas.markershape, markersize=meas.markersize, markeralpha=meas.markeralpha, label= labeled_measurements ? "" : "Measurements")
            labeled_measurements = true
        end
    end

    ## Plot obstacles
    labeled = false
    for x in 1:xmax
        for y in 1:ymax
            if all(val!= 0.0 for val in mdp.obstacles_grid[x,y,:])
                scatter!([x], [y], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled ? "" : "Obstacles")
                labeled = true
            end
        end
    end
    
    ## Plot finish point
    finish_point = last(sar_history)[1]
    scatter!([(finish_point.x, finish_point.y)], color=finish.color, markershape=finish.markershape, markersize=finish.markersize, markeralpha=finish.markeralpha, label=finish.label)
    
    ## Other
    # rock_img = load("C:\\Users\\sbanerj6\\.julia\\dev\\BiMDPs\\src/imgs/rock.jpg")
    # fig = plot!(rock_img, yflip=false, aspect_ratio=1.0, label="Obstacles")
    # for (i, (s, a, r)) in enumerate(sar_history)
    #     if s isa HLRoverWorld.HLState
    #         color = "blue"
    #         markersize = 10
    #         markeralpha = i / length(sar_history)
    #     else
    #         color = "red"
    #         markersize = 5
    #         markeralpha = i / length(sar_history)
    #     end
    #     scatter!([(s.x, s.y)], color=color, markersize=markersize, markeralpha=markeralpha)
    # end
    title!(fig_title)
    savefig(fig, joinpath(dir, fname))
    savefig(fig, joinpath(dir, fname*".pdf"))
    return fig
end

function plot_bilevel_simulated_episode(mdp::MRoverWorld.MRoverWorldMDP, 
                                        sar_history::Vector{Tuple{Union{HLRoverWorld.HLState, MLLRoverWorld.MLLState}, Union{HLRoverWorld.HLAction, MLLRoverWorld.MLLAction}, Union{Float64, Nothing}}}; 
                                        dir="", 
                                        fname = "bilevel_mdp_episode", 
                                        fig_title="Bi-level MDP Episode")
    gr()
    (xmax, ymax) = mdp.grid_size
    fig = plot([], framestyle=:box, aspect_ratio=:equal, xlims=(0.5, xmax+0.5), ylims=(0.5, ymax+0.5), xticks=1:xmax, yticks=1:ymax, grid=false, minorgrid=true, minorticks = 2, minorgridalpha = 0.1, label="", legend=true)
    
    ## Plot start point
    start_point = sar_history[1][1]
    scatter!([(start_point.x, start_point.y)], color=start.color, markershape=start.markershape, markersize=start.markersize, markeralpha=start.markeralpha, label=start.label)
    
    ## Plot high level targets and path leading to them
    hl_num = 0; hl_locs = Vector{Int64}();
    for tstep in 1:length(sar_history)
        (s, a, r) = sar_history[tstep]
        if s isa HLRoverWorld.HLState && tstep > 1
            hl_num += 1
            push!(hl_locs, tstep)
            color_seq = hl_num
            scatter!([(s.x, s.y)], color=color_seq, markersize=targets.markersize, markeralpha=targets.markeralpha, label="High-level tgt $hl_num")
            prev = hl_num == 1 ? 0 : hl_locs[hl_num-1]
            LL_history = [(j, (s, a, r)) for (j, (s, a, r)) in enumerate(sar_history) if (prev<j<tstep && s isa MLLRoverWorld.MLLState)]
            plot!([s.x for (j, (s, a, r)) in LL_history], [s.y for (j, (s, a, r)) in LL_history], 
                color = color_seq,
                marker = (llp.markershape, llp.markersize, llp.markeralpha),
                label="")
        end
        if tstep == length(sar_history)
            prev = hl_num > 0 ? hl_locs[hl_num] : 0
            LL_history = [(j, (s, a, r)) for (j, (s, a, r)) in enumerate(sar_history) if (prev<j<=tstep && s isa MLLRoverWorld.MLLState)]
            plot!([s.x for (j, (s, a, r)) in LL_history], [s.y for (j, (s, a, r)) in LL_history], 
                color = defpath.color,
                marker = (llp.markershape, llp.markersize, llp.markeralpha),
                label="")
        end
    end

    ## Plot measurements
    labeled_measurements = false
    for tstep in 1:length(sar_history)
        (s, a, r) = sar_history[tstep]
        if (a == MLLRoverWorld.MEASURE)
            scatter!([(s.x, s.y)], color=meas.color, markershape=meas.markershape, markersize=meas.markersize, markeralpha=meas.markeralpha, label= labeled_measurements ? "" : "Measurements")
            labeled_measurements = true
        end
    end

    ## Plot obstacles
    labeled = false
    for x in 1:xmax
        for y in 1:ymax
            if all(val!= 0.0 for val in mdp.obstacles_grid[x,y,:])
                scatter!([x], [y], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled ? "" : "Obstacles")
                labeled = true
            end
        end
    end
    
    ## Plot finish point
    finish_point = last(sar_history)[1]
    scatter!([(finish_point.x, finish_point.y)], color=finish.color, markershape=finish.markershape, markersize=finish.markersize, markeralpha=finish.markeralpha, label=finish.label)
    
    title!(fig_title)
    savefig(fig, joinpath(dir, fname))
    savefig(fig, joinpath(dir, fname*".pdf"))
    return fig
end

function plot_finegrained_simulated_episode(mdp::RoverWorld.RoverWorldMDP, 
                                            sar_history::Vector{Tuple{RoverWorld.State, RoverWorld.Action, Float64}}; 
                                            dir="", 
                                            fname = "flat_mdp_episode", 
                                            fig_title="Flat MDP Episode")
    gr()
    (xmax, ymax) = mdp.grid_size
    fig = plot([], framestyle=:box, aspect_ratio=:equal, xlims=(0.5, xmax+0.5), ylims=(0.5, ymax+0.5), xticks=1:xmax, yticks=1:ymax, grid=false, minorgrid=true, minorticks = 2, minorgridalpha = 0.1, label="", legend=true)
    
    ## Plot start point
    start_point = sar_history[1][1]
    scatter!([(start_point.x, start_point.y)], color=start.color, markershape=start.markershape, markersize=start.markersize, markeralpha=start.markeralpha, label=start.label)
    
    ## Plot path
    plot!([s.x for (i, (s, a, r)) in enumerate(sar_history)], [s.y for (i, (s, a, r)) in enumerate(sar_history)], 
            color = defpath.color, 
            marker=(defpath.markershape, defpath.markersize, defpath.markeralpha, defpath.color), 
            label="Path")

    ## Plot target rewards + measurement rewards
    labeled_tgts = false
    labeled_measurements = false
    for tstep in 1:length(sar_history)
        (s, a, r) = sar_history[tstep]
        if r > 0
            found = false
            if a == RoverWorld.MEASURE
                scatter!([(s.x, s.y)], color=meas.color, markershape=meas.markershape, markersize=meas.markersize, markeralpha=meas.markeralpha, label= labeled_measurements ? "" : "Measurements")
                found = true
                labeled_measurements = true
            end
            if !found
                for (tgt_id, ((x, y), (t0,tf), val)) in mdp.tgts
                    if (s.x, s.y) == (x, y) && t0 <= s.t <= tf
                        scatter!([(s.x, s.y)], color=targets.color, markershape=targets.markershape, markersize=targets.markersize, markeralpha=targets.markeralpha, label = labeled_tgts ? "" : "Rewards")
                        labeled_tgts = true
                        found = true
                        break
                    end
                end
            end
        end
    end
    
    ## Plot obstacles
    labeled = false
    for x in 1:xmax
        for y in 1:ymax
            if all(val!= 0.0 for val in mdp.obstacles_grid[x,y,:])
                scatter!([x], [y], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled ? "" : "Obstacles")
                labeled = true
            end
        end
    end

    ## Plot finish point
    finish_point = last(sar_history)[1]
    scatter!([(finish_point.x, finish_point.y)], color=finish.color, markershape=finish.markershape, markersize=finish.markersize, markeralpha=finish.markeralpha, label=finish.label)
    plot!(legend=:best)
    title!(fig_title)
    !isdir(dir) && mkdir(dir) # create directory
    savefig(fig, joinpath(dir, fname))
    savefig(fig, joinpath(dir, fname*".pdf"))
    return fig
end

function plot_finegrained_simulated_episode(mdp::MRoverWorld.MRoverWorldMDP, 
                                            sar_history::Vector{Tuple{MRoverWorld.MState, MRoverWorld.MAction, Float64}}; 
                                            dir="", 
                                            fname = "flat_mdp_episode", 
                                            fig_title="Flat MDP Episode")
    gr()
    (xmax, ymax) = mdp.grid_size
    fig = plot([], framestyle=:box, aspect_ratio=:equal, xlims=(0.5, xmax+0.5), ylims=(0.5, ymax+0.5), xticks=1:xmax, yticks=1:ymax, grid=false, minorgrid=true, minorticks = 2, minorgridalpha = 0.1, label="", legend=true)
    
    ## Plot start point
    start_point = sar_history[1][1]
    scatter!([(start_point.x, start_point.y)], color=start.color, markershape=start.markershape, markersize=start.markersize, markeralpha=start.markeralpha, label=start.label)
    
    ## Plot path
    plot!([s.x for (i, (s, a, r)) in enumerate(sar_history)], [s.y for (i, (s, a, r)) in enumerate(sar_history)], 
            color = defpath.color, 
            marker=(defpath.markershape, defpath.markersize, defpath.markeralpha, defpath.color), 
            label="Path")

    ## Plot target rewards + measurement rewards
    labeled_tgts = false
    labeled_measurements = false
    for tstep in 1:length(sar_history)
        (s, a, r) = sar_history[tstep]
        if r > 0
            found = false
            if a == MRoverWorld.MEASURE
                scatter!([(s.x, s.y)], color=meas.color, markershape=meas.markershape, markersize=meas.markersize, markeralpha=meas.markeralpha, label= labeled_measurements ? "" : "Measurements")
                found = true
                labeled_measurements = true
            end
            if !found
                for (tgt_id, ((x, y), (t0,tf), val)) in mdp.tgts
                    if (s.x, s.y) == (x, y) && t0 <= s.t <= tf
                        scatter!([(s.x, s.y)], color=targets.color, markershape=targets.markershape, markersize=targets.markersize, markeralpha=targets.markeralpha, label = labeled_tgts ? "" : "Rewards")
                        labeled_tgts = true
                        found = true
                        break
                    end
                end
            end
        end
    end
    
    ## Plot obstacles
    labeled = false
    for x in 1:xmax
        for y in 1:ymax
            if all(val!= 0.0 for val in mdp.obstacles_grid[x,y,:])
                scatter!([x], [y], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled ? "" : "Obstacles")
                labeled = true
            end
        end
    end

    ## Plot finish point
    finish_point = last(sar_history)[1]
    scatter!([(finish_point.x, finish_point.y)], color=finish.color, markershape=finish.markershape, markersize=finish.markersize, markeralpha=finish.markeralpha, label=finish.label)
    plot!(legend=:best)
    title!(fig_title)
    !isdir(dir) && mkdir(dir) # create directory
    savefig(fig, joinpath(dir, fname))
    savefig(fig, joinpath(dir, fname*".pdf"))
    return fig
end

struct PlotElement
    color::String
    markershape::Symbol
    markersize::Int64
    markeralpha::Float64
    label::String
end

defpath = PlotElement("black", :circle, 5, 0.5, "Path")
obs = PlotElement("red", :square, 10, 0.5, "Obstacles")
llp = PlotElement("red", :circle, 3, 0.5, "Low-level path")
hlp = PlotElement("blue", :star5, 10, 0.5, "High-level path")
targets = PlotElement(hlp.color, hlp.markershape, hlp.markersize, hlp.markeralpha, "Targets")
meas = PlotElement("green", :dtriangle, 10, 0.5, "Measurements")

start = PlotElement("black", :x, 8, 1.0, "Start point")
finish = PlotElement("black", :diamond, 8, 1.0, "End point")