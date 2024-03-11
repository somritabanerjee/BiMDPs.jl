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
    augment_title=true,
    dpi = 1200)

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
                minorgridalpha = 1.0,
                dpi=dpi)
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

function create_reward_field_evolution_gif(mdp; dir="", fname = "reward_evolution", dpi=1200)
	sim_frames = Frames(MIME("image/png"), fps=2)
    frame_i = nothing
	for i in 1:1:mdp.max_time
		frame_i = plot_grid_world(mdp, RoverWorld.NothingPolicy();
			timestep = i, outline=false, 
            show_rewards = true, fig_title="Reward Field Evolution", dpi=dpi)
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

function helper_plot_optimality_vs_compute!(fig, ordering, results; use_title = true)
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
    use_title && title!("Optimality vs. Compute Time")
    xlims!((0,200))
    ylims!((0,45))
end


function plot_optimality_vs_compute(results; dir="", fname="optimality_vs_compute",dpi=1200, with_ablations=false, use_title=true)
    if !with_ablations
        ordering = "bl_vi", "vi", "qlearning", "sarsa"
        fig = plot(dpi=dpi)
        helper_plot_optimality_vs_compute!(fig, ordering, results, use_title=use_title)
        !isdir(dir) && mkdir(dir) # create directory
        savefig(fig, dir*"/"*fname)
        savefig(fig, dir*"/"*fname*".pdf")
        return fig
    else
        orderings = [("dummy","vi"),("bl_vi", "vi"),("bl_vi", "vi", "qlearning", "sarsa")]
        for (i,ord) in enumerate(orderings)
            fig = plot(dpi=dpi)
            helper_plot_optimality_vs_compute!(fig, ord, results, use_title=use_title)
            plot!(legend=:bottomright)
            !isdir(dir) && mkdir(dir) # create directory
            savefig(fig, dir*"/"*fname*"_ablation_$i")
        end
    end
end

function helper_plot_optimality_compute_vs_gridsize!(fig, ordering, results; use_title=true, show_compute=true, show_rewards=true, show_relative_reward=true)
    # num_grid_sizes = length(results[ordering[1]][1])
    num_grid_sizes = 5 # hard-coding to allow for ablations
    grid_sizes = ["10x10\n5 tgts", "20x20\n10tgts", "30x30\n15tgts", "40x40\n15tgts", "50x50\n15tgts"]
    if (show_compute && show_rewards)
        p = twinx()
    end
    max_reward = 0
    max_time = 0
    # Plot best fit line
    xs = [1,4,9,16,25]
    ys = 2.53153.*(xs.^2) .- 14.4211
    plot!(1:length(results["vi"][2]), ys, label=L"\mathcal{O}( \mathcal{S}^2 \mathcal{A} )", color = :black, linestyle=:dashdot)

    for (i, solver_name) in enumerate(ordering)
        !(solver_name in keys(results)) && continue
        (comp_times, mean_rewards, stddev_rewards) = results[solver_name]
        errors = log.(stddev_rewards ./ 3)
        errors = [e>0 ? e : 0.0 for e in errors]
        lbl = (solver_name == "bl_vi") ? "bl_vi (ours)" : solver_name
        clr = (solver_name == "bl_vi") ? 1 : ((solver_name == "vi") ? 2 : i)
        show_compute && plot!(1:length(mean_rewards), comp_times, label=lbl, color = clr, legend = :left, ylabel = "Computation time (s)", box=:on)
        if show_rewards
            if show_relative_reward
                relative_rewards = mean_rewards ./ results["vi"][2] * 100.0
                relative_reward_errors = (solver_name != "vi") ? (errors ./ results["vi"][2] * 100.0) : zeros(length(mean_rewards))
                plot!(p, 1:length(mean_rewards), relative_rewards, linestyle=:dash, ribbon=relative_reward_errors, fillalpha = 0.2, label=lbl, color = clr, legend = :right, ylabel = "Percent of max reward", grid = :off)
            else
               plot!(p, 1:length(mean_rewards), mean_rewards, linestyle=:dash, ribbon=errors, fillalpha = 0.2, label=lbl, color = clr, legend = :right, ylabel = "Mean discounted reward", grid = :off)
            end
        end
        
        max_reward = (maximum(mean_rewards) > max_reward) ? maximum(mean_rewards) : max_reward
        max_time = (maximum(comp_times) > max_time) ? maximum(comp_times) : max_time
    end
    xlabel!("Problem complexity")
    xticks!((1:num_grid_sizes), grid_sizes[1:num_grid_sizes])
    show_compute && ylims!((0, max_time))
    show_rewards && !show_relative_reward && ylims!(p, (0, max_reward))
    show_rewards && show_relative_reward && ylims!(p, (0, 110))
    use_title && title!("Rewards and Computation Time vs. Problem Complexity")
end

function plot_optimality_compute_vs_gridsize(results; dir="varygrid", fname="vary_gridsize",dpi=1200, use_title=true, with_ablations=false, show_relative_reward=true)
    if !with_ablations
        fig = plot(dpi=dpi)
        ordering = "vi", "bl_vi"
        helper_plot_optimality_compute_vs_gridsize!(fig, ordering, results, use_title=use_title, show_relative_reward = show_relative_reward)
        !isdir(dir) && mkdir(dir) # create directory
        savefig(fig, dir*"/"*fname)
        savefig(fig, dir*"/"*fname*".pdf")
        return fig
    else
        
        i = 1 # only vi and only compute
        ordering = "vi", "dummy"
        show_compute = true
        show_rewards = false
        fig = plot(dpi=dpi)
        helper_plot_optimality_compute_vs_gridsize!(fig, ordering, results, use_title=use_title, show_compute=show_compute, show_rewards=show_rewards, show_relative_reward=show_relative_reward)
        !isdir(dir) && mkdir(dir) # create directory
        savefig(fig, dir*"/"*fname*"_ablation_$i")

        i = 2 # vi and blvi and only compute
        ordering = "vi", "bl_vi"
        show_compute = true
        show_rewards = false
        fig = plot(dpi=dpi)
        helper_plot_optimality_compute_vs_gridsize!(fig, ordering, results, use_title=use_title, show_compute=show_compute, show_rewards=show_rewards, show_relative_reward=show_relative_reward)
        !isdir(dir) && mkdir(dir) # create directory
        savefig(fig, dir*"/"*fname*"_ablation_$i")
       
        i = 3  # vi and blvi and both rewards and compute
        ordering = "vi", "bl_vi"
        show_compute = true
        show_rewards = true
        fig = plot(dpi=dpi)
        helper_plot_optimality_compute_vs_gridsize!(fig, ordering, results, use_title=use_title, show_compute=show_compute, show_rewards=show_rewards, show_relative_reward=show_relative_reward)
        !isdir(dir) && mkdir(dir) # create directory
        savefig(fig, dir*"/"*fname*"_ablation_$i")
    end
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
                                        fig_title="Bi-level MDP Episode",
                                        dpi::Int64 = 1200)
    fig = plot_frame_bilevel(mdp, sar_history, fig_title=fig_title, dpi=dpi)
    savefig(fig, joinpath(dir, fname))
    savefig(fig, joinpath(dir, fname*".pdf"))
    return fig
end

function plot_frame_bilevel(mdp::MRoverWorld.MRoverWorldMDP, 
                                sar_history::Vector{Tuple{Union{HLRoverWorld.HLState, MLLRoverWorld.MLLState}, Union{HLRoverWorld.HLAction, MLLRoverWorld.MLLAction}, Union{Float64, Nothing}}};
                                timestep::Int64 = -1,
                                fig_title::String = "Title ??",
                                dpi::Int64 = 1200,
                                dynamic_obstacles = false)
    total_time = sar_history[end][1].t
    show_current_position = true
    if timestep == -1
        timestep = total_time
        show_current_position = false
    end
    # Limit sar_history to timestep
    sar_history_trim = [(s, a, r) for (s, a, r) in sar_history if s.t <= timestep]
    gr()
    (xmax, ymax) = mdp.grid_size
    fig = plot([], framestyle=:box, aspect_ratio=:equal, xlims=(0.5, xmax+0.5), ylims=(0.5, ymax+0.5), xticks=1:xmax, yticks=1:ymax, grid=false, minorgrid=true, minorticks = 2, minorgridalpha = 0.1, label="", legend=true, dpi=dpi)
    
    ## Plot start point
    start_point = sar_history[1][1]
    scatter!([(start_point.x, start_point.y)], color=start.color, markershape=start.markershape, markersize=start.markersize, markeralpha=start.markeralpha, label=start.label, dpi=dpi)
    
    if timestep == total_time
        # Do the plotting efficiently
        ## Plot high level targets and path leading to them
        hl_num = 0; hl_locs = Vector{Int64}();
        for tstep in 1:length(sar_history)
            (s, a, r) = sar_history[tstep]
            if s isa HLRoverWorld.HLState && tstep > 1
                hl_num += 1
                push!(hl_locs, tstep)
                color_seq = colorschemes[:tab10][hl_num]
                scatter!([(s.x, s.y)], color=color_seq, markersize=targets.markersize, markeralpha=targets.markeralpha, label="High-level tgt $hl_num", dpi=dpi)
                prev = hl_num == 1 ? 0 : hl_locs[hl_num-1]
                LL_history = [(j, (s, a, r)) for (j, (s, a, r)) in enumerate(sar_history) if (prev<j<tstep && s isa MLLRoverWorld.MLLState)]
                plot!([s.x for (j, (s, a, r)) in LL_history], [s.y for (j, (s, a, r)) in LL_history], 
                    color = color_seq,
                    marker = (llp.markershape, llp.markersize, llp.markeralpha),
                    label="", dpi=dpi)
            end
            if tstep == length(sar_history)
                prev = hl_num > 0 ? hl_locs[hl_num] : 0
                LL_history = [(j, (s, a, r)) for (j, (s, a, r)) in enumerate(sar_history) if (prev<j<=tstep && s isa MLLRoverWorld.MLLState)]
                plot!([s.x for (j, (s, a, r)) in LL_history], [s.y for (j, (s, a, r)) in LL_history], 
                    color = defpath.color,
                    marker = (llp.markershape, llp.markersize, llp.markeralpha),
                    label="", dpi=dpi)
            end
        end
    else
        num_hl = 0; ind_complete = 0;
        for (ind,(s,a,r)) in enumerate(sar_history_trim)
            if s isa HLRoverWorld.HLState
                num_hl += 1
                if num_hl<=2 # to prevent counting end as an hl tgt
                    # plot hl tgt with scatter
                    tgt_location = mdp.tgts[a.tgt][1]
                    color_seq = colorschemes[:tab10][num_hl]
                    scatter!([(tgt_location[1], tgt_location[2])], color=color_seq, markersize=targets.markersize, markeralpha=targets.markeralpha, label="High-level tgt $num_hl", dpi=dpi)
                end
                if num_hl > 1
                    # plot path from previous hl tgt to this hl tgt
                    LL_subset = sar_history_trim[ind_complete+1:ind]
                    color_seq = colorschemes[:tab10][num_hl-1]
                    plot!([s.x for (s,a,r) in LL_subset], [s.y for (s,a,r) in LL_subset], 
                    color = color_seq,
                    marker = (llp.markershape, llp.markersize, llp.markeralpha),
                    label="", dpi=dpi)
                end
                ind_complete = ind
            end
        end
        # plot path from last hl tgt to end
        LL_subset = sar_history_trim[ind_complete+1:end]
        plot!([s.x for (s,a,r) in LL_subset], [s.y for (s,a,r) in LL_subset],
            color = (num_hl<=2) ? colorschemes[:tab10][num_hl] : defpath.color,
            marker = (llp.markershape, llp.markersize, llp.markeralpha),
            label="", dpi=dpi)
    end

    ## Plot measurements
    labeled_measurements = false
    for (s,a,r) in sar_history_trim
        if (a == MLLRoverWorld.MEASURE)
            scatter!([(s.x, s.y)], color=meas.color, markershape=meas.markershape, markersize=meas.markersize, markeralpha=meas.markeralpha, label= labeled_measurements ? "" : "Measurements", dpi=dpi)
            labeled_measurements = true
        end
    end

    ## Plot obstacles
    labeled_obstacles = false
    if dynamic_obstacles
        # Plot all obstacles
        for x in 1:xmax
            for y in 1:ymax
                if mdp.obstacles_grid[x,y,timestep] != 0.0
                    scatter!([x], [y], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled_obstacles ? "" : "Obstacles", dpi=dpi)
                    labeled_obstacles = true
                end
            end
        end
    else
        # Plot only static obstacles
        for x in 1:xmax
            for y in 1:ymax
                if all(val!= 0.0 for val in mdp.obstacles_grid[x,y,:])
                    scatter!([x], [y], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled_obstacles ? "" : "Obstacles", dpi=dpi)
                    labeled_obstacles = true
                end
            end
        end
    end

    # Plot marker for current rover position
    if show_current_position
        (s, a, r) = sar_history_trim[end]
        scatter!([(s.x, s.y)], color=current.color, markershape=current.markershape, markersize=current.markersize, markeralpha=current.markeralpha, label="")
    end

    ## Plot finish point
    if timestep == total_time
        finish_point = last(sar_history)[1]
        scatter!([(finish_point.x, finish_point.y)], color=finish.color, markershape=finish.markershape, markersize=finish.markersize, markeralpha=finish.markeralpha, label=finish.label, dpi=dpi)
    end
    plot!(legend=:topleft, dpi=dpi)
    title!(fig_title)
    return fig
end

function animate_bilevel_simulated_episode(mdp::MRoverWorld.MRoverWorldMDP, 
                                                sar_history::Vector{Tuple{Union{HLRoverWorld.HLState, MLLRoverWorld.MLLState}, Union{HLRoverWorld.HLAction, MLLRoverWorld.MLLAction}, Union{Float64, Nothing}}};
                                                dir="", 
                                                fname = "bilevel_mdp_episode_gif", 
                                                fig_title="Bi-level MDP Episode",
                                                dpi::Int64 = 1200,
                                                fps::Int64 = 1)
	sim_frames = Frames(MIME("image/png"), fps=fps)
    frame_i = nothing
    total_time = sar_history[end][1].t
	for i in 1:total_time
		frame_i = plot_frame_bilevel(mdp, sar_history, timestep=i, fig_title=fig_title, dpi=dpi, dynamic_obstacles=true)
		push!(sim_frames, frame_i)
	end
    [push!(sim_frames, frame_i) for _ in 1:4] # duplicate last frame
	!isdir(dir) && mkdir(dir) # create directory
	write(dir*"/"*fname*".gif", sim_frames)
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
    fig = plot_frame_finegrained(mdp, sar_history, fig_title=fig_title)
    !isdir(dir) && mkdir(dir) # create directory
    savefig(fig, joinpath(dir, fname))
    savefig(fig, joinpath(dir, fname*".pdf"))
    return fig
end

function plot_frame_finegrained(mdp::MRoverWorld.MRoverWorldMDP, 
                                sar_history::Vector{Tuple{MRoverWorld.MState, MRoverWorld.MAction, Float64}};
                                timestep::Int64 = -1,
                                fig_title::String = "Title ??",
                                dpi::Int64 = 1200,
                                dynamic_obstacles = false)
    show_current_position = true
    if timestep == -1
        timestep = length(sar_history)
        show_current_position = false
    end
    gr()
    (xmax, ymax) = mdp.grid_size
    fig = plot([], framestyle=:box, aspect_ratio=:equal, xlims=(0.5, xmax+0.5), ylims=(0.5, ymax+0.5), xticks=1:xmax, yticks=1:ymax, grid=false, minorgrid=true, minorticks = 2, minorgridalpha = 0.1, label="", legend=true, dpi=dpi)
    ## Plot start point
    start_point = sar_history[1][1]
    scatter!([(start_point.x, start_point.y)], color=start.color, markershape=start.markershape, markersize=start.markersize, markeralpha=start.markeralpha, label=start.label)
    
    ## Plot path
    plot!([s.x for (i, (s, a, r)) in enumerate(sar_history[1:timestep])], [s.y for (i, (s, a, r)) in enumerate(sar_history[1:timestep])], 
            color = defpath.color, 
            marker=(defpath.markershape, defpath.markersize, defpath.markeralpha, defpath.color), 
            label="Path")

    ## Plot target rewards + measurement rewards
    labeled_tgts = false
    labeled_measurements = false
    for tstep in 1:length(sar_history[1:timestep])
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
    labeled_obstacles = false
    if dynamic_obstacles
        # Plot all obstacles
        for x in 1:xmax
            for y in 1:ymax
                if mdp.obstacles_grid[x,y,timestep] != 0.0
                    scatter!([x], [y], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled_obstacles ? "" : "Obstacles")
                    labeled_obstacles = true
                end
            end
        end
    else
        # Plot only static obstacles
        for x in 1:xmax
            for y in 1:ymax
                if all(val!= 0.0 for val in mdp.obstacles_grid[x,y,:])
                    scatter!([x], [y], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled_obstacles ? "" : "Obstacles")
                    labeled_obstacles = true
                end
            end
        end
    end

    # Plot marker for current rover position
    if show_current_position
        (s, a, r) = sar_history[timestep]
        scatter!([(s.x, s.y)], color=current.color, markershape=current.markershape, markersize=current.markersize, markeralpha=current.markeralpha, label="")
    end

    ## Plot finish point
    if timestep == length(sar_history)
        finish_point = last(sar_history)[1]
        scatter!([(finish_point.x, finish_point.y)], color=finish.color, markershape=finish.markershape, markersize=finish.markersize, markeralpha=finish.markeralpha, label=finish.label)
    end
    plot!(legend=:topleft)
    title!(fig_title)
    return fig
end

function animate_finegrained_simulated_episode(mdp::MRoverWorld.MRoverWorldMDP, 
                                                sar_history::Vector{Tuple{MRoverWorld.MState, MRoverWorld.MAction, Float64}}; 
                                                dir="", 
                                                fname = "flat_mdp_episode_gif", 
                                                fig_title="Flat MDP Episode",
                                                dpi::Int64 = 1200,
                                                fps::Int64 = 1)
	sim_frames = Frames(MIME("image/png"), fps=fps)
    frame_i = nothing
    num_steps = length(sar_history)
	for i in 1:num_steps
		frame_i = plot_frame_finegrained(mdp, sar_history, timestep=i, fig_title=fig_title, dpi=dpi, dynamic_obstacles=true)
		push!(sim_frames, frame_i)
	end
    [push!(sim_frames, frame_i) for _ in 1:4] # duplicate last frame
	!isdir(dir) && mkdir(dir) # create directory
	write(dir*"/"*fname*".gif", sim_frames)
end

struct PlotElement
    color::RGB{Float64}
    markershape::Symbol
    markersize::Int64
    markeralpha::Float64
    label::String
end

defpath = PlotElement(RGB(0,0,0), :circle, 5, 0.5, "Path") # black
obs = PlotElement(colorschemes[:tab10][4], :square, 10, 0.5, "Obstacles") # red
llp = PlotElement(colorschemes[:tab10][4], :circle, 3, 0.5, "Low-level path") # red
hlp = PlotElement(colorschemes[:tab10][1], :circle, 10, 0.5, "High-level path") # blue
targets = PlotElement(colorschemes[:tab10][1], :circle, 10, 0.5, "Targets") # blue
meas = PlotElement(colorschemes[:tab10][3], :dtriangle, 10, 0.5, "Measurements")

start = PlotElement(RGB(0,0,0), :x, 8, 1.0, "Start point")
finish = PlotElement(RGB(0,0,0), :diamond, 8, 1.0, "End point")
current = PlotElement(RGB(0,0,0), :circle, 10, 0.5, "Current position")