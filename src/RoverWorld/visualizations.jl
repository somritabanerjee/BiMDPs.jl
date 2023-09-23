
# plot the U values (maximum Q over the actions)
# x = row, y = column, z = U-value
function plot_grid_world(mdp::RoverWorldMDP,
    policy::Policy=NothingPolicy(),
    iter=0,
    discount=NaN;
    outline=true,
    show_policy=true,
    show_rewards=false,
    outline_state::Union{State, Nothing}=nothing,
    timestep=nothing,
    fig_title="Grid World Policy Plot")

    gr()

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
    Uxy = reshape(U, xmax, ymax, mdp.max_time, 2^length(mdp.tgts))
    t = isnothing(outline_state) ? (isnothing(timestep) ? 1 : timestep) : outline_state.t
    visited = isnothing(outline_state) ? fill(false, length(mdp.tgts)) : outline_state.visited
    visited_idx = findfirst(isequal(visited), visited_list(mdp))
    Uxy_slice = Uxy[:,:,t,visited_idx]


    # plot values (i.e the U matrix)
    max_val = maximum(abs, Uxy)
    fig = heatmap(Uxy_slice',
                legend=:none,
                aspect_ratio=:equal,
                framestyle=:box,
                tickdirection=:out,
                color=cmap.colors,
                clims=(-max_val, max_val))
    xlims!(0.5, xmax+0.5)
    ylims!(0.5, ymax+0.5)
    xticks!(1:xmax)
    yticks!(1:ymax)

    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

    if show_rewards
        for s in filter(s->reward(mdp, s) != 0, states(mdp))
            r = reward(mdp, s)
            annotate!([(s.x, s.y, (r, :white, :center, 12, "Computer Modern"))])
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

    fig_title = fig_title * 
                    (isnan(discount)            ? "" : " (iter=$iter, γ=$discount)") * 
                    (isnothing(outline_state)   ? "" : " (t=$t)") * 
                    (isnothing(timestep)        ? "" : " (t=$t)")


    title!(fig_title)

    return fig
end

cmap = ColorScheme([Colors.RGB(180/255, 0.0, 0.0), Colors.RGB(1, 1, 1), Colors.RGB(0.0, 100/255, 0.0)], "custom", "threetone, red, white, and green")

function create_value_iteration_gif(mdp::RoverWorldMDP; dir="gifs")
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
			outline_state=steps[i].s, outline=false)
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
		frame_i = plot_grid_world(mdp, NothingPolicy();
			timestep = i, outline=false, fig_title="Reward Field Evolution")
		push!(sim_frames, frame_i)
	end
    [push!(sim_frames, frame_i) for _ in 1:4] # duplicate last frame
	!isdir(dir) && mkdir(dir) # create directory
	write(dir*"/"*fname*".gif", sim_frames)
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
