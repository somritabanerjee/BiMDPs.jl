using Plots
using Images

function plot_bilevel_simulated_episode(mdp::RoverWorld.RoverWorldMDP, 
                                        sar_history::Vector{Tuple{Union{HLRoverWorld.HLState, LLRoverWorld.LLState}, Union{HLRoverWorld.HLAction, LLRoverWorld.LLAction}, Union{Float64, Nothing}}}; 
                                        dir="", 
                                        fname = "bilevel_episode", 
                                        fig_title="Bi-level Simulated Episode")
    gr()
    (xmax, ymax) = mdp.grid_size
    fig = plot([], framestyle=:box, aspect_ratio=:equal, xlims=(0.5, xmax+0.5), ylims=(0.5, ymax+0.5), xticks=1:xmax, yticks=1:ymax, grid=false, minorgrid=true, minorticks = 2, minorgridalpha = 0.1, label="", legend=true)
    
    ## Plot start point
    start_point = sar_history[1][1]
    scatter!([(start_point.x, start_point.y)], color=start.color, markershape=start.markershape, markersize=start.markersize, markeralpha=start.markeralpha, label=start.label)
    
    ## Plot high level targets
    HL_history = [(i, (s, a, r)) for (i, (s, a, r)) in enumerate(sar_history) if s isa HLRoverWorld.HLState && i>1]
    for (hl_num, (i, (s, a, r))) in enumerate(HL_history)
        markeralpha = hl_num / length(HL_history)
        scatter!([(s.x, s.y)], color=targets.color, markersize=targets.markersize, markeralpha=markeralpha, label="High-level tgt $hl_num")
    end
    
    ## Plot low level path
    LL_history = [(i, (s, a, r)) for (i, (s, a, r)) in enumerate(sar_history) if s isa LLRoverWorld.LLState]
    plot!([s.x for (i, (s, a, r)) in LL_history], [s.y for (i, (s, a, r)) in LL_history], 
            color = llp.color, 
            marker=(llp.markershape, llp.markersize, [i/length(sar_history) for (i, (s, a, r)) in LL_history], llp.color), 
            label="Low-level path")
    
    ## Plot obstacles    
    for ((x_obs, y_obs), (t0_obs, tf_obs), val_obs) in mdp.obstacles
        labeled = false
        if t0_obs == 1 && tf_obs == mdp.max_time
            scatter!([x_obs], [y_obs], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled ? "" : "Obstacles")
            # plot!([x_obs], [y_obs], reverse(rock_img, dims=1), yflip=false, aspect_ratio=:none, label= labeled ? "" : "Obstacles")
            labeled = true
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
    return fig
end

function plot_finegrained_simulated_episode(mdp::RoverWorld.RoverWorldMDP, 
                                            sar_history::Vector{Tuple{RoverWorld.State, RoverWorld.Action, Float64}}; 
                                            dir="", 
                                            fname = "finegrained_episode", 
                                            fig_title="Fine-grained Simulated Episode")
    gr()
    (xmax, ymax) = mdp.grid_size
    fig = plot([], framestyle=:box, aspect_ratio=:equal, xlims=(0.5, xmax+0.5), ylims=(0.5, ymax+0.5), xticks=1:xmax, yticks=1:ymax, grid=false, minorgrid=true, minorticks = 2, minorgridalpha = 0.1, label="", legend=true)
    # for (i, (s, a, r)) in enumerate(sar_history)
    #     color = "black"
    #     markersize = 5
    #     markeralpha = i / length(sar_history)
    #     scatter!([(s.x, s.y)], color=color, markersize=markersize, markeralpha=markeralpha)
    # end
    plot!([s.x for (i, (s, a, r)) in enumerate(sar_history)], [s.y for (i, (s, a, r)) in enumerate(sar_history)], 
            color = "black", 
            marker=(:circle, 5, [i/length(sar_history) for (i, (s, a, r)) in enumerate(sar_history)], :black), 
            label="Fine-grained path")
    for ((x_obs, y_obs), (t0_obs, tf_obs), val_obs) in mdp.obstacles
        labeled = false
        if t0_obs == 1 && tf_obs == mdp.max_time
            scatter!([x_obs], [y_obs], color="black", markershape=:utriangle, markersize=10, markeralpha=0.5, label= labeled ? "" : "Obstacles")
            labeled = true
        end
    end    
    # Landing site
    # Rewards/targets
    # Measurements
    title!(fig_title)
    savefig(fig, joinpath(dir, fname))
    return fig
end

struct PlotElement
    color::String
    markershape::Symbol
    markersize::Int64
    markeralpha::Float64
    label::String
end

fgp = PlotElement("black", :circle, 5, 0.5, "Path")
obs = PlotElement("black", :utriangle, 10, 0.5, "Obstacles")
llp = PlotElement("red", :circle, 5, 0.5, "Low-level path")
# hlp = PlotElement("blue", :star, 10, 0.5, "High-level path")
hlp = PlotElement("blue", :star5, 10, 0.5, "High-level path")
targets = PlotElement(hlp.color, hlp.markershape, hlp.markersize, hlp.markeralpha, "Targets")
meas = PlotElement("green", :diamond, 5, 0.5, "Measurements")
# meas = PlotElement("green", :star-triangle-down-dot, 5, 0.5, "Measurements")

start = PlotElement("red", :x, 8, 0.5, "Start point")
finish = PlotElement("red", :diamond, 8, 1.0, "End point")
