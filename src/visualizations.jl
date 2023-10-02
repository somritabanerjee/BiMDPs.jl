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
    for ((x_obs, y_obs), (t0_obs, tf_obs), val_obs) in mdp.obstacles
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
                println("Found measurement at $(s.x), $(s.y), $(s.t) $a !")
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
    for ((x_obs, y_obs), (t0_obs, tf_obs), val_obs) in mdp.obstacles
        if t0_obs == 1 && tf_obs == mdp.max_time
            scatter!([x_obs], [y_obs], color=obs.color, markershape=obs.markershape, markersize=obs.markersize, markeralpha=obs.markeralpha, label= labeled ? "" : "Obstacles")
            labeled = true
        end
    end

    ## Plot finish point
    finish_point = last(sar_history)[1]
    scatter!([(finish_point.x, finish_point.y)], color=finish.color, markershape=finish.markershape, markersize=finish.markersize, markeralpha=finish.markeralpha, label=finish.label)

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

defpath = PlotElement("black", :circle, 5, 0.5, "Path")
obs = PlotElement("red", :square, 10, 0.5, "Obstacles")
llp = PlotElement("red", :circle, 3, 0.5, "Low-level path")
hlp = PlotElement("blue", :star5, 10, 0.5, "High-level path")
targets = PlotElement(hlp.color, hlp.markershape, hlp.markersize, hlp.markeralpha, "Targets")
meas = PlotElement("green", :dtriangle, 10, 0.5, "Measurements")

start = PlotElement("black", :x, 8, 1.0, "Start point")
finish = PlotElement("black", :diamond, 8, 1.0, "End point")