using GLMakie, GraphMakie
using Printf
using NetworkLayout

# function to generically plot a metagraph
function metagraphplot!(ax, mg::MetaGraph;
        nodelabelfunc=nothing, nodecolorfunc=nothing, nodeshapefunc=nothing,
        edgelabelfunc=nothing, edgecolorfunc=nothing, edgewidthfunc=nothing,
        title=nothing, layout=Spring(), nlabeloffsetscale=0.15)
    # we will provide node properties as lists of values, one per vertex code, and edge properties as a Dict of Edge => edgeprop
    make_nodeprop_list = f -> map(f, labels(mg))
    make_edgeprop_dict = f -> Dict(Edge(code_for.((mg,),e)) => f(e) for e in edge_labels(mg))

    # common properties
    propsdict = Dict(:force_straight_edges=>true,
                     #:nlabels_distance=>5,
                     :nlabels_align=>(:center,:center),
                     :layout=>layout,
                    )

    # add other properties based on what was provided to the function
    if nodelabelfunc != nothing push!(propsdict, :nlabels=>make_nodeprop_list(nodelabelfunc)) end
    if edgelabelfunc != nothing push!(propsdict, :elabels=>make_edgeprop_dict(edgelabelfunc)) end
    if nodecolorfunc != nothing push!(propsdict, :node_color=>make_nodeprop_list(nodecolorfunc)) end
    if edgecolorfunc != nothing push!(propsdict, :edge_color=>make_edgeprop_dict(edgecolorfunc)) end
    if nodeshapefunc != nothing push!(propsdict, :node_marker=>make_nodeprop_list(nodeshapefunc)) end
    if edgewidthfunc != nothing push!(propsdict, :edge_width=>make_edgeprop_dict(edgewidthfunc)) end

    # plot
    p = graphplot!(ax, mg; propsdict...)
    if title != nothing ax.title = title end

    # offset node labels
    offsets = -(p[:node_pos][])# .- p[:node_pos][][1])
    offsets ./= replace!(norm.(offsets), 0=>1)
    replace!(offsets, 0=>1)
    p.nlabels_offset[] = offsets * nlabeloffsetscale
    autolimits!(ax)
    p
end

# define some convenient functions to generate display strings from iterators of indices and tensors respectively
indexiter2label = ii -> join([get_idtag(i) for i in ii], ';')
tensoriter2label = ti -> join([indexiter2label(inds(t)) for t in ti], " ")

# define some convenience functions to determine node shapes and colors based on tensor type
type2shape = t -> t == GSTriangle ? :hexagon : t == StringTripletVector ? :star4 : :circle
type2color = t -> t == GSTriangle ? :black : t == StringTripletVector ? :gray : :blue

# use Julia's closures to make partially filled functions for specialized plotting
rsgplot!(ax, mg::MetaGraph; args...) = metagraphplot!(ax, mg;
                                                      nodelabelfunc=string,
                                                      edgecolorfunc=e -> e ∈ mg[] || reverse(e) ∈ mg[] ? :red : :black,
                                                      args...
                                                     )

igplot!(ax, mg::MetaGraph; args...) = metagraphplot!(ax, mg;
                                                     nodelabelfunc=string,
                                                     edgelabelfunc=e -> indexiter2label(ig[e...]),
                                                     nodeshapefunc=n -> type2shape(mg[n].type),
                                                     nodecolorfunc=n -> type2color(mg[n].type),
                                                     args...
                                                    )

tgplot!(ax, mg::MetaGraph; args...) = metagraphplot!(ax, mg;
                                                     nodelabelfunc=string,
                                                     edgelabelfunc=e -> tensoriter2label(tg[e...]),
                                                     nodeshapefunc=n -> type2shape(mg[n].type),
                                                     nodecolorfunc=n -> type2color(mg[n].type),
                                                     args...
                                                    )

qgplot!(ax, mg::MetaGraph; vlabels=true, args...) = metagraphplot!(ax, mg;
                                                                   nodelabelfunc=vlabels ? string : nothing,
                                                                   nodecolorfunc=n -> mg[n] ? :red : :black,
                                                                   edgecolorfunc=e -> mg[e...] ? :red : :black,
                                                                   args...
                                                                  )

function calculategridsidelengths(area::Int)
    width = height = floor(sqrt(area))
    if width == sqrt(area) return Int(width), Int(height) end
    width += 1
    while width * height < area
        width += 1
    end
    Int(width), Int(height)
end

function getaxisgrid(f, area::Int)
    w, h = calculategridsidelengths(area)
    axs = [Axis(f[r, c]; aspect = DataAspect()) for r in 1:h, c in 1:w]
    w, h, axs
end

function finalize(f, axs)
    for ax in axs
        hidespines!(ax)
        hidedecorations!(ax)
    end
    resize_to_layout!(f)
end

# convenience function to plot all results of a computation on a vector of axes
function statesplot!(axs, qg::MetaGraph, states::Dict{<:CartesianIndex, <:Tuple{<:Dict{<:Index, Int}, Float64}}; vlabels::Bool=true, layout::Any=Spring(), args...)
    plots = []
    for (ax, (idx, (pvals, amp))) in zip(axs, states)
        fillfrompvals(qg, pvals)
        #p = qgplot!(ax, qg; vlabels=vlabels, layout=layout, title="$(Tuple(idx)) $(@sprintf("%.4f", amp))", args...)
        p = qgplot!(ax, qg; vlabels=vlabels, layout=layout, title="$(@sprintf("%.3f", amp))", args...)
        push!(plots, p)
        # remove the default interactions from this axis
        for i in [:dragpan, :limitreset, :rectanglezoom, :scrollzoom] deregister_interaction!(ax, i) end
        # add the popout window interaction to this axis
        if :popout_axis ∉ keys(ax.interactions)
            register_interaction!(ax, :popout_axis) do event::MouseEvent, ax
                if event.type === MouseEventTypes.leftclick
                    f_popout = Figure()
                    ax_popout = Axis(f_popout[1, 1]; aspect = DataAspect())
                    for i in [:dragpan, :limitreset, :rectanglezoom, :scrollzoom] deregister_interaction!(ax_popout, i) end
                    fillfrompvals(qg, pvals)
                    p_popout = qgplot!(ax_popout, qg; vlabels=vlabels, layout=layout, title="$(@sprintf("%.3f", amp))", args...)
                    finalize(f_popout, [ax_popout])
                    GLFW_win = GLMakie.to_native(display(GLMakie.Screen(), f_popout))
                    # make window easily closeable via keyboard
                    register_interaction!(ax_popout, :close_window_with_q) do event::KeysEvent, ax
                        if Keyboard.q ∈ event.keys
                            GLMakie.GLFW.SetWindowShouldClose(GLFW_win, true)
                        end
                    end
                end
            end
        end
    end
    #add_close_window_with_q_interaction!(axs[1])
    # add normalization to figure title
    N = get_normalization(s)
    axs[1].parent[0, :] = Label(axs[1].parent, "N = $(@sprintf("%.3f", N))")
    plots
end
