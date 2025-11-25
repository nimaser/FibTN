# function to generically plot a metagraph
function metagraphplot!(ax, mg::MetaGraph;
        nodelabelfunc=nothing, nodecolorfunc=nothing, nodeshapefunc=nothing,
        edgelabelfunc=nothing, edgecolorfunc=nothing,
        title=nothing, layout=Spring())
    # we will provide node properties as lists of values, one per vertex code, and edge properties as a Dict of Edge => edgeprop
    make_nodeprop_list = f -> map(f, labels(mg))
    make_edgeprop_dict = f -> Dict(Edge(code_for.((mg,),e)) => f(e) for e in edge_labels(mg))

    # common properties
    propsdict = Dict(:force_straight_edges=>true,
                     :nlabels_distance=>10,
                     :layout=>layout,
                    )

    # add other properties based on what was provided to the function
    if nodelabelfunc != nothing push!(propsdict, :nlabels=>make_nodeprop_list(nodelabelfunc)) end
    if edgelabelfunc != nothing push!(propsdict, :elabels=>make_edgeprop_dict(edgelabelfunc)) end
    if nodecolorfunc != nothing push!(propsdict, :node_color=>make_nodeprop_list(nodecolorfunc)) end
    if edgecolorfunc != nothing push!(propsdict, :edge_color=>make_edgeprop_dict(edgecolorfunc)) end

    # plot
    p = graphplot!(ax, mg; propsdict...)
    hidespines!(ax)
    hidedecorations!(ax)
    if title != nothing ax.title = title end
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

qgplot!(ax, mg::MetaGraph; vlabels=true, args...) = metagraphplot!(ax, mg;
                                                                   nodelabelfunc=vlabels ? string : nothing,
                                                                   edgecolorfunc=e -> mg[e...] ? :red : :black,
                                                                   args...
                                                                  )

tgplot!(ax, mg::MetaGraph; args...) = metagraphplot!(ax, mg;
                                                     nodelabelfunc=string,
                                                     edgelabelfunc=e -> tensoriter2label(tg[e...]),
                                                     nodeshapefunc=n -> type2shape(mg[n].type),
                                                     nodecolorfunc=n -> type2color(mg[n].type),
                                                     args...
                                                    )

# convenience function to plot all results of a computation
function statesplot(qg::MetaGraph, states::Dict{<:CartesianIndex, <:Tuple{<:Dict{<:Index, Int}, Float64}}; vlabels::Bool=true, layout::Any=Spring())
    qgs = []
    for (idx, (pvals, amp)) in states
        fillfrompvals(qg, pvals)
        f, ax, p = qgplot(qg, vlabels=vlabels, layout=layout)
        ax.title = "$(@sprintf("%.2f", amp))"
        push!(qgs, p)
    end
    plot(qgs...)
end
