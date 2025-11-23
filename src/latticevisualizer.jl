type2shape = t -> t == GSTriangle ? :hexagon : t == StringTripletVector ? :rect : :circle
type2color = t -> t == GSTriangle ? :black : t == StringTripletVector ? :blue : :circle
idxset2label = s -> join([get_idtag(i) for i in s], ';')

commonargs = Dict(:force_straight_edges=>true,
                  :nlabels_fontsize=>12,
                  :nlabels_distance=>10,
                  :node_size=>10,
                 )
 
function rsgplot(rsg::MetaGraph; layout=Spring())
    # graphplot uses Graphs.jl codes, so provide vertex properties in vertex code ordered list
    # and provide edge properties as a dict of of (code1, code2) => edgeprop
    labelstrings = map(string, labels(rsg))
    separateargs = Dict(:nlabels=>collect(labelstrings),
                        :edge_color=>Dict(Edge(code_for.((rsg,),e)) => e ∈ rsg[] || reverse(e) ∈ rsg[] ? :red : :black for e in edge_labels(rsg))
                       )
    f, ax, p = graphplot(rsg; commonargs..., separateargs..., layout=layout)
    ax.title = "Rotation System Graph (rsg)"
    hidespines!(ax)
    hidedecorations!(ax)
    f, ax, p
end

function igplot(ig::MetaGraph; layout::Any=Spring())
    labelstrings = map(string, labels(ig))
    separateargs = Dict(:nlabels=>collect(labelstrings),
                        :elabels=>Dict(Edge(code_for.((ig,),e)) => idxset2label(ig[e...]) for e in edge_labels(ig))
                       )
    f, ax, p = graphplot(ig; commonargs..., separateargs..., layout=layout)
    ax.title = "Index Graph (ig)"
    hidespines!(ax)
    hidedecorations!(ax)
    f, ax, p
end

function qgplot(qg::MetaGraph; vlabels=true, layout::Any=Spring())
    labelstrings = map(string, labels(qg))
    separateargs = Dict{Any, Any}(:edge_color=>Dict(Edge(code_for.((qg,),e)) => qg[e...] ? :red : :black for e in edge_labels(qg)),
                       )
    if vlabels
        separateargs[:nlabels] = collect(labelstrings)
    end
    f, ax, p = graphplot(qg; commonargs..., separateargs..., layout=layout)
    ax.title = "Qubit Graph (qg)"
    hidespines!(ax)
    hidedecorations!(ax)
    f, ax, p
end

function tgplot(tg::MetaGraph; layout::Any=Spring())
    tensorset2label = ts -> join([idxset2label(inds(t)) for t in ts], " ")

    labelstrings = map(string, labels(tg))
    separateargs = Dict(:nlabels=>collect(labelstrings),
                        :elabels=>Dict(Edge(code_for.((tg,),e)) => tensorset2label(tg[e...]) for e in edge_labels(tg)),
                       )
    f, ax, p = graphplot(tg; commonargs..., separateargs..., layout=layout)
    ax.title = "Tensor Graph (tg)"
    hidespines!(ax)
    hidedecorations!(ax)
    f, ax, p
end

function statesplot(qg::MetaGraph, states::Dict{<:CartesianIndex, <:Tuple{<:Dict{<:Index, Int}, Float64}}; vlabels::Bool=true, layout::Any=Spring())
    qgs = []
    for (idx, (pvals, amp)) in states
        fillfrompvals(qg, pvals)
        qg_plot = qgplot(qg, vlabels)
        qg_plot.subplots[1].attr[:title] = "$(@sprintf("%.2f", amp))"
        qg_plot.subplots[1].attr[:titlefontsize] = 4
        push!(qgs, qg_plot)
    end
    plot(qgs...)
end
