using FibErrThresh.TensorNetworks

using GLMakie, GeometryBasics

export visualize

"""
Plots `tn` on `ax`.

Specifically, plots each tensor node using a scatterplot and each
contraction using a linesegment. Modifies the hover tooltip of the GLMakie
DataInspector so:
- hovering over an edge shows the contracted indices
- hovering over a node shows the tensor group number

Inputs are sorted by group so that the scatter index `i` corresponds to the
`i`-th element in group-sorted order, enabling reliable inspector lookups.
"""
function FibErrThresh.visualize(
    ax::Axis,
    tn::TensorNetwork,
    groups::Vector{Int},
    positions::Vector{Point2f},
    colors::Vector{Symbol},
    markers::Vector{Symbol},
)
    # sort all display arrays by group so scatter index i == sorted index i
    order = sortperm(groups)
    groups    = groups[order]
    positions = positions[order]
    colors    = colors[order]
    markers   = markers[order]

    position_from_group = Dict(g => p for (g, p) in zip(groups, positions))
    # get endpoints of edges, skipping contractions that touch a hidden (unpositioned) group
    edge_endpoints = []
    tnc = filter(c -> haskey(position_from_group, c.a.group) && haskey(position_from_group, c.b.group),
                 collect(get_contractions(tn)))
    for c in tnc
        pos1 = position_from_group[c.a.group]
        pos2 = position_from_group[c.b.group]
        push!(edge_endpoints, (pos1, pos2))
    end
    # plot linesegments
    segmentsresult = linesegments!(ax, edge_endpoints, color=:gray)
    segmentsresult.inspector_label = (plot, i, idx) -> begin
        i = i ÷ 2 # i counts up by endpoints, so by two per segment plotted
        ic = tnc[i]
        "$(ic.a.group) $(ic.a.port)\n$(ic.b.group) $(ic.b.port)"
    end
    # scatterplot to represent tensors (plotted in group-sorted order)
    scatterresult = scatter!(ax, positions, color=colors, marker=markers, markersize=40)
    scatterresult.inspector_label = (plot, i, idx) -> "$(groups[i])"
    autolimits!(ax) # compute limits based on the axis content
    segmentsresult, scatterresult
end

"""
Wraps the TensorNetwork visualize function, autofilling the groups, markers, and colors.
Groups with no entry in `position_from_group` are silently skipped (neither their node
nor any edge touching them is drawn).
"""
function FibErrThresh.visualize(ax::Axis, ttn::TypedTensorNetwork, position_from_group::Dict{Int,Point2f})
    groups    = filter(g -> haskey(position_from_group, g), collect(get_groups(ttn.tn)))
    positions = [position_from_group[g] for g in groups]
    colors    = [tensor_color(ttn.tensortype_from_group[g]) for g in groups]
    markers   = [tensor_marker(ttn.tensortype_from_group[g]) for g in groups]
    visualize(ax, ttn.tn, groups, positions, colors, markers)
end

"""
Wraps the TensorNetwork visualize function, creating (and returning) a new Makie `Figure` and
`Axis`, hiding axis decorations, adding the `DataInspector`, and ensuring the layout is fitted
correctly to the plotted objects.
"""
function FibErrThresh.visualize(
    tn::TensorNetwork,
    groups::Vector{Int},
    positions::Vector{Point2f},
    colors::Vector{Symbol},
    markers::Vector{Symbol},
)
    f = Figure()
    ax = Axis(f[1, 1]; aspect=DataAspect())
    visualize(ax, tn, groups, positions, colors, markers)
    DataInspector(f, range=30)
    hidespines!(ax)
    hidedecorations!(ax)
    resize_to_layout!(f)
    f, ax
end

"""
Wraps the TypedTensorNetwork visualize function, creating (and returning) a new Makie `Figure` and
`Axis`, hiding axis decorations, adding the `DataInspector`, and ensuring the layout is fitted
correctly to the plotted objects.
"""
function FibErrThresh.visualize(ttn::TypedTensorNetwork, position_from_group::Dict{Int, Point2f})
    f = Figure()
    ax = Axis(f[1, 1]; aspect=DataAspect())
    visualize(ax, ttn, position_from_group)
    DataInspector(f, range=30)
    hidespines!(ax)
    hidedecorations!(ax)
    resize_to_layout!(f)
    f, ax
end
