module TensorNetworksVisualizer

using ..TensorNetworks
using GLMakie

export TensorNetworkDisplaySpec, visualize

"""
Holds vectors of group number, position, color, and marker type for the
tensors in a TensorNetwork. All vectors must be the same length, and
should be ordered such that zipping them would result in tuples where
each tuple described one TensorLabel's display.
"""
struct TensorNetworkDisplaySpec
    groups::Vector{Int}
    positions::Vector{NTuple{2, Float64}}
    colors::Vector{Symbol}
    markers::Vector{Symbol}
    function TensorNetworkDisplaySpec(
            g::Vector{Int},
            p::Vector{NTuple{2, Float64}},
            c::Vector{Symbol},
            m::Vector{Symbol},
        )
        lg, lp, lc, lm = length(g), length(p), length(c), length(m)
        if lg != lp != lc != lm
            throw(ArgumentError("lengths of all arguments must be the same, got $(lg, lp, lc, lm)"))
        end
        new(g, p, c, m)
    end
end

"""
Plots a line between the positions of each contracted pair of tensors,
then plots a marker at the position of each tensor. Modifies the
hover tooltip of the GLMakie DataInspector so that hovering over an edge
displays the contracted indices while hovering over a tensor displays
its group number.
"""
function visualize(tn::TensorNetwork, tnds::TensorNetworkDisplaySpec, ax::Axis)
    # get endpoints of edges, and plot linesegments using them
    edge_endpoints = []
    for c in tn.contractions
        pos1 = tnds.positions[c.a.group]
        pos2 = tnds.positions[c.b.group]
        push!(edge_endpoints, (pos1, pos2))
    end
    segmentsresult = linesegments!(ax, edge_endpoints, color=:gray)
    segmentsresult.inspector_label = (plot, i, idx) -> begin
        i = i ÷ 2 # otherwise i counts up by twos per segment
        ic = tn.contractions[i]
        "$(ic.a.group) $(ic.a.port)\n$(ic.b.group) $(ic.b.port)"
    end
    # scatterplot to represent tensors
    scatterresult = scatter!(ax, tnds.positions, color=tnds.colors, marker=tnds.markers, markersize=40)
    scatterresult.inspector_label = (plot, i, idx) -> "$(tnds.groups[i])"
    # compute limits based on the axis content
    autolimits!(ax)
    # return the segments and scatterplot
    segmentsresult, scatterresult
end

end # module TensorNetworksVisualizer
