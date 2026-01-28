module Visualizer

using Graphs

using GLMakie
using GraphMakie

using ..Indices
using ..TensorNetworks
using ..QubitLattices

export TensorNetworkDisplaySpec, QubitLatticeDisplaySpec, visualize

struct TensorNetworkDisplaySpec
    groups::Vector{Int}
    positions::Vector{NTuple{2, Float64}}
    colors::Vector{Symbol}
    markers::Vector{Symbol}
end

function visualize(tn::TensorNetwork, tnds::TensorNetworkDisplaySpec, ax::Axis)
    edge_endpoints = []
    for c in tn.contractions
        pos1 = tnds.positions[c.a.group]
        pos2 = tnds.positions[c.b.group]
        push!(edge_endpoints, (pos1, pos2))
    end
    segmentsresult = linesegments!(ax, edge_endpoints, color=:gray)
    segmentsresult.inspector_label = (plot, i, idx) -> begin
        ip = tn.contractions[i]
        "$(ip.a.group) $(ip.a.port); $(ip.b.group) $(ip.b.port)"
    end
    
    scatterresult = scatter!(ax, tnds.positions, color=tnds.colors, marker=tnds.markers)
    scatterresult.inspector_label = (plot, i, idx) -> "$(tnds.groups[i])"
    
    segmentsresult, scatterresult
end

struct QubitLatticeDisplaySpec
    index_positions::Dict{IndexLabel, Tuple{Float64, Float64}}
    qubit_colors::Dict{Int, Symbol}
end

function visualize(ql::QubitLattice, qlds::QubitLatticeDisplaySpec, ax::Axis)
    # convert index position dict to node position array
    node_positions = Vector{Tuple{Float64, Float64}}(undef, nv(ql.graph))
    for (idx, node_num) in ql._node_from_index
        node_positions[node_num] = qlds.index_positions[idx]
    end
    # convert qubit colors to edge colors
    edge_colors = Dict{Edge, Symbol}()
    for (qubit, edge) in ql._edge_from_qubit
        edge_colors[edge] = haskey(qlds.qubit_colors, qubit) ? qlds.qubit_colors[qubit] : :gray
    end
    graphplot!(ax, ql.graph, layout=node_positions, edge_color=[edge_colors[e] for e in edges(ql.graph)])
end

end # module Visualizer
