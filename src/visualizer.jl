module Visualizer

using GLMakie
using GraphMakie

using ..TensorNetworks
using ..QubitLattices

struct TensorNetworkDisplaySpec
    groups::Vector{Int}
    positions::Vector{NTuple{2, Float64}}
    colors::Vector{Symbol}
    markers::Vector{Symbol}
end

function visualize(tn::TensorNetwork, tnds::TensorNetworkDisplaySpec, ax::Axis)
    scatter!(ax, tnds.positions, color=tnds.colors, marker=tnds.markers)
    for c in tn.contractions
        pos1 = tnds.positions[c.a.group]
        pos2 = tnds.positions[c.b.group]
        lines!(ax, [pos1, pos2])
    end
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
    edge_colors = Dict{SimpleEdge, Symbol}()
    for (qubit, edge) in ql._edge_from_qubit
        edge_colors[edge] = haskey(qlds.qubit_colors, qubit) ? qlds.qubit_colors[qubit] : :gray
    end
    graphplot!(ax, ql.graph, layout=node_positions, edge_color=[edge_colors[e] for e in edges(ql.graph)])
end

end # module Visualizer
