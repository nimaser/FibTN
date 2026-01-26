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
    node_positions::Dict{Int, Tuple{Int, Int}}
    edge_colors::Dict{Int, Symbol}
end

function visualize(ql::QubitLattice, qlds::QubitLatticeDisplaySpec, ax::Axis)
    pos = [qlds.node_positions[k] for k in sort!(collect(keys(qlds.node_positions)))]
    graphplot!(ax, ql.graph, layout=pos, )
    # plot the lattice connectivity graph in ql, using qlls for the styling and to
    # change the color of the graph edges to denote the qubit states |0> and |1>
end

end # module Visualizer
