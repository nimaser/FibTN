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
        i = i ÷ 2 # because i is the index of the end point
        ip = tn.contractions[i]
        "$(ip.a.group) $(ip.a.port)\n$(ip.b.group) $(ip.b.port)"
    end
    
    scatterresult = scatter!(ax, tnds.positions, color=tnds.colors, marker=tnds.markers, markersize=40)
    scatterresult.inspector_label = (plot, i, idx) -> "$(tnds.groups[i])"
    
    segmentsresult, scatterresult
end

struct QubitLatticeDisplaySpec
    index_positions::Dict{IndexLabel, Tuple{Float64, Float64}}
    qubit_colors::Dict{Int, Symbol}
    unpaired_qubit_length::Real
end

function visualize(ql::QubitLattice, qlds::QubitLatticeDisplaySpec, ax::Axis)
    # convert index_positions to map from node to position
    index_positions = Dict(ql._node_from_index[i] => p for (i, p) in qlds.index_positions)
    
    # for each unpaired qubit, make a dummy index and node for display
    g = copy(ql.graph) # copy graph so the original is unaltered
    edge_from_qubit = copy(ql._edge_from_qubit)
    for q in ql._unpaired_qubits
        add_vertex!(g)
        # get the index associated with the qubit, then its node and pos
        idx = only(ql.indices_from_qubit[q])
        n = ql._node_from_index[idx]
        pos = index_positions[n]
        # set pos of the new node to be slightly to the +x of the index
        index_positions[nv(g)] = (pos[1] + qlds.unpaired_qubit_length, pos[2])
        # add a new edge to the graph and map to it from the qubit
        add_edge!(g, n, nv(g))
        edge_from_qubit[q] = Edge(n, nv(g))
    end

    # convert node position dict to node position array
    node_positions = Vector{Tuple{Float64, Float64}}(undef, nv(g))
    for n in 1:nv(g)
        node_positions[n] = index_positions[n]
    end
    
    # convert qubit colors to edge colors
    edge_colors = Dict{Edge, Symbol}()
    for (qubit, edge) in edge_from_qubit
        edge_colors[edge] = haskey(qlds.qubit_colors, qubit) ? qlds.qubit_colors[qubit] : :gray
    end

    graphplot!(ax, g, layout=node_positions, edge_color=[edge_colors[e] for e in edges(g)])
end

end # module Visualizer
