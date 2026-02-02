module QubitLatticesVisualizer

using ..QubitLattices
using Graphs
using GLMakie
using GraphMakie

export QubitLatticeDisplaySpec, visualize

"""
Holds mappings from physical IndexLabel to position and from qubit
number to color. Also has a parameter controlling how long tails
(which are only associated with one index, and thus have a free node)
should be.
"""
struct QubitLatticeDisplaySpec
    position_from_index::Dict{IndexLabel, Tuple{Float64, Float64}}
    color_from_qubit::Dict{Int, Symbol}
    tail_length::Real
end

"""
Plots a 
"""
function visualize(ql::QubitLattice, qlds::QubitLatticeDisplaySpec, ax::Axis)
    # convert index_positions to map from node to position
    position_from_node = Dict(ql._node_from_index[i] => qlds.position_from_index[i] for i in keys(ql.qubits_from_index))
    
    # for each unpaired qubit, make a dummy index and node for display
    g = copy(ql.graph) # copy graph so the original is unaltered
    edge_from_qubit = copy(ql._edge_from_qubit)
    for q in ql._unpaired_qubits
        add_vertex!(g)
        # get the index associated with the qubit, then its node and pos
        idx = only(ql.indices_from_qubit[q])
        n = ql._node_from_index[idx]
        pos = position_from_node[n]
        # set pos of the new node to be slightly to the +x of the index
        position_from_node[nv(g)] = (pos[1] + qlds.tail_length, pos[2])
        # add a new edge to the graph and map to it from the qubit
        add_edge!(g, n, nv(g))
        edge_from_qubit[q] = Edge(n, nv(g))
    end

    # convert node position dict to node position array
    node_positions = Vector{Tuple{Float64, Float64}}(undef, nv(g))
    for n in 1:nv(g)
        node_positions[n] = position_from_node[n]
    end
    
    # convert qubit colors to edge colors
    color_from_edge = Dict{Edge, Symbol}()
    for (qubit, edge) in edge_from_qubit
        color_from_edge[edge] = get(qlds.color_from_qubit, qubit) do; :gray end
    end

    graphplot!(ax, g, layout=node_positions, edge_color=[color_from_edge[e] for e in edges(g)])
    autolimits!(ax)
end

function state_explorer(ql::QubitLattice, qlds::QubitLatticeDisplaySpec, ax::Axis)
    # convert index_positions to map from node to position
    position_from_node = Dict(ql._node_from_index[i] => qlds.position_from_index[i] for i in keys(ql.qubits_from_index))
    
    # for each unpaired qubit, make a dummy index and node for display
    g = copy(ql.graph) # copy graph so the original is unaltered
    edge_from_qubit = copy(ql._edge_from_qubit)
    for q in ql._unpaired_qubits
        add_vertex!(g)
        # get the index associated with the qubit, then its node and pos
        idx = only(ql.indices_from_qubit[q])
        n = ql._node_from_index[idx]
        pos = position_from_node[n]
        # set pos of the new node to be slightly to the +x of the index
        position_from_node[nv(g)] = (pos[1] + qlds.tail_length, pos[2])
        # add a new edge to the graph and map to it from the qubit
        add_edge!(g, n, nv(g))
        edge_from_qubit[q] = Edge(n, nv(g))
    end

    # convert node position dict to node position array
    node_positions = Vector{Tuple{Float64, Float64}}(undef, nv(g))
    for n in 1:nv(g)
        node_positions[n] = position_from_node[n]
    end
    
    # convert qubit colors to edge colors
    color_from_edge = Dict{Edge, Symbol}()
    for (qubit, edge) in edge_from_qubit
        color_from_edge[edge] = get(qlds.color_from_qubit, qubit) do; :gray end
    end

    graphplot!(ax, g, layout=node_positions, edge_color=[color_from_edge[e] for e in edges(g)])
    autolimits!(ax)
end

end # module QubitLatticesVisualizer
