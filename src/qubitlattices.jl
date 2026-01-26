module QubitLattices

using Graphs
using ..Indices
using ..IndexTriplets

export QubitLattice, add_index!

struct QubitLattice
    qubits_from_index::Dict{IndexLabel, Vector{Int}}
    _node_from_index::Dict{IndexLabel, Int}
    indices_from_qubit::Dict{Int, Vector{IndexLabel}}
    _edge_from_qubit::Dict{Int, SimpleEdge}
    _unpaired_qubits::Vector{Int}
    graph::SimpleGraph
    QubitLattice() = new(Dict(), Dict(), Dict(), Dict(), [], SimpleGraph())
end

function add_index!(ql::QubitLattice, i::IndexLabel, qubits::Vector{Int})
    # register qubits to index
    ql.qubits_from_index[i] = qubits
    # add index to graph
    add_vertex!(ql.graph)
    ql._node_from_index[i] = nv(ql.graph)
    # register indices per qubit, adding edges if an added qubit completes a match
    for qubit in qubits
        if haskey(ql.indices_from_qubit, qubit)
            # there is already another index for this qubit
            # we are assuming that each qubit can be matched to at most two indices
            otherindexnode = ql._node_from_index[ql.indices_from_qubit[qubit][1]]
            add_edge!(ql.graph, nv(ql.graph), otherindexnode)
            _edge_from_qubit[qubit] = SimpleEdge(nv(ql.graph), otherindexnode)
            push!(ql.indices_from_qubit[qubit], i)
            filter!(!=(qubit), ql._unpaired_qubits)
        else
            # this is the first index for this qubit
            ql.indices_from_qubit[qubit] = [i]
            push!(ql._unpaired_qubits, qubit)
        end
    end
end

# TODO function to get qubit to value mapping

end # module QubitLattices
