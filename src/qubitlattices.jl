module QubitLattices

using Graphs
using ..Indices
using ..IndexTriplets

export QubitLattice, add_index!
export get_qubit_states, get_lattice_state, get_state

struct QubitLattice
    qubits_from_index::Dict{IndexLabel, Vector{Int}}
    _node_from_index::Dict{IndexLabel, Int}
    indices_from_qubit::Dict{Int, Vector{IndexLabel}}
    _edge_from_qubit::Dict{Int, Edge}
    _unpaired_qubits::Vector{Int}
    graph::SimpleGraph
    QubitLattice() = new(Dict(), Dict(), Dict(), Dict(), [], SimpleGraph())
end

function add_index!(ql::QubitLattice, idx::IndexLabel, qubits::Vector{Int})
    # check that there are no duplicate qubits or indices
    if haskey(ql.qubits_from_index, idx) error("index $idx already mapped to qubits") end
    if length(Set(qubits)) != length(qubits) error("duplicate qubits provided") end

    # register qubits to index
    ql.qubits_from_index[idx] = qubits
    # add index to graph
    add_vertex!(ql.graph)
    ql._node_from_index[idx] = nv(ql.graph)
    # register indices per qubit, adding edges if an added qubit completes a match
    for qubit in qubits
        if haskey(ql.indices_from_qubit, qubit)
            # there is already another index for this qubit
            # we are assuming that each qubit can be matched to at most two indices
            otherindexnode = ql._node_from_index[only(ql.indices_from_qubit[qubit])]
            add_edge!(ql.graph, nv(ql.graph), otherindexnode)
            ql._edge_from_qubit[qubit] = Edge(otherindexnode, nv(ql.graph))
            push!(ql.indices_from_qubit[qubit], idx)
            filter!(!=(qubit), ql._unpaired_qubits)
        else
            # this is the first index for this qubit
            ql.indices_from_qubit[qubit] = [idx]
            push!(ql._unpaired_qubits, qubit)
        end
    end
end

indices(ql::QubitLattice) = keys(ql.qubits_from_index)

function get_qubit_states(ql::QubitLattice, idx::IndexLabel, idxval::Int)
    qvals = split_index(idxval)
    Dict(q => v for (q, v) in zip(ql.qubits_from_index[idx], qvals))
end

function get_lattice_state(ql::QubitLattice, inds::Vector{IndexLabel}, vals::Vector{Int})
    lattice_state = Dict{Int, Int}()
    for (idx, idxval) in zip(inds, vals)
        mergewith!(lattice_state, get_qubit_states(ql, idx, idxval)) do x,y
            if x != y error("inconsistent qubit values $x and $y found for qubit") end
            x
        end
    end
    lattice_state
end

end # module QubitLattices
