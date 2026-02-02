module QubitLattices

using Graphs
using ..TensorNetworks
using ..IndexTriplets

export QubitLattice, add_index!, get_indices
export idxval2qubitvals, idxvals2qubitvals, qubitvals2idxvals

"""

"""
struct QubitLattice
    qubits_from_index::Dict{IndexLabel, Vector{Int}}
    _node_from_index::Dict{IndexLabel, Int}
    indices_from_qubit::Dict{Int, Vector{IndexLabel}}
    _edge_from_qubit::Dict{Int, Edge}
    _unpaired_qubits::Vector{Int}
    graph::SimpleGraph
    QubitLattice() = new(Dict(), Dict(), Dict(), Dict(), [], SimpleGraph())
end

"""

"""
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

""""""
get_indices(ql::QubitLattice) = keys(ql.qubits_from_index)

"""

"""
function idxval2qubitvals(ql::QubitLattice, idx::IndexLabel, val::Int)
    qvals = split_index(val)
    Dict(q => v for (q, v) in zip(ql.qubits_from_index[idx], qvals))
end

"""

"""
function idxvals2qubitvals(ql::QubitLattice, inds::Vector{IndexLabel}, vals::Vector{Int})
    qubitvals = Dict{Int, Int}()
    for (idx, val) in zip(inds, vals)
        mergewith!(qubitvals, idxval2qubitvals(ql, idx, val)) do x,y
            if x != y error("inconsistent qubit values $x and $y found for qubit") end
            x
        end
    end
    qubitvals
end

"""

"""
function qubitvals2idxvals(ql::QubitLattice, qubitvals::Dict{Int, Int})
    inds = Vector{IndexLabel}
    vals = Vector{Int}
    for idx in indices(ql)
        push!(inds, idx)
        push!(vals, combine_indices(qubitvals[q] for q in ql.qubits_from_index[idx]))
    end
    inds, vals
end

end # module QubitLattices
