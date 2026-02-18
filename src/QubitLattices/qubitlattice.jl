using ..TensorNetworks
using ..IndexTriplets
import ..TensorNetworks: get_indices # merge method tables

export QubitLattice, get_indices, get_qubits, add_index!
export idxval2qubitvals, idxvals2qubitvals, qubitvals2idxvals
export get_states_and_amps

"""
A `QubitLattice` handles the conversion between values of physical
indices in the PEPS tensor network and the values of physical qubits
in the lattice. Each physical index codes for three qubits in total.
Qubits are identified by an `Int`.

The qubit lattice is a graph where each node is a physical index and
each edge between nodes is a qubit. Thus most qubits are coded for by
two physical indices. Some qubits may be 'unpaired', meaning they are
only coded for by one index. Furthermore, some unpaired qubits may
always be trivial, and therefore of no interest when e.g. plotting.
All such qubits can be labeled 0, which will cause them to effectively
be ignored.

Only the 0 qubit can be 'coded for' by more than two indices, and if the
0 qubit is ever nontrivial, an error is raised. In addition, if two
indices 'disagree' on a qubit they both code for, an error will also
be raised.
"""
struct QubitLattice
    qubits_from_index::Dict{IndexLabel,NTuple{3,Int}}
    indices_from_qubit::Dict{Int,Vector{IndexLabel}}
    _unpaired_qubits::Vector{Int}
    QubitLattice() = new(Dict(), Dict(), [])
end

"""Gets a `KeySet` of all indices added to `ql`."""
get_indices(ql::QubitLattice) = keys(ql.qubits_from_index)

"""Gets a `Vector{IndexLabel}` of all indices coding for qubit `q`."""
get_indices(ql::QubitLattice, q::Int) = ql.indices_from_qubit[q]

"""Gets a `KeySet` of all qubits added to `ql`."""
get_qubits(ql::QubitLattice) = keys(ql.indices_from_qubit)

"""Gets the `NTuple{3,Int}` of qubits coded for by `il`."""
get_qubits(ql::QubitLattice, il::IndexLabel) = ql.qubits_from_index[il]

"""
Sets `idx` as encoding for the three qubits in `qubits` in `ql`. The encoding
order matches position: if `split_index` called on a value for `idx` gives `vals`,
then `qubits[p]` takes the value `vals[p]` for each position `p`.
"""
function add_index!(ql::QubitLattice, idx::IndexLabel, qubits::NTuple{3,Int})
    # check that there are no duplicate qubits or indices
    if haskey(ql.qubits_from_index, idx)
        error("index $idx already mapped to qubits")
    end
    if length(Set(qubits)) != length(qubits)
        error("duplicate qubits provided")
    end

    # register qubits to index
    ql.qubits_from_index[idx] = qubits
    # register indices per qubit
    for qubit in qubits
        if haskey(ql.indices_from_qubit, qubit)
            # there is already another index for this qubit
            push!(ql.indices_from_qubit[qubit], idx)
            # if the qubit is 0, we don't record it in unpaired
            if qubit != 0
                filter!(!=(qubit), ql._unpaired_qubits)
            end
        else
            # this is the first index for this qubit
            ql.indices_from_qubit[qubit] = [idx]
            # if the qubit is 0, we don't record it in unpaired
            if qubit != 0
                push!(ql._unpaired_qubits, qubit)
            end
        end
    end
end

"""
Returns a dictionary mapping each qubit id to its value for the qubits
coded for by `idx`, given that `idx` takes the value `val`. If the 0
qubit takes a nontrivial value, an error is thrown.
"""
function idxval2qubitvals(ql::QubitLattice, idx::IndexLabel, val::Int)
    qvals = split_index(val)
    qvals = Dict(q => v for (q, v) in zip(ql.qubits_from_index[idx], qvals))
    !haskey(qvals, 0) || qvals[0] == 0 || throw(ArgumentError("qubit 0 nontrivial"))
    qvals
end

"""
Returns a dictionary mapping all qubits in `ql` to their values, given the list
of indices `inds` taking values `vals`, which must be ordered so that `inds[i]`
takes the value `vals[i]`. If inconsistencies arise, an error is thrown. If the
0 qubit takes a nontrivial value, an error is thrown.
"""
function idxvals2qubitvals(ql::QubitLattice, inds::Vector{IndexLabel}, vals::Vector{Int})
    qubitvals = Dict{Int,Int}()
    for (idx, val) in zip(inds, vals)
        newqubitvals = idxval2qubitvals(ql, idx, val)
        # merge the dictionaries
        for (k, v) in newqubitvals
            # check that key isn't already there with an inconsistent value
            !haskey(qubitvals, k) || qubitvals[k] == v || error("inconsistent vals for qubit $k")
            # merge
            qubitvals[k] = v
        end
    end
    qubitvals
end

"""
Given `qubitvals` mapping from qubit id to qubit value, returns corresponding lists
of indices in `ql` and their values, i.e. `vals[i]` is the value that index `inds[i]`
takes, where inds, vals are the returned lists. If the 0 qubit is mapped to a nontrivial
value, an error is thrown.

`inds` can be provided to fix the ordering of the returned values.
"""
function qubitvals2idxvals(ql::QubitLattice, qubitvals::Dict{Int,Int}; inds::Vector{IndexLabel}=get_indices(ql))
    !haskey(qubitvals, 0) || qubitvals[0] == 0 || throw(ArgumentError("qubit 0 nontrivial"))
    vals = Vector{Int}()
    for idx in inds
        t = ql.qubits_from_index[idx]
        push!(vals, combine_qubits(qubitvals[t[1]], qubitvals[t[2]], qubitvals[t[3]]))
    end
    inds, vals
end
