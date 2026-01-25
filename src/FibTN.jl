module FibTN

using SparseArrayKit

struct IndexLabel
    group::Int
    port::Symbol
end

# so that pairs of IndexLabels can be ordered
Base.isless(a::IndexLabel, b::IndexLabel) = a.group < b.group || (a.group == b.group && a.port < b.port)

struct IndexPair
    a::IndexLabel
    b::Indexlabel
    function IndexPair(a, b)
        # check that indices aren't the same
        if a == b error("labels of contracted indices mustn't match") end
        # enforce ordering to prevent duplicates
        if b < a a, b = b, a end
        new(a, b)
    end
end

struct TensorHandle
    group::Int
    indices::Vector{IndexLabel}
    data::SparseArray
    function TensorHandle(group, indices, data)
        if length(indices) != ndims(data) error("number of indices differs from number of array dims") end
        for idx in indices
            if idx.group != group error("tensorhandle group must be the same as each of its indices") end
        end
        new(group, indices)
    end
end

struct TensorNetwork
    tensors::Vector{TensorHandle}
    contractions::Vector{IndexPair}
    tensor_with_index::Dict{IndexLabel, TensorHandle}
    contraction_with_index::Dict{IndexLabel, IndexPair}
    _index_use_count::Dict{IndexLabel, UInt}
    qubits_from_index::Dict{IndexLabel, NTuple{3, Int}}
    TensorNetwork() = new(Array(), Array(), Dict(), Dict())
end

function add_tensor!(tn::TensorNetwork, th::TensorHandle; qubit_index_map=Dict{IndexLabel, NTuple{3, Int}}())
    # check that all indices are unique
    for idx in th.indices
        if haskey(tn.tensor_with_index, idx) error("index $idx already in tensor network") end
    end
    # check that all indices in qubit index map are in this tensor
    for (k, v) in qubit_index_map
        if k.group != th.group error("got qubit index map for index not associated with this tensor") end
    end
    # add tensor and update bookkeeping
    push!(tn.tensors, th)
    for idx in th.indices
        tn.tensor_with_index[idx] = th
        tn._index_use_count[idx] = 1
    end
    merge!(tn.qubits_from_index, qubit_index_map)
end

function add_contraction!(tn::TensorNetwork, ip::IndexPair)
    # check that indices are present
    if !haskey(tn._index_use_count, ip.a) error("index $(ip.a) not found in network") end
    if !haskey(tn._index_use_count, ip.b) error("index $(ip.b) not found in network") end
    # check that indices have not already been contracted
    if tn._index_use_count[ip.a] > 1 error("index $(ip.a) has already been contracted") end
    if tn._index_use_count[ip.b] > 1 error("index $(ip.b) has already been contracted") end
    # add index pair and adjust bookkeeping
    push!(tn.contractions, ip)
    tn.contraction_with_index[ip.a] = ip
    tn.contraction_with_index[ip.b] = ip
    tn._index_use_count[ip.a] += 1
    tn._index_use_count[ip.b] += 1
end

struct TensorNetworkLayoutSpec
    positions::Dict{Int, Tuple{Int, Int}}
    colors::Dict{Int, Symbol}
end

function visualize(tn::TensorNetwork, tnls::TensorNetworkLayoutSpec)
    # for each tensor in tn, plot a node in a graph
    # for each contraction in tn, plot an edge between nodes
    # use ls to specify how to visualize each one
end

struct QubitLattice
    
    states::Dict{Int, Bool}
    QubitLattice() = new(Dict())
end

struct QubitLatticeLayoutSpec
    positions::Dict{Int, Tuple{Int, Int}}
end

function visualize(ql::QubitLattice, qlls::QubitLatticeLayoutSpec)
    # plot the lattice connectivity graph in ql, using qlls for the styling
    # change the color of the graph edges to denote the states |0> and |1>
end

function materialize(tn::TensorNetwork)
    out = Dict{Int, SparseArray}
    for tensor in tn.tensors
        out[tensor.group] = SparseArray(tensor_data(get_type(tensor)))
    end
    out
end

f I wanted to do boundary MPS, I'd want to create some additional datastructure to represent a tensor network in the process of being contracted, where this one actually contains the data for each tensor in each node struct. This is because there's no 'type' to define the composite results of contracting several tensors to get a grid shape.

end # module FibTN
