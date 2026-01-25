struct TensorLabel
    group::Int
    indices::Vector{IndexLabel}
    function TensorLabel(group, indices, data)
        for idx in indices
            if idx.group != group error("tensorlabel group must be the same as each of its indices") end
        end
        new(group, indices)
    end
end

struct FibTensorNetwork
    tensors::Vector{TensorLabel}
    contractions::Vector{IndexPair}
    tensor_with_index::Dict{IndexLabel, TensorLabel}
    contraction_with_index::Dict{IndexLabel, IndexPair}
    _index_use_count::Dict{IndexLabel, UInt}
    qubits_from_index::Dict{IndexLabel, NTuple{3, Int}}
    FibTensorNetwork() = new(Array(), Array(), Dict(), Dict())
end

function add_tensor!(tn::TensorNetwork, tl::TensorLabel; qubit_index_map=Dict{IndexLabel, NTuple{3, Int}}())
    # check that all indices are unique
    for idx in tl.indices
        if haskey(tn.tensor_with_index, idx) error("index $idx already in tensor network") end
    end
    # check that all indices in qubit index map are in this tensor
    for (k, v) in qubit_index_map
        if k.group != tl.group error("got qubit index map for index not associated with this tensor") end
    end
    # add tensor and update bookkeeping
    push!(tn.tensors, tl.group, th)
    for idx in tl.indices
        tn.tensor_with_index[idx] = tl
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

struct FibTensorNetworkLayoutSpec
    positions::Dict{Int, Tuple{Int, Int}}
    colors::Dict{Int, Symbol}
end

function visualize(ftn::FibTensorNetwork, ftnls::FibTensorNetworkLayoutSpec)
    # for each tensor in tn, plot a node in a graph
    # for each contraction in tn, plot an edge between nodes
    # use ls to specify how to visualize each one
end
