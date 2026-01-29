module TensorNetworks

using ..Indices

export TensorLabel, TensorNetwork, add_tensor!, add_contraction!
export indices, find_indices, find_indices_by_group, find_indices_by_port, find_index

struct TensorLabel
    group::Int
    indices::Vector{IndexLabel}
    function TensorLabel(group, indices)
        for idx in indices
            if idx.group != group error("TensorLabel group must be the same as each of its indices") end
        end
        new(group, indices)
    end
end

struct TensorNetwork
    tensors::Vector{TensorLabel}
    contractions::Vector{IndexPair}
    tensor_with_index::Dict{IndexLabel, TensorLabel}
    contraction_with_index::Dict{IndexLabel, IndexPair}
    _index_use_count::Dict{IndexLabel, UInt}
    TensorNetwork() = new([], [], Dict(), Dict(), Dict())
end

function add_tensor!(tn::TensorNetwork, tl::TensorLabel)
    # check that all indices are unique
    for idx in tl.indices
        if haskey(tn._index_use_count, idx) error("index $idx already in tensor network") end
    end
    # add tensor and update bookkeeping
    push!(tn.tensors, tl)
    for idx in tl.indices
        tn.tensor_with_index[idx] = tl
        tn._index_use_count[idx] = 1
    end
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

tensor_with_group(tn::TensorNetwork, group::Int)
    for t in tn.tensors
        if t.group == group return t end
    end
    nothing
end

indices(tn::TensorNetwork) = keys(tn.tensor_with_index)

function find_indices(f::Function, tn::TensorNetwork)
    matches = IndexLabel[]
    for idx in indices(tn)
        if f(idx) push!(matches, idx) end
    end
    matches
end

find_indices_by_group(tn::TensorNetwork, group::Int) = find_indices(idx -> idx.group == group, tn)
find_indices_by_port(tn::TensorNetwork, port::Symbol) = find_indices(idx -> idx.port == port, tn)

function find_index(tn::TensorNetwork, group::Int, port::Symbol)
    matches = find_indices(idx -> idx.group == group && idx.port == port, tn)
    if length(matches) != 1 error("no index found with group $group and port $port") end
    only(matches)
end

end # module TensorNetworks
