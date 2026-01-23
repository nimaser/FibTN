module TensorNetworks

export TensorNetwork, add_tensor!, add_contraction!

using ..TensorHandles

mutable struct TensorNetwork{B <: AbstractBackend}
    tensors::Vector{TensorHandle{B}}
    contractions::Vector{IndexPair}
    tensor_with_index::Dict{IndexLabel, TensorHandle{B}}
    index_with_label::Dict{IndexLabel, IndexData}
    _index_use_count::Dict{IndexLabel, UInt}
    TensorNetwork{B}() where B <: AbstractBackend = new(Vector(), Vector(), Dict(), Dict(), Dict())
end

function add_tensor!(tn::TensorNetwork{B}, th::TensorHandle{B})
    # check that all indices are unique to this network
    for idxdat in keys(th.index_map)
        if haskey(tn.tensor_with_index, idxdat.label) error("index $idxdat already in tensor network") end
    end
    # add the tensor and its indices to the bookkeeping datastructures
    push!(tn.tensors, th)
    for idxdat in keys(th.index_map)
        tn.tensor_with_index[idxdat.label] = th
        tn.index_with_label[idxdat.label] = idxdat
        _index_use_count[idxdat.label] = 1
    end
end

function add_contraction!(tn::TensorNetwork{B}, ip::IndexPair)
    # check that indices are present
    if !haskey(tn._index_use_count, ip.a.label) error("index $(ip.a) not found in network") end
    if !haskey(tn._index_use_count, ip.b.label) error("index $(ip.b) not found in network") end
    # check that indices have not already been contracted
    if tn._index_use_count[ip.a.label] > 1 error("index $(ip.a) has already been contracted") end
    if tn._index_use_count[ip.b.label] > 1 error("index $(ip.b) has already been contracted") end
    # add index pair and adjust bookkeeping
    push!(tn.contractions, ip)
    tn._index_use_count[ip.a.label] += 1
    tn._index_use_count[ip.b.label] += 1
end

end # module TensorNetworks
