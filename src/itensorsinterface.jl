module ITensorsInterface

using ..TensorHandles
using ITensors

export ITensorsBackend, TensorHandle, contract, trace

struct ITensorsBackend <: AbstractBackend end

function TensorHandle(::Type{ITensorsBackend}, tensor_data::AbstractArray, ids::Vector{IndexData})
    itensorindices = [Index(id.dim) for id in ids]
    index_map = Dict(id => iti for (id, iti) in zip(ids, itensorindices))
    tensor = ITensor(tensor_data, itensorindices...)
    TensorHandle{ITensorsBackend, ITensor, Index}(tensor, index_map)
end

function contract(th1::TensorHandle{ITensorsBackend}, th2::TensorHandle{ITensorsBackend}, cs::ContractionSpec)
    # check that there are no duplicate indices between the two tensors
    d = merge(th1.index_map, th2.index_map)
    if length(d) != length(th1.index_map) + length(th2.index_map)
        error("tensor handle indices were not unique")
    end
    # make new tensor
    new_tensor = th1.tensor
    for p in cs.pairs 
        new_tensor = new_tensor * δ(d[p.first], d[p.second])
    end
    new_tensor = new_tensor * th2.tensor
    # make new index map
    new_index_map = filter(merge(th1.index_map, th2.index_map)) do kv
        kv.first ∉ cs.indices
    end
    TensorHandle{ITensorsBackend, ITensor, Index}(new_tensor, new_index_map)
end

function trace(th::TensorHandle{ITensorsBackend}, cs::ContractionSpec)
    # make new tensor
    new_tensor = th.tensor
    for p in cs.pairs
        new_tensor = new_tensor * δ(th.index_map[p.first], th.index_map[p.second])
    end
    # make new index map
    new_index_map = filter(th.index_map) do kv
        kv.first ∉ cs.indices
    end
    TensorHandle{ITensorsBackend, ITensor, Index}(new_tensor, new_index_map)
end

end # module ITensorsInterface
