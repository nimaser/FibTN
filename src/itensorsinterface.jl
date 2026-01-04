module ITensorsInterface

using ..TensorHandles
using ITensors

struct ITensorsBackend <: AbstractBackend end

function TensorHandle(::Type{ITensorsBackend}, tensor_data::AbstractArray, ids::Vector{IndexData})
    itensorindices = [Index(id.dim) for id in ids]
    index_map = Dict(id => iti for (id, iti) in zip(ids, itensorindices))
    tensor = ITensor(tensor_data, itensorindices...)
    TensorHandle{ITensorsBackend}(tensor, index_map)
end
    
function contract(th1::TensorHandle{ITensorsBackend}, th2::TensorHandle{ITensorsBackend}, cs::ContractionSpec)
    contractedindices = []
    new_tensor = th1.tensor
    for p in cs 
        new_tensor = new_tensor * δ(th1.index_map[p.first], th2.index_map[p.second])
        push!(contractedindices, p.first, p.second)
    end
    new_tensor = new_tensor * th2.tensor
    
    new_index_map = filter(merge(th1.index_map, th2.index_map)) do kv
        kv.first ∉ contractedindices
    end
    TensorHandle{ITensorsBackend}(new_tensor, new_index_map)
end

function trace(th::TensorHandle{ITensorsBackend}, cs::ContractionSpec)
    contract(th, TensorHandle(ITensorsBackend, [], Dict()), cs)
end

end # module ITensorsInterface
