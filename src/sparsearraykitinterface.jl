module SparseArrayKitInterface

using ..TensorHandles
using SparseArrayKit
using TensorOperations

export SparseArrayKitBackend, TensorHandle, contract, trace

struct SparseArrayKitBackend <: AbstractBackend end

function TensorHandle(::Type{SparseArrayKitBackend}, tensor_data::AbstractArray, ids::Vector{IndexData})
    index_map = Dict(id => num for (num, id) in enumerate(ids))
    tensor = SparseArray(tensor_data)
    TensorHandle{SparseArrayKitBackend, SparseArray, UInt}(tensor, index_map)
end

function contract(th1::TensorHandle{SparseArrayKitBackend}, th2::TensorHandle{SparseArrayKitBackend}, ip::IndexPair)
    # check that there are no duplicate indices between the two tensors
    d = merge(th1.index_map, th2.index_map)
    if length(d) != length(th1.index_map) + length(th2.index_map)
        error("tensor handle indices were not unique")
    end
    # make new tensor
    @tensor new_tensor := # TODO
    # make new index map
    new_index_map = filter(merge(th1.index_map, th2.index_map)) do kv
        kv.first ∉ [ip.a, ip.b]
    end
    TensorHandle{SparseArrayKitBackend, SparseArray, UInt}(new_tensor, new_index_map)
end

function trace(th::TensorHandle{SparseArrayKitBackend}, ip::IndexPair)
    # make new tensor
    @tensor new_tensor := # TODO
    # make new index map
    new_index_map = filter(th.index_map) do kv
        kv.first ∉ [ip.a, ip.b]
    end
    TensorHandle{SparseArrayKitBackend, SparseArray, Index}(new_tensor, new_index_map)
end

end # module SparseArrayKitInterface
