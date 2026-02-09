module FibTensorNetworks

using ..TensorNetworks
using ..TOBackend
using ..FibTensorTypes

# so we can extend it and avoid name resolution issues
import ..TensorNetworks: add_tensor!

export FibTensorNetwork, add_tensor!, tensordata_from_group

const IL = IndexLabel
const IC = IndexContraction

"""
Convenience abstraction
"""
struct FibTensorNetwork
    tn::TensorNetwork
    tensortype_from_group::Dict{Int, Type{<:AbstractFibTensorType}}
    FibTensorNetwork() = new(TensorNetwork(), Dict())
end

"""
Add
"""
function add_tensor!(ftn::FibTensorNetwork, group::Int, ::Type{T}) where T <: AbstractFibTensorType
    index_labels = [IL(group, p) for p in tensor_ports(T)]
    ftn.tensortype_from_group[group] = T
    add_tensor!(ftn.tn, TensorLabel(group, index_labels))
end

"""Group to tensor data mapping for this FibTensorNetwork"""
function tensordata_from_group(ftn::FibTensorNetwork)
    Dict(g=>tensor_data(tt) for (g, tt) in ftn.tensortype_from_group)
end

end # module FibTensorNetworks
