export TensorType, tensor_ports, tensor_data
export tensor_color, tensor_marker
export TypedTensorNetwork, add_tensor!, tensordata_from_group

"""
An abstract supertype for all user-implemented tensor types. For each tensor type
`CustomTensorType <: TensorType` a user creates, the following methods must also be
implemented:
- `tensor_ports(::Type{CustomTensorType})` -> `::Vector{Symbol}`
- `tensor_data(::Type{CustomTensorType})` -> `<:AbstractArray`
Note that dispatch is done on the type itself rather than on any instances of it.

Optionally, the following methods can be used to specify plotting parameters:
- `tensor_color(::Type{CustomTensorType})` -> `:Symbol`
- `tensor_marker(::Type{CustomTensorType})` -> `:Symbol`
"""
abstract type TensorType end

function tensor_ports end
function tensor_data end

function tensor_color end
function tensor_marker end

"""
A `TypedTensorNetwork` is one where all of its tensors can be assigned a 'type',
such that all tensors with the same type have the same data (and port names).

Each distinct tensor type is user-implemented as a subtype of `TensorType`.
"""
struct TypedTensorNetwork
    tn::TensorNetwork
    tensortype_from_group::Dict{Int,Type{<:TensorType}}
    TypedTensorNetwork() = new(TensorNetwork(), Dict())
end

"""Adds a tensor of the specified `group` and type `T` to `ttn`."""
function add_tensor!(ttn::TypedTensorNetwork, group::Int, ::Type{T}) where {T<:TensorType}
    index_labels = [IndexLabel(group, p) for p in tensor_ports(T)]
    ttn.tensortype_from_group[group] = T
    add_tensor!(ttn.tn, TensorLabel(group, index_labels))
end

"""Returns a group to tensor data mapping for the tensors `ttn`."""
function tensordata_from_group(ttn::TypedTensorNetwork)
    Dict(g => tensor_data(tt) for (g, tt) in ttn.tensortype_from_group)
end
