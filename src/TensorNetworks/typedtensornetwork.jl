export TensorType, tensor_ports, tensor_data
export tensor_color, tensor_marker
export TypedTensorNetwork, add_tensor!, replace_tensor!, tensordata_from_group

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

"""
Replaces the tensor at `group` with a new tensor of type `NewT`. All ports
of the old type must be present in the new type. By default all contractions
on shared ports are preserved. If `preserve_contractions` is provided, only
contractions on those ports are kept.
"""
function replace_tensor!(ttn::TypedTensorNetwork, group::Int, ::Type{NewT};
                          preserve_contractions::Union{Nothing,Vector{Symbol}}=nothing) where {NewT<:TensorType}
    old_tl = get_tensor(ttn.tn, group)
    new_indices = [IndexLabel(group, p) for p in tensor_ports(NewT)]
    new_tl = TensorLabel(group, new_indices)
    if preserve_contractions !== nothing
        preserve_contractions = [IndexLabel(group, p) for p in preserve_contractions]
    end
    replace_tensor!(ttn.tn, old_tl, new_tl; preserve_contractions)
    ttn.tensortype_from_group[group] = NewT
    nothing
end

"""Returns a group to tensor data mapping for the tensors `ttn`."""
function tensordata_from_group(ttn::TypedTensorNetwork)
    Dict{Int,AbstractArray}(g => tensor_data(tt) for (g, tt) in ttn.tensortype_from_group)
end
