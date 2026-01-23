module TensorNetworks

export TensorNetwork

using ..TensorHandles

struct TensorNetwork
    tensorhandless::Vector{TensorHandle}
    indexcontractions::Vector{IndexContraction}
    lookuptable::Dict{IndexData, TensorHandle}
    TensorNetwork(ths::Vector{TensorHandle}, ics::Vector{IndexContraction}) = begin
        
    end
end

# IndexLabel
# - group::Int
# - port::Symbol

# IndexLevel
# - VIRT or PHYS

# IndexData
# - label::IndexLabel
# - dim::UInt
# - type::IndexLevel

# IndexPair
# - indices::Pair{IndexData}
# validation: must have different labels
# validation: must have matching dims
# validation: must have matching types

# TensorHandle{B <: AbstractBackend, T, I}
# - tensor::T
# - index_map::Dict{IndexData, I}

# TensorNetwork{B <: AbstractBackend}
# - tensorhandles::Vector{TensorHandle{B}}
# - contractions::Vector{IndexPair}
# - tensor_with_index::Dict{IndexData, TensorHandle{B}}
# - index_with_label::Dict{IndexLabel, IndexData}
# validation: every PHYS IndexData must appear exactly once in a th
# validation: every VIRT IndexData must appear exactly once in a th and once in an ip




end # module TensorNetworks
