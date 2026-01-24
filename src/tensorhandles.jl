module TensorHandles

export AbstractBackend
export IndexLabel, VIRT, PHYS, IndexData, IndexPair
export TensorHandle, contract, trace

abstract type AbstractBackend end

### Index ###

struct IndexLabel
    group::Int
    port::Symbol
end

# so that pairs of IndexLabels are ordered
Base.isless(a::IndexLabel, b::IndexLabel) = a.group < b.group || (a.group == b.group && a.port < b.port)

@enum IndexLevel VIRT PHYS

struct IndexData
    label::IndexLabel
    level::IndexLevel
    dim::UInt
end

struct IndexPair
    a::IndexData
    b::IndexData
    function IndexPair(a, b)
        # check invariants about contracted indices
        if a.dim != b.dim error("dimensions of contracted indices must match") end
        if a.label == b.label error("labels of contracted indices mustn't match") end
        # enforce ordering to prevent duplicates
        if b.label < a.label a, b = b, a end
        new(a, b)
    end
end

### Tensor ###

struct TensorHandle{B <: AbstractBackend, T, I}
    tensor::T
    index_map::Dict{IndexData, I}
end

function contract(::TensorHandle{B}, ::TensorHandle{B}, ::IndexPair) where {B <: AbstractBackend}
    error("contract not implemented for backend $(B)")
end

function trace(::TensorHandle{B}, ::IndexPair) where {B <: AbstractBackend}
    error("trace not implemented for backend $(B)")
end

end # module TensorHandles
