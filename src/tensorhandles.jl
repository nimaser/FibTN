module TensorHandles

export AbstractBackend
export VIRT, PHYS, IndexData
export TensorHandle, ContractionSpec, contract, trace

abstract type AbstractBackend end

@enum IndexLevel VIRT PHYS

struct IndexData
    tensor_id::Int
    name::Symbol
    level::IndexLevel
    dim::Int
end

struct ContractionSpec
    pairs::Vector{Pair{IndexData, IndexData}}
    indices::Set{IndexData}
    ContractionSpec(pairs) = begin
        indices = Set()
        for p in pairs
            # check for dimension mismatch
            if p.first.dim != p.second.dim
                error("dimension mismatch for indices $(p.first) and $(p.second)")
            end
            # check that indices each appear only once
            if p.first ∈ indices
                error("index $(p.first) appears twice")
            end
            if p.second ∈ indices
                error("index $(p.second) appears twice")
            end
            push!(indices, p.first)
            push!(indices, p.second)
        end
        new(pairs, indices)
    end
end

struct TensorHandle{B <: AbstractBackend, T, I}
    tensor::T
    index_map::Dict{IndexData, I}
end

function contract(::TensorHandle{B}, ::TensorHandle{B}, ::ContractionSpec) where {B <: AbstractBackend}
    error("contract not implemented for backend $(B)")
end

function trace(::TensorHandle{B}, ::ContractionSpec) where {B <: AbstractBackend}
    error("trace not implemented for backend $(B)")
end

end # module TensorHandles
