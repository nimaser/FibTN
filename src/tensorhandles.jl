module TensorHandles

export AbstractBackend
export VIRT, PHYS, IndexData
export TensorHandle, ContractionSpec, contract

abstract type AbstractBackend end

@enum IndexLevel VIRT PHYS

struct IndexData
    tensor_id::Int
    name::Symbol
    level::IndexLevel
    dim::Int
end

struct ContractionSpec
    pairs::Tuple{Pair{IndexData, IndexData}}
    ContractionSpec(pairs) = begin
        for p in pairs
            if p.first.dim != p.second.dim
                error("dimension mismatch for indices $(p.first) and $(p.second)")
            end
        end
        new(pairs)
    end
end

struct TensorHandle{B <: AbstractBackend, T, I}
    tensor::T
    index_map::Dict{IndexData, I}
end

function contract(::TensorHandle{B}, ::TensorHandle{B}, ::ContractionSpec) where {B <: AbstractBackend}
    error("contract not implemented for backend $(B)")
end

end # module TensorHandles
