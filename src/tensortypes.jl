module TensorTypes

export AbstractTensorType
export Reflector, LoopAmplitude, Vertex, Tail, Crossing, Fusion, End, Excitation, DoubledFusion
export index_labels, tensor_data

using ..IndexLabels

### TENSOR TYPES ###

abstract type AbstractTensorType end

struct Reflector        <: AbstractTensorType end
struct LoopAmplitude    <: AbstractTensorType end
struct Vertex           <: AbstractTensorType end
struct Tail             <: AbstractTensorType end
struct Crossing         <: AbstractTensorType end
struct Fusion           <: AbstractTensorType end
struct End              <: AbstractTensorType end
struct Excitation       <: AbstractTensorType end
struct DoubledFusion    <: AbstractTensorType end

### TENSOR INDEX LABELS ###

index_labels(::Type{Reflector}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
]

index_labels(::Type{LoopAmplitude}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
]

index_labels(::Type{Vertex}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
    IndexLabel(:c, VIRT),
    IndexLabel(:q, PHYS),
]

index_labels(::Type{Tail}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
    IndexLabel(:q, PHYS),
]

index_labels(::Type{Crossing}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
    IndexLabel(:c, VIRT),
    IndexLabel(:d, VIRT),
]

index_labels(::Type{Fusion}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
    IndexLabel(:c, VIRT),
]

index_labels(::Type{End}) = [
#TODO
]

index_labels(::Type{Excitation}) = [
#TODO
]

index_labels(::Type{DoubledFusion}) = [
#TODO
]

### TENSOR DATA ###

const _cache = IdDict{DataType,Any}()

function tensor_data(::Type{T}) where {T <: AbstractTensorType}
    key = T
    get!(_cache, key) do
        generate_tensor_data(T)
    end
end

function generate_tensor_data(::Type{Reflector})
    # TODO
end

function generate_tensor_data(::Type{LoopAmplitude})
    # TODO
end

function generate_tensor_data(::Type{Vertex})
    # TODO
end

function generate_tensor_data(::Type{Tail})
    # TODO
end

function generate_tensor_data(::Type{Crossing})
    # TODO
end

function generate_tensor_data(::Type{Fusion})
    # TODO
end

function generate_tensor_data(::Type{End})
    # TODO
end

function generate_tensor_data(::Type{Excitation})
    # TODO
end

function generate_tensor_data(::Type{DoubledFusion})
    # TODO
end

end # module TensorTypes
