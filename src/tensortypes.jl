module TensorTypes

export AbstractTensorType
export Reflector, LoopAmplitude, Vertex, Tail, Crossing, Fusion, End, Excitation, DoubledFusion
export index_data, tensor_data

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

### TENSOR INDEX DATA (OMITTING TENSOR ID) ###

index_data(::Type{Reflector}) = (
    Tuple(:a, VIRT, 5),
    Tuple(:b, VIRT, 5),
)

index_data(::Type{LoopAmplitude}) = (
    Tuple(:a, VIRT, 5),
    Tuple(:b, VIRT, 5),
)

index_data(::Type{Vertex}) = (
    Tuple(:a, VIRT, 5),
    Tuple(:b, VIRT, 5),
    Tuple(:c, VIRT, 5),
    Tuple(:q, PHYS, 5),
)

index_data(::Type{Tail}) = (
    Tuple(:a, VIRT, 5),
    Tuple(:b, VIRT, 5),
    Tuple(:q, PHYS, 5),
)

index_data(::Type{Crossing}) = (
    Tuple(:a, VIRT, 5),
    Tuple(:b, VIRT, 5),
    Tuple(:c, VIRT, 5),
    Tuple(:d, VIRT, 5),
)

index_data(::Type{Fusion}) = (
    Tuple(:a, VIRT, 5),
    Tuple(:b, VIRT, 5),
    Tuple(:c, VIRT, 5),
)

index_data(::Type{End}) = (
#TODO
)

index_data(::Type{Excitation}) = (
#TODO
)

index_data(::Type{DoubledFusion}) = (
#TODO
)

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
