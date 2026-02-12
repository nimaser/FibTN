module FibTensorTypes

# Fibonacci input category data; dim and N, F, R symbols
using TensorKitSectors
const qdim = TensorKitSectors.dim # to avoid name conflict

export AbstractFibTensorType
export Reflector, LoopAmplitude, Vertex, Tail, Crossing, Fusion, End, Excitation, DoubledFusion
export index_data, tensor_data

export Gsymbol, ijk2p, p2ijk, abc2ijkλμνp
include("fibtensortypesutils.jl")

### TENSOR TYPES ###

abstract type AbstractFibTensorType end

struct Reflector        <: AbstractFibTensorType end
struct LoopAmplitude    <: AbstractFibTensorType end
struct Vertex           <: AbstractFibTensorType end
struct Tail             <: AbstractFibTensorType end
struct Crossing         <: AbstractFibTensorType end
struct Fusion           <: AbstractFibTensorType end
struct End              <: AbstractFibTensorType end
struct Excitation       <: AbstractFibTensorType end
struct DoubledFusion    <: AbstractFibTensorType end

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
    [1 0 0 0 0;
     0 0 0 1 0;
     0 0 1 0 0;
     0 1 0 0 0;
     0 0 0 0 1]
end

function generate_tensor_data(::Type{LoopAmplitude})
    [1 0 0 0 0;
     0 1 0 0 0;
     0 0 \phi 0 0;
     0 0 0 \phi 0;
     0 0 0 0 \phi]
end

function generate_tensor_data(::Type{Vertex})
    GSTriangle_data = zeros(Float64, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                local i, j, k, λ, μ, ν, p
                try i, j, k, λ, μ, ν, p = abc2etc(a, b, c) catch; continue end
                GSTriangle_data[a, b, c, p] = Gsymbol(i, j, λ, μ, k, ν) * √√(qdim(i)*qdim(j)*qdim(k))
            end
        end
    end
    GSTriangle_data
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
