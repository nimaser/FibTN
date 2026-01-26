module FibTensorTypes

using ..IndexTriplets
using ..Indices

# Fibonacci input category data; dim and N, F, R symbols
using TensorKitSectors
const qdim = TensorKitSectors.dim # to avoid name conflict

export AbstractFibTensorType
export Reflector, LoopAmplitude, Vertex, Tail, Crossing, Fusion, End, Excitation, DoubledFusion
export index_labels, tensor_data

### UTILS ###

const ϕ = (1 + √5) / 2

anyon(a::Int) = a == 0 ? FibonacciAnyon(:I) : a == 1 ? FibonacciAnyon(:τ) : error("a must be 0 or 1")
#int(a::FibonacciAnyon) = a == FibonacciAnyon(:I) ? 0 : a == FibonacciAnyon(:τ) ? 1 : error("a must be FibonacciAnyon(:I or :τ)")

function Gsymbol(
        a::FibonacciAnyon, b::FibonacciAnyon, c::FibonacciAnyon,
        d::FibonacciAnyon, e::FibonacciAnyon, f::FibonacciAnyon
    )
    Fsymbol(a, b, c, d, e, f) / √(qdim(e)*qdim(f))
end

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

### TENSOR INDEX DATA ###

index_labels(::Type{T}, group::Int) where T <: AbstractFibTensorType = [IndexLabel(group, p) for p in index_ports(T)]

index_ports(::Type{Reflector}) = (:a, :b)

index_ports(::Type{LoopAmplitude}) = (:a, :b)

index_ports(::Type{Vertex}) = (:a, :b, :c, :q)

index_ports(::Type{Tail}) = (:a, :b, :q)

index_ports(::Type{Crossing}) = (:a, :b, :c, :d)

index_ports(::Type{Fusion}) = (:a, :b, :c)

index_ports(::Type{End}) = (
#TODO
)

index_ports(::Type{Excitation}) = (
#TODO
)

index_ports(::Type{DoubledFusion}) = (
#TODO
)

### TENSOR DATA ###

const _cache = IdDict{DataType,Any}()

function tensor_data(::Type{T}) where {T <: AbstractFibTensorType}
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
     0 0 ϕ 0 0;
     0 0 0 ϕ 0;
     0 0 0 0 ϕ]
end

function generate_tensor_data(::Type{Vertex})
    GSTriangle_data = zeros(Float64, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                local i, j, k, λ, μ, ν, p
                try i, j, k, λ, μ, ν, p = abc2ijkλμνp(a, b, c) catch; continue end
                i, j, k, λ, μ, ν = anyon.([i, j, k, λ, μ, ν])
                GSTriangle_data[a, b, c, p] = Gsymbol(i, j, λ, μ, k, ν) * √√(qdim(i)*qdim(j)*qdim(k))
            end
        end
    end
    GSTriangle_data
end

function generate_tensor_data(::Type{Tail})
    [1 0 0 0 0;
     0 0 0 0 0;
     0 0 1 0 0;
     0 0 0 0 0;
     0 0 0 0 0]
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
