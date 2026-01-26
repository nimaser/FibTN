module FibTensorTypes

using ..IndexTriplets

# Fibonacci input category data; dim and N, F, R symbols
using TensorKitSectors
const qdim = TensorKitSectors.dim # to avoid name conflict

export AbstractFibTensorType
export Reflector, LoopAmplitude, Vertex, Tail, Crossing, Fusion, End, Excitation, DoubledFusion
export index_data, tensor_data

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

### TENSOR INDEX DATA (PORTS) ###

index_data(::Type{Reflector}) = (:a, :b)

index_data(::Type{LoopAmplitude}) = (:a, :b)

index_data(::Type{Vertex}) = (:a, :b, :c, :q)

index_data(::Type{Tail}) = (:a, :b, :q)

index_data(::Type{Crossing}) = (:a, :b, :c, :d)

index_data(::Type{Fusion}) = (:a, :b, :c)

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
