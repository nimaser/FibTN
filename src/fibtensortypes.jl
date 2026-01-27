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

index_ports(::Type{Vertex}) = (:a, :b, :c, :p)

index_ports(::Type{Tail}) = (:a, :b, :p)

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
    arr = zeros(Float64, 5, 5)
    for a in 1:5
        for b in 1:5
            μ, i, ν = split_index(a)
            ν2, i2, μ2 = split_index(b)
            if μ != μ2 || ν != ν2 || i != i2 continue end
            arr[a, b] = 1
        end
    end
    arr
end

function generate_tensor_data(::Type{LoopAmplitude})
    arr = zeros(Float64, 5, 5)
    for a in 1:5
        for b in 1:5
            μ, i, ν = split_index(a)
            ν2, i2, μ2 = split_index(b)
            if μ != μ2 || ν != ν2 || i != i2 continue end
            arr[a, b] = μ == 0 ? 1 : ϕ
        end
    end
    arr
end

function generate_tensor_data(::Type{Vertex})
    arr = zeros(Float64, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                # break out indices
                μ, i, ν = split_index(a)
                ν2, j, λ = split_index(b)
                λ2, k, μ2 = split_index(c)
                # abort inconsistent cases, leaving them 0
                if μ != μ2 || ν != ν2 || λ != λ2 continue end
                # get p, convert integers to anyons, and calculate the entry
                p = combine_indices(i, j, k)
                i, j, k, λ, μ, ν = anyon.([i, j, k, λ, μ, ν])
                arr[a, b, c, p] = Gsymbol(i, j, λ, μ, k, ν) * √√(qdim(i)*qdim(j)*qdim(k))
            end
        end
    end
    arr
end

function generate_tensor_data(::Type{Tail})
    arr = zeros(Float64, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            μ, x, ν = split_index(a)
            ν2, x2, μ2 = split_index(b)
            if μ != μ2 || ν != ν2 || x != x2 continue end
            p = combine_indices(x, 0, x2)
            arr[a, b, p] = 1
        end
    end
    arr
end

function generate_tensor_data(::Type{Crossing})
    arr = zeros(Float64, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                for d in 1:5
                    ξ, k, λ = split_index(a)
                    λ2, s, μ = split_index(b)
                    μ2, k2, ν = split_index(c)
                    ν2, s2, ξ2 = split_index(d)
                    if ξ != ξ2 || λ != λ2 || μ != μ2 || ν != ν2 || k != k2 || s != s2 continue end
                    μ, λ, ξ, ν, s, k = anyon.([μ, λ, ξ, ν, s, k])
                    arr[a, b, c, d] = Gsymbol(μ, λ, ξ, ν, s, k)
                end
            end
        end
    end
    arr
end

function generate_tensor_data(::Type{Fusion})
    arr = zeros(Float64, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                μ, s, ν = split_index(a)
                ν2, t, λ = split_index(b)
                λ2, u, μ2 = split_index(c)
                if λ != λ2 || μ != μ2 || ν != ν2 continue end
                s, t, λ, ν, u, μ = anyon.([s, t, λ, ν, u, μ])
                arr[a, b, c] = Gsymbol(s, t, λ, ν, u, μ) * √√(qdim(s)*qdim(t)*qdim(u))
            end
        end
    end
    arr
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
