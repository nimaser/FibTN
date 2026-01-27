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
    # break out indices
    function _abc2ijkλμνp(a::Int, b::Int, c::Int)
        μ, i, ν = split_index(a)
        ν2, j, λ = split_index(b)
        λ2, k, μ = split_index(c)
        # eliminate cases which go to 0 due to inconsistency
        if μ != μ2 || ν != ν2 || λ != λ2 throw(ArgumentError("incompatible indices")) end
        i, j, k, λ, μ, ν, combine_indices(i, j, k)
    end

    # generate data
    arr = zeros(Float64, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                local i, j, k, λ, μ, ν, p
                try i, j, k, λ, μ, ν, p = _abc2ijkλμνp(a, b, c) catch; continue end
                i, j, k, λ, μ, ν = anyon.([i, j, k, λ, μ, ν])
                arr[a, b, c, p] = Gsymbol(i, j, λ, μ, k, ν) * √√(qdim(i)*qdim(j)*qdim(k))
            end
        end
    end
    arr
end

function generate_tensor_data(::Type{Tail})
    # break out indices
    function _ab2xyμνp(a::Int, b::Int)
        μ, x, ν = p2ijk(a)
        ν2, y, μ2 = p2ijk(b)
        # eliminate cases which go to 0 due to inconsistency
        if μ != μ2 || ν != ν2 throw(ArgumentError("incompatible indices")) end
        x, y, μ, ν, combine_indices(x, 1, y)
    end

    # generate data
    arr = zeros(Float64, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            try x, y, μ, ν, p = _ab2xyμνp(a, b) catch; continue end
            arr[a, b, p] = x == y ? 1 : 0
        end
    end
    arr
end

function generate_tensor_data(::Type{Crossing})
    # break out indices
    function _abcd2ksλμνξ(a::Int, b::Int, c::Int, d::Int)
        ξ, k, λ = split_index(a)
        λ2, s, μ = split_index(b)
        μ2, k2, ν = split_index(c)
        ν2, s2, ξ2 = split_index(d)
        if ξ != ξ2 || λ != λ2 || μ != μ2 || ν != ν2 || k != k2 || s != s2 throw(ArgumentError("incompatible indices")) end
        k, s, λ, μ, ν, ξ
    end
    arr = zeros(Float64, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                for d in 1:5
                    try k, s, λ, μ, ν, ξ = _abcd2ksλμνξ(a, b, c, d) catch; continue end
                    arr[a, b, c, d] = Gsymbol(μ, λ, ξ, ν, s, k)
                end
            end
        end
    end
    arr
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
