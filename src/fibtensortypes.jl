module FibTensorTypes

# Fibonacci input category data; dim and N, F, R symbols
using TensorKitSectors
const qdim = TensorKitSectors.dim # to avoid name conflict

export AbstractFibTensorType
export Reflector, LoopAmplitude, Vertex, Tail, Crossing, Fusion, End, Excitation, DoubledFusion
export index_data, tensor_data

export Gsymbol, ijk2p, p2ijk, abc2ijkλμνp

### UTILS ###

const \phi = (1 + \sqrt5) / 2

function Gsymbol(
        a::FibonacciAnyon, b::FibonacciAnyon, c::FibonacciAnyon,
        d::FibonacciAnyon, e::FibonacciAnyon, f::FibonacciAnyon
    )
    Fsymbol(a, b, c, d, e, f) / √(qdim(e)*qdim(f))
end

function ijk2p(i::FibonacciAnyon, j::FibonacciAnyon, k::FibonacciAnyon)
    if (i == j == k) p = (i == FibonacciAnyon(:I)) ? 1 : 5
    elseif (i == FibonacciAnyon(:I)) p = 2
    elseif (j == FibonacciAnyon(:I)) p = 3
    elseif (k == FibonacciAnyon(:I)) p = 4
    end
    p
end

function p2ijk(p::Int)
    i = j = k = FibonacciAnyon(:τ)
    if p == 1 i = j = k = FibonacciAnyon(:I)
    elseif p == 2 i = FibonacciAnyon(:I)
    elseif p == 3 j = FibonacciAnyon(:I)
    elseif p == 4 k = FibonacciAnyon(:I)
    end
    i, j, k
end

function abc2ijkλμνp(a::Int, b::Int, c::Int)
    # eliminate cases which go to 0 due to inconsistency
    # side ends with 1 but next side doesn't start with 1
    if (a ∈ [1, 4] && b ∉ [1, 2]) throw(ArgumentError("a and b not compatible")) end
    if (b ∈ [1, 4] && c ∉ [1, 2]) throw(ArgumentError("b and c not compatible")) end
    if (c ∈ [1, 4] && a ∉ [1, 2]) throw(ArgumentError("c and a not compatible")) end
    # side ends with τ but next side doesn't start with τ
    if (a ∉ [1, 4] && b ∈ [1, 2]) throw(ArgumentError("a and b not compatible")) end
    if (b ∉ [1, 4] && c ∈ [1, 2]) throw(ArgumentError("b and c not compatible")) end
    if (c ∉ [1, 4] && a ∈ [1, 2]) throw(ArgumentError("c and a not compatible")) end
    # break out indices
    i = a ∈ [1, 3] ? FibonacciAnyon(:I) : FibonacciAnyon(:τ)
    ν = a ∈ [1, 4] ? FibonacciAnyon(:I) : FibonacciAnyon(:τ)
    j = b ∈ [1, 3] ? FibonacciAnyon(:I) : FibonacciAnyon(:τ)
    λ = b ∈ [1, 4] ? FibonacciAnyon(:I) : FibonacciAnyon(:τ)
    k = c ∈ [1, 3] ? FibonacciAnyon(:I) : FibonacciAnyon(:τ)
    μ = c ∈ [1, 4] ? FibonacciAnyon(:I) : FibonacciAnyon(:τ)
    i, j, k, λ, μ, ν, ijk2p(i, j, k)
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
