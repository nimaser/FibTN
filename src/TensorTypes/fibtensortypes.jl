module FibTensorTypes

using ..IndexTriplets
using ..TensorNetworks
import ..TensorNetworks: tensor_ports, tensor_data, tensor_color, tensor_marker

using TensorKitSectors # Fibonacci input category data; dim and N, F, R symbols

export FibTensorType
export REFLECTOR, BOUNDARY, VACUUMLOOP, TAIL, T_ELBOW, ELBOW_T
export VERTEX, CROSSING, FUSION
export END, EXCITATION, DOUBLEDFUSION

### UTILS ###

const ϕ = (1 + √5) / 2
const qdim = TensorKitSectors.dim
anyon(a::Int) = a == 0 ? FibonacciAnyon(:I) : a == 1 ? FibonacciAnyon(:τ) : error("a must be 0 or 1")

function Gsymbol(
    a::FibonacciAnyon, b::FibonacciAnyon, c::FibonacciAnyon,
    d::FibonacciAnyon, e::FibonacciAnyon, f::FibonacciAnyon
)
    Fsymbol(a, b, c, d, e, f) / √(qdim(e) * qdim(f))
end

### TENSOR TYPES ###

abstract type FibTensorType <: TensorType end

struct REFLECTOR <: FibTensorType end
struct BOUNDARY <: FibTensorType end
struct VACUUMLOOP <: FibTensorType end
struct TAIL <: FibTensorType end
struct T_ELBOW <: FibTensorType end
struct ELBOW_T <: FibTensorType end

struct VERTEX <: FibTensorType end
struct CROSSING <: FibTensorType end
struct FUSION <: FibTensorType end

struct END <: FibTensorType end
struct EXCITATION <: FibTensorType end
struct DOUBLEDFUSION <: FibTensorType end

### TENSOR PORTS ###

tensor_ports(::Type{REFLECTOR}) = (:a, :b)
tensor_ports(::Type{BOUNDARY}) = (:a, :b)
tensor_ports(::Type{VACUUMLOOP}) = (:a, :b)
tensor_ports(::Type{TAIL}) = (:a, :c, :p)
tensor_ports(::Type{T_ELBOW}) = (:b, :c, :p)
tensor_ports(::Type{ELBOW_T}) = (:a, :b, :p)

tensor_ports(::Type{VERTEX}) = (:a, :b, :c, :p)
tensor_ports(::Type{CROSSING}) = (:a, :b, :c, :d)
tensor_ports(::Type{FUSION}) = (:a, :b, :c)

tensor_ports(::Type{END}) = (
#TODO
)
tensor_ports(::Type{EXCITATION}) = (
#TODO
)
tensor_ports(::Type{DOUBLEDFUSION}) = (
#TODO
)

### TENSOR DATA ###

const _cache = IdDict{DataType,Any}()

function tensor_data(::Type{T}) where {T<:FibTensorType}
    # try to get the data from the cache, else generate it
    get!(_cache, T) do
        _generate_tensor_data(T)
    end
end

function _generate_tensor_data(::Type{REFLECTOR})
    arr = zeros(Float64, 5, 5)
    for a in 1:5
        for b in 1:5
            μ, i, ν = split_index(a)
            ν2, i2, μ2 = split_index(b)
            if μ != μ2 || ν != ν2 || i != i2
                continue
            end
            arr[a, b] = 1
        end
    end
    arr
end

function _generate_tensor_data(::Type{BOUNDARY})
    arr = zeros(Float64, 5, 5)
    for a in 1:5
        for b in 1:5
            μ, i, ν = split_index(a)
            ν2, i2, μ2 = split_index(b)
            if μ != μ2 || ν != ν2 || i != i2
                continue
            end
            if ν == 0
                arr[a, b] = 1
            end
        end
    end
    arr
end

function _generate_tensor_data(::Type{VACUUMLOOP})
    arr = zeros(Float64, 5, 5)
    for a in 1:5
        for b in 1:5
            μ, i, ν = split_index(a)
            ν2, i2, μ2 = split_index(b)
            if μ != μ2 || ν != ν2 || i != i2
                continue
            end
            arr[a, b] = ν == 0 ? 1 : ϕ
        end
    end
    arr
end

function _generate_tensor_data(::Type{TAIL})
    arr = zeros(Float64, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            μ, x, ν = split_index(a)
            ν2, x2, μ2 = split_index(b)
            if μ != μ2 || ν != ν2 || x != x2
                continue
            end
            p = combine_indices(x, 0, x2)
            arr[a, b, p] = 1
        end
    end
    arr
end

function _generate_tensor_data(::Type{T_ELBOW})
    # just permute the standard tail
    dat = tensor_data(TAIL)
    arr = zeros(Float64, 5, 5, 5)
    for a in 1:5, b in 1:5, p in 1:5
        if dat[a, b, p] == 1
            arr[a, b, permute_index_mapping(p, (2, 3, 1))] = 1
        end
    end
    arr
end

function _generate_tensor_data(::Type{ELBOW_T})
    # just permute the standard tail
    dat = tensor_data(TAIL)
    arr = zeros(Float64, 5, 5, 5)
    for a in 1:5, b in 1:5, p in 1:5
        if dat[a, b, p] == 1
            arr[a, b, permute_index_mapping(p, (3, 1, 2))] = 1
        end
    end
    arr
end

function _generate_tensor_data(::Type{VERTEX})
    arr = zeros(Float64, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                # break out indices
                μ, i, ν = split_index(a)
                ν2, j, λ = split_index(b)
                λ2, k, μ2 = split_index(c)
                # abort inconsistent cases, leaving them 0
                if μ != μ2 || ν != ν2 || λ != λ2
                    continue
                end
                # get p and ensure this combination is fusion-valid
                p = combine_indices(i, j, k)
                p <= 5 || continue
                # convert integers to anyons, and calculate the entry
                i, j, k, λ, μ, ν = anyon.([i, j, k, λ, μ, ν])
                arr[a, b, c, p] = Gsymbol(i, j, λ, μ, k, ν) * √√(qdim(i) * qdim(j) * qdim(k))
            end
        end
    end
    arr
end

function _generate_tensor_data(::Type{CROSSING})
    arr = zeros(Float64, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                for d in 1:5
                    ξ, k, λ = split_index(a)
                    λ2, s, μ = split_index(b)
                    μ2, k2, ν = split_index(c)
                    ν2, s2, ξ2 = split_index(d)
                    if ξ != ξ2 || λ != λ2 || μ != μ2 || ν != ν2 || k != k2 || s != s2
                        continue
                    end
                    μ, λ, ξ, ν, s, k = anyon.([μ, λ, ξ, ν, s, k])
                    arr[a, b, c, d] = Gsymbol(μ, λ, ξ, ν, s, k)
                end
            end
        end
    end
    arr
end

function _generate_tensor_data(::Type{FUSION})
    arr = zeros(Float64, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                μ, s, ν = split_index(a)
                ν2, t, λ = split_index(b)
                λ2, u, μ2 = split_index(c)
                if λ != λ2 || μ != μ2 || ν != ν2
                    continue
                end
                s, t, λ, ν, u, μ = anyon.([s, t, λ, ν, u, μ])
                arr[a, b, c] = Gsymbol(s, t, λ, ν, u, μ) * √√(qdim(s) * qdim(t) * qdim(u))
            end
        end
    end
    arr
end

function _generate_tensor_data(::Type{END})
    # TODO
end

function _generate_tensor_data(::Type{EXCITATION})
    # TODO
end

function _generate_tensor_data(::Type{DOUBLEDFUSION})
    # TODO
end

### TENSOR DISPLAY PROPERTIES ###

tensor_color(::Type{REFLECTOR}) = :gray
tensor_color(::Type{BOUNDARY}) = :black
tensor_color(::Type{VACUUMLOOP}) = :orange
tensor_color(::Type{TAIL}) = :blue
tensor_color(::Type{T_ELBOW}) = :cyan
tensor_color(::Type{ELBOW_T}) = :cyan
tensor_color(::Type{VERTEX}) = :red
tensor_color(::Type{CROSSING}) = :green
tensor_color(::Type{FUSION}) = :teal

tensor_marker(::Type{REFLECTOR}) = :vline
tensor_marker(::Type{BOUNDARY}) = :xcross
tensor_marker(::Type{VACUUMLOOP}) = :circle
tensor_marker(::Type{TAIL}) = :rect
tensor_marker(::Type{T_ELBOW}) = :rect
tensor_marker(::Type{ELBOW_T}) = :rect
tensor_marker(::Type{VERTEX}) = :star6
tensor_marker(::Type{CROSSING}) = :star4
tensor_marker(::Type{FUSION}) = :star3

end # module TensorTypes
