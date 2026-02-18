module FibTensorTypes

using ..IndexTriplets
using ..TensorNetworks
import ..TensorNetworks: tensor_ports, tensor_data, tensor_color, tensor_marker

using TensorKitSectors # Fibonacci input category data; dim and N, F, R symbols

export FibTensorType
export REFLECTOR, BOUNDARY, VACUUMLOOP, ELBOW_T1, ELBOW_T2, ELBOW_T3
export VERTEX, CROSSING, FUSION
export STRINGEND, EXCITATION, DOUBLEDFUSION

### UTILS ###

const ϕ = (1 + √5) / 2
const D = √(1 + Φ^2)
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
struct ELBOW_T1 <: FibTensorType end
struct ELBOW_T2 <: FibTensorType end
struct ELBOW_T3 <: FibTensorType end

struct VERTEX <: FibTensorType end
struct CROSSING <: FibTensorType end
struct FUSION <: FibTensorType end

struct STRINGEND <: FibTensorType end
struct EXCITATION <: FibTensorType end
struct DOUBLEDFUSION <: FibTensorType end

### TENSOR PORTS ###

tensor_ports(::Type{REFLECTOR}) = (:V1, :V2)
tensor_ports(::Type{BOUNDARY}) = (:V1, :V2)
tensor_ports(::Type{VACUUMLOOP}) = (:V1, :V2)
tensor_ports(::Type{ELBOW_T1}) = (:V2, :V3, :P)
tensor_ports(::Type{ELBOW_T2}) = (:V1, :V3, :P)
tensor_ports(::Type{ELBOW_T3}) = (:V1, :V2, :P)

tensor_ports(::Type{VERTEX}) = (:V1, :V2, :V3, :P)
tensor_ports(::Type{CROSSING}) = (:L, :U, :R, :D)
tensor_ports(::Type{FUSION}) = (:V1, :V2, :V3)

tensor_ports(::Type{STRINGEND}) = (:α, :β, :k, :l, :D, :S, :U, :P)
tensor_ports(::Type{EXCITATION}) = (:a, :b, :D, :S, :U, :P)
tensor_ports(::Type{DOUBLEDFUSION}) = (:a, :b, :c, :d, :e, :f, :V1, :V2, :V3)

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
    for V1 in 1:5, V2 in 1:5
        μ, i, ν = split_index(V1)
        ν2, i2, μ2 = split_index(V2)
        if μ != μ2 || ν != ν2 || i != i2 continue end
        arr[V1, V2] = 1
    end
    arr
end

function _generate_tensor_data(::Type{BOUNDARY})
    arr = zeros(Float64, 5, 5)
    for V1 in 1:5, V2 in 1:5
        μ, i, ν = split_index(V1)
        ν2, i2, μ2 = split_index(V2)
        if μ != μ2 || ν != ν2 || i != i2 continue end
        if ν == 0 arr[V1, V2] = 1 end
    end
    arr
end

function _generate_tensor_data(::Type{VACUUMLOOP})
    arr = zeros(Float64, 5, 5)
    for V1 in 1:5, V2 in 1:5
        μ, i, ν = split_index(V1)
        ν2, i2, μ2 = split_index(V2)
        if μ != μ2 || ν != ν2 || i != i2 continue end
        arr[V1, V2] = ν == 0 ? 1 : ϕ
    end
    arr
end

function _generate_tensor_data(::Type{ELBOW_T1})
    # just permute the standard tail (ELBOW_T2)
    dat = tensor_data(ELBOW_T2)
    arr = zeros(Float64, 5, 5, 5)
    for V2 in 1:5, V3 in 1:5, P in 1:5
        if dat[V2, V3, P] == 1
            arr[V2, V3, permute_index_mapping(P, (2, 3, 1))] = 1
        end
    end
    arr
end

function _generate_tensor_data(::Type{ELBOW_T2})
    arr = zeros(Float64, 5, 5, 5)
    for V1 in 1:5, V3 in 1:5
        μ, x, ν = split_index(V1)
        ν2, x2, μ2 = split_index(V3)
        if μ != μ2 || ν != ν2 || x != x2 continue end
        P = combine_indices(x, 0, x2)
        arr[V1, V3, P] = 1
    end
    arr
end

function _generate_tensor_data(::Type{ELBOW_T3})
    # just permute the standard tail (ELBOW_T2)
    dat = tensor_data(ELBOW_T2)
    arr = zeros(Float64, 5, 5, 5)
    for V1 in 1:5, V2 in 1:5, P in 1:5
        if dat[V1, V2, P] == 1
            arr[V1, V2, permute_index_mapping(P, (3, 1, 2))] = 1
        end
    end
    arr
end

function _generate_tensor_data(::Type{VERTEX})
    arr = zeros(Float64, 5, 5, 5, 5)
    for V1 in 1:5, V2 in 1:5, V3 in 1:5
        # break out indices
        μ, i, ν = split_index(V1)
        ν2, j, λ = split_index(V2)
        λ2, k, μ2 = split_index(V3)
        # abort inconsistent cases, leaving them 0
        if μ != μ2 || ν != ν2 || λ != λ2 continue end
        # get p and ensure this combination is fusion-valid
        P = combine_indices(i, j, k)
        P <= 5 || continue
        # convert integers to anyons, and calculate the entry
        i, j, k, λ, μ, ν = anyon.([i, j, k, λ, μ, ν])
        arr[V1, V2, V3, P] = Gsymbol(i, j, λ, μ, k, ν) * √√(qdim(i) * qdim(j) * qdim(k))
    end
    arr
end

function _generate_tensor_data(::Type{CROSSING})
    arr = zeros(Float64, 5, 5, 5, 5)
    for L in 1:5, U in 1:5, R in 1:5, D in 1:5
        ξ, k, λ = split_index(L)
        λ2, s, μ = split_index(U)
        μ2, k2, ν = split_index(R)
        ν2, s2, ξ2 = split_index(D)
        if ξ != ξ2 || λ != λ2 || μ != μ2 || ν != ν2 || k != k2 || s != s2
            continue
        end
        μ, λ, ξ, ν, s, k = anyon.([μ, λ, ξ, ν, s, k])
        arr[L, U, R, D] = Gsymbol(μ, λ, ξ, ν, s, k)
    end
    arr
end

function _generate_tensor_data(::Type{FUSION})
    arr = zeros(Float64, 5, 5, 5)
    for V1 in 1:5, V2 in 1:5, V3 in 1:5
        μ, s, ν = split_index(V1)
        ν2, t, λ = split_index(V2)
        λ2, u, μ2 = split_index(V3)
        if λ != λ2 || μ != μ2 || ν != ν2 continue end
        s, t, λ, ν, u, μ = anyon.([s, t, λ, ν, u, μ])
        arr[V1, V2, V3] = Gsymbol(s, t, λ, ν, u, μ) * √√(qdim(s) * qdim(t) * qdim(u))
    end
    arr
end

# TODO extra R factor for fusion tree base
function _generate_tensor_data(::Type{STRINGEND})
    arr = zeros(Float64, 2, 2, 2, 2, 3, 3, 3, 3)
    for α in 1:2, β in 1:2, k in 1:2, l in 1:2
        for D in 1:5, S in 1:5, U in 1:5
            αprime, y, μ = split_index(D)
            μ2, kprime, ν = split_index(S)
            ν2, x, αprime2 = split_index(U)
            if αprime != αprime2 || μ != μ2 || ν != ν2 continue end
            if k != kprime || α != αprime continue end
            P = combine_indices(x, l, y)
            α, β, k, l, x, y, μ, ν = anyon.([α, β, k, l, x, y, μ, ν])
            arr[α, β, k, l, D, S, U, P] = √√(qdim(k) * qdim(x) * qdim(y)) * √qdim(β) *
                Gsymbol(μ, β, α, ν, x, k) * Gsymbol(α, y, x, β, μ, l) / √qdim(α)
        end
    end
    arr
end

function _generate_tensor_data(::Type{EXCITATION})
    arr = zeros(Float64, 2, 2, 3, 3, 3, 3)
    dat = tensor_data(STRINGEND)
    for a in 1:2, b in 1:2
        for D in 1:5, S in 1:5, U in 1:5
            # TODO
        end
    end
    arr
end


function _generate_tensor_data(::Type{DOUBLEDFUSION})
    arr = zeros(Float64, 2, 2, 2, 2, 2, 2, 3, 3, 3)
    dat = tensor_data(FUSION)
    # TODO
    arr
end

### TENSOR DISPLAY PROPERTIES ###

tensor_color(::Type{REFLECTOR}) = :gray
tensor_color(::Type{BOUNDARY}) = :black
tensor_color(::Type{VACUUMLOOP}) = :orange
tensor_color(::Type{ELBOW_T1}) = :cyan
tensor_color(::Type{ELBOW_T2}) = :blue
tensor_color(::Type{ELBOW_T3}) = :cyan

tensor_color(::Type{VERTEX}) = :red
tensor_color(::Type{CROSSING}) = :green
tensor_color(::Type{FUSION}) = :teal

tensor_color(::Type{STRINGEND}) = :gray
tensor_color(::Type{EXCITATION}) = :red
tensor_color(::Type{DOUBLEDFUSION}) = :red

tensor_marker(::Type{REFLECTOR}) = :vline
tensor_marker(::Type{BOUNDARY}) = :xcross
tensor_marker(::Type{VACUUMLOOP}) = :circle
tensor_marker(::Type{ELBOW_T1}) = :rect
tensor_marker(::Type{ELBOW_T2}) = :rect
tensor_marker(::Type{ELBOW_T3}) = :rect

tensor_marker(::Type{VERTEX}) = :star6
tensor_marker(::Type{CROSSING}) = :star4
tensor_marker(::Type{FUSION}) = :star3

tensor_marker(::Type{STRINGEND}) = :rect
tensor_marker(::Type{EXCITATION}) = :rect
tensor_marker(::Type{DOUBLEDFUSION}) = :star3

end # module TensorTypes
