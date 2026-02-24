module FibTensorTypes

using ..IndexTriplets
using ..TensorNetworks
import ..TensorNetworks: tensor_ports, tensor_data, tensor_color, tensor_marker

using TensorKitSectors # Fibonacci input category data; dim and N, F, R symbols
using TensorOperations # to do some small and compile-time-known contractions

export FibTensorType
export REFLECTOR, BOUNDARY, VACUUMLOOP, ELBOW_T1, ELBOW_T2, ELBOW_T3, TAIL
export VERTEX, CROSSING, FUSION
export STRINGEND, EXCITATION, EXCITATION_CONTROL, FUSIONTREEROOT, DOUBLEDFUSION

### UTILS ###

const ϕ = (1 + √5) / 2
const D = √(1 + ϕ^2)
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
struct TAIL <: FibTensorType end # alias for ELBOW_T2 with differently named ports

struct VERTEX <: FibTensorType end
struct CROSSING <: FibTensorType end
struct FUSION <: FibTensorType end

struct STRINGEND <: FibTensorType end
struct EXCITATION <: FibTensorType end
struct EXCITATION_CONTROL{A,B,L} <: FibTensorType end
struct FUSIONTREEROOT <: FibTensorType end
struct DOUBLEDFUSION <: FibTensorType end

### TENSOR PORTS ###

tensor_ports(::Type{REFLECTOR}) = (:V1, :V2)
tensor_ports(::Type{BOUNDARY}) = (:V1, :V2)
tensor_ports(::Type{VACUUMLOOP}) = (:V1, :V2)
tensor_ports(::Type{ELBOW_T1}) = (:V2, :V3, :P)
tensor_ports(::Type{ELBOW_T2}) = (:V1, :V3, :P)
tensor_ports(::Type{ELBOW_T3}) = (:V1, :V2, :P)
tensor_ports(::Type{TAIL}) = (:V1, :V2, :P)

tensor_ports(::Type{VERTEX}) = (:V1, :V2, :V3, :P)
tensor_ports(::Type{CROSSING}) = (:L, :U, :R, :D)
tensor_ports(::Type{FUSION}) = (:V1, :V2, :V3)

tensor_ports(::Type{STRINGEND}) = (:α, :β, :k, :l, :V1, :V2, :S, :P)
tensor_ports(::Type{EXCITATION}) = (:a, :b, :l, :V1, :V2, :S, :P)
tensor_ports(::Type{<:EXCITATION_CONTROL}) = (:a, :b, :l)
tensor_ports(::Type{FUSIONTREEROOT}) = (:ai, :bi, :li, :ao, :bo, :lo)
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
        P = combine_qubits(x, 0, x2)
        arr[V1, V3, P] = 1
    end
    arr
end

_generate_tensor_data(::Type{TAIL}) = _generate_tensor_data(ELBOW_T2)

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
        P = combine_qubits(i, j, k)
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
        arr[V1, V2, V3] = Gsymbol(s, t, λ, μ, u, ν) * √√(qdim(s) * qdim(t) * qdim(u))
    end
    arr
end

# TODO extra R factor for fusion tree base
function _generate_tensor_data(::Type{STRINGEND})
    arr = zeros(Float64, 2, 2, 2, 2, 5, 5, 5, 5)
    for α in 1:2, β in 1:2, k in 1:2, l in 1:2
        # note that because Julia uses 1-based indexes, we have to be careful
        # about subtracting 1 from indices at the correct times; 1 or 2 in the
        # indices of this tensor are actually a 0 or 1 value in qubit terms
        for V1 in 1:5, V2 in 1:5, S in 1:5
            ν, x, αprime = split_index(V1)
            αprime2, y, μ = split_index(V2)
            μ2, kprime, ν2 = split_index(S)
            if αprime != αprime2 || μ != μ2 || ν != ν2 continue end
            if k-1 != kprime || α-1 != αprime continue end
            P = combine_qubits(x, l-1, y)
            # vara = var (anyon representation)
            αa, βa, ka, la, x, y, μ, ν = anyon.([α-1, β-1, k-1, l-1, x, y, μ, ν])
            val = √√(qdim(ka) * qdim(x) * qdim(y)) * √qdim(βa) *
                Gsymbol(μ, βa, αa, ν, x, ka) * Gsymbol(αa, y, x, βa, μ, la) / √qdim(αa)
            # P may take index values outside of 1:5, because the tail value l is not fixed to
            # trivial. However, these entries go to 0, so to avoid bounds errors we just check
            # val before writing and indeed all nonzero entries fit in the string-net subspace
            if !iszero(val) arr[α, β, k, l, V1, V2, S, P] = val end
        end
    end
    arr
end

function _generate_tensor_data(::Type{EXCITATION})
    dat = tensor_data(STRINGEND)
    evil = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2) # evil prefactor
    # first fill evil tensor, which is a prefactor relating the string end tensor
    # to the excitation tensor, and partially encodes the map from the doubled anyonic fusion
    # basis to the single one
    for a in 1:2, b in 1:2, l in 1:2, α in 1:2, β in 1:2, k in 1:2, λ in 1:2
        # 1 or 2 in the indices of this tensor are actually a 0 or 1 value in qubit terms
        if l != λ continue end
        aa, ba, la, αa, βa, ka, λa = anyon.([a, b, l, α, β, k, λ] .- 1)
        val = √√(qdim(aa) * qdim(ba) * qdim(ka)) * √qdim(αa) * √qdim(βa) / D
        sum = 0
        for γ in 0:1, δ in 0:1 # since these aren't array indices we can do normal 0 to 1
            γa, δa = anyon.([γ, δ])
            sum += qdim(γa) * qdim(δa) * Rsymbol(aa, αa, γa) * Rsymbol(αa, ba, δa) *
                Gsymbol(ka, δa, αa, aa, γa, ba) *
                Gsymbol(βa, aa, ba, αa, δa, la) *
                Gsymbol(ka, βa, aa, γa, αa, δa)
        end
        evil[a, b, l, α, β, k, λ] = val * sum
    end
    # the contraction between the evil tensor and the stringend tensor gives the excitation
    @tensor arr[a, b, l, V1, V2, S, P] := evil[a, b, l, α, β, k, λ] * dat[α, β, k, λ, V1, V2, S, P]
    arr
end

function _generate_tensor_data(::Type{EXCITATION_CONTROL{A,B,L}}) where {A,B,L}
    arr = zeros(Float64, 2, 2, 2)
    arr[A+1, B+1, L+1] = 1.0
    arr
end

function _generate_tensor_data(::Type{FUSIONTREEROOT})
    arr = zeros(ComplexF64, 2, 2, 2, 2, 2, 2)
    for a in 1:2, b in 1:2, l in 1:2
        aa, ba, la = anyon.([a, b, l] .- 1)
        arr[a, b, l, a, b, l] = conj(Rsymbol(ba, aa, la))
    end
    arr
end

function _generate_tensor_data(::Type{DOUBLEDFUSION})
    # we could do an evil contraction like with EXCITATION, but it's easier to just recompute FUSION
    # in the process of computing this one
    arr = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 5, 5, 5)
    for a in 1:2, b in 1:2, c in 1:2, d in 1:2, e in 1:2, f in 1:2
        # remember to subtract one from these guys when using them in anyon computations ^
        aa, ba, ca, da, ea, fa = anyon.([a, b, c, d, e, f] .- 1)
        for V1 in 1:5, V2 in 1:5, V3 in 1:5
            μ, s, ν = split_index(V1)
            ν2, t, λ = split_index(V2)
            λ2, u, μ2 = split_index(V3)
            if λ != λ2 || μ != μ2 || ν != ν2 continue end
            s, t, λ, ν, u, μ = anyon.([s, t, λ, ν, u, μ])
            fusionfactor = Gsymbol(s, t, λ, μ, u, ν) * √√(qdim(s) * qdim(t) * qdim(u))
            if fusionfactor == 0 continue end
            prefactor = √√(qdim(s)*qdim(t)*qdim(u)*qdim(aa)*qdim(ba)*qdim(ca)*qdim(da)*qdim(ea)*qdim(fa))
            sum = 0
            for γ in 0:1, δ in 0:1 # since these aren't array indices we can do normal 0 to 1
                γa, δa = anyon.([γ, δ])
                sum += qdim(γa) * qdim(δa) * Rsymbol(aa, da, γa) *
                    Gsymbol(u, ba, da, ea, δa, fa) *
                    Gsymbol(ca, δa, da, aa, γa, ea) *
                    Gsymbol(t, δa, γa, da, aa, ca) *
                    Gsymbol(s, t, δa, ba, u, aa)
            end
            arr[a, b, c, d, e, f, V1, V2, V3] = prefactor * sum * fusionfactor
        end
    end
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
tensor_color(::Type{<:EXCITATION_CONTROL}) = :orange
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
tensor_marker(::Type{<:EXCITATION_CONTROL}) = :diamond
tensor_marker(::Type{DOUBLEDFUSION}) = :star3

end # module TensorTypes
