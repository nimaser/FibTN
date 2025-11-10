#=
# This module provides an API to build the building-block tensors used to represent
# the many-body state as a tensor network.
#
# Tensors have a type, which is provided by the TensorType enum and is used to
# construct a tensor's indices and the tensor itself.
#
# Tensors have a set of virtual and a set of physical indices, each of which has a
# typetag which can be either of
#     virt
#     phys
# depending on the index type.
#
# Tensors have a label, which is stored in each of their indices.
#
# Tensor indices have a human-readable idtag which has the tensorlabel and an indexidx
# indicating which virtual or physical index it is, which looks like
#     tensorlabel-vm
#     tensorlabel-pn
# depending on whether the index was virtual or physical, where this is the mth or nth
# virtual or physical index for this tensor respectively.
=#

using ITensors

###############################################################################
# FIBONACCI DATA
###############################################################################

# Fibonacci input category data; dim and N, F, R symbols
using TensorKitSectors
const qdim = TensorKitSectors.dim # to avoid name conflict

function Gsymbol(
        a::FibonacciAnyon, b::FibonacciAnyon, c::FibonacciAnyon,
        d::FibonacciAnyon, e::FibonacciAnyon, f::FibonacciAnyon
    )
    Fsymbol(a, b, c, d, e, f) / √(qdim(e)*qdim(f))
end

###############################################################################
# INDEX CONVERSIONS
###############################################################################

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

function abc2etc(a::Int, b::Int, c::Int)
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

###############################################################################
# MISC TENSORS
###############################################################################

function make_StringTripletVector(vinds::Vector{Index}, a::Int)
    if length(vinds) != 1 throw(ArgumentError("got $(length(vinds)) vinds, not 1")) end
    if vinds[1].space != 5 throw(ArgumentError("got vind with dimension $(vinds[1].space), not 5")) end
    onehot(vinds[1]=>a)
end

function make_StringTripletReflector(vinds::Vector{Index})
    if length(vinds) != 2 throw(ArgumentError("got $(length(vinds)) vinds, not 2")) end
    for vind in vinds
        if vind.space != 5 throw(ArgumentError("got vind with dimension $(vind.space), not 5")) end
    end
    arr = [1 0 0 0 0;
           0 0 0 1 0;
           0 0 1 0 0;
           0 1 0 0 0;
           0 0 0 0 1]
    ITensor(arr, vinds...)
end

###############################################################################
# GROUND STATE TENSOR DATA
###############################################################################

function GSTriangle_data()
    GSTriangle_data = zeros(Float64, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                local i, j, k, λ, μ, ν, p
                try i, j, k, λ, μ, ν, p = abc2etc(a, b, c) catch; continue end
                GSTriangle_data[a, b, c, p] = (qdim(λ)*qdim(μ)*qdim(ν))^(1/6) * Gsymbol(i, j, λ, μ, k, ν) * √√(qdim(i)*qdim(j)*qdim(k))
            end
        end
    end
    GSTriangle_data
end

function GSTail_data()
    nothing
end

function GSSquare_data()
    nothing
end

function GSCircle_data()
    nothing
end

###############################################################################
# GROUND STATE TENSORS
###############################################################################

function make_GSTriangle(vinds, pinds)
    # check that right number of vinds and pinds were passed in
    if length(vinds) != 3 throw(ArgumentError("got $(length(vinds)) vinds, not 3")) end
    if length(pinds) != 1 throw(ArgumentError("got $(length(pinds)) pinds, not 1")) end

    # check that indices have the right dimensionality
    for vind in vinds
        if vind.space != 5 throw(ArgumentError("got vind with dimension $(vind.space), not 5")) end
    end
    if pinds[1].space != 5 throw(ArgumentError("got pind with dimension $(pinds[1].space), not 5")) end

    ITensor(GSTriangle_data(), vinds..., pinds...)
end

function make_GSTail()
    nothing
end

function make_GSSquare()
    nothing
end

function make_GSCircle()
    nothing
end

###############################################################################
# EXCITED STATE TENSORS
###############################################################################


###############################################################################
# TENSOR BUILDER API
###############################################################################

@enum TensorType begin
    # misc
    StringTripletVector
    Composite

    # GS
    GSTriangle
    GSTail
    GSSquare
    GSCircle

    # ES

end

function make_tensor_indices(tensorlabel::Any, type::TensorType)
    # misc
    if type == StringTripletVector
        v1 = Index(5, "virt,$(tensorlabel)-v1")
        return [v1], []
    end
    if type == Composite
        throw(ArgumentError("Composite tensor types result from contractions"))
    end

    # GS
    if type == GSTriangle
        v1 = Index(5, "virt,$(tensorlabel)-v1")
        v2 = Index(5, "virt,$(tensorlabel)-v2")
        v3 = Index(5, "virt,$(tensorlabel)-v3")
        p1 = Index(5, "phys,$(tensorlabel)-p1")
        return [v1, v2, v3], [p1]
    end
    if type == GSTail
        return nothing
    end
    if type == GSSquare
        return nothing
    end
    if type == GSCircle
        return nothing
    end

    # ES
end

function get_idtag(i::Index)
    for tag in tags(i)
        if '-' ∈ string(tag) return tag end
    end
    throw(ArgumentError("index $i doesn't have a tagid"))
end

function get_tensorlabel(i::Index)
    idtag = get_idtag(i)
    split(idtag, '-')[1]
end

function get_indexidx(i::Index)
    idtag = get_idtag(i)
    split(idtag, '-')[2]
end

function make_tensor(type::TensorType, vinds::Vector{Index}, pinds::Vector{Index}, data::Any=nothing)
    if type == StringTripletVector
        data = data == nothing ? 1 : data
        return make_StringTripletVector(vinds, data)
    end
    if type == Composite
        throw(ArgumentError("Composite tensor types result from contractions"))
    end
    
    if type == GSTriangle
        return make_GSTriangle(vinds, pinds)
    end
end

