#=
# This module creates the tensors which are used as building blocks for the networks
# which represent full many-body states. They are all symmetric TODO, and follow the
# convention that any physical index will be at the end of its index list, and have
# a tag of pn, where n is a number. Virtual indices will have tags of in, where n is
# a number. Those numbers n are just indices internal to the tensor, and have no
# bearing on any other tensor or any relations between tensors. Physical and virtual
# indices will also have a "phys" or "virt" tag.
=#

using ITensors

# Fibonacci input category data; dim and N, F, R symbols
using TensorKitSectors
const qdim = TensorKitSectors.dim # to avoid name conflict

function Gsymbol(
        a::FibonacciAnyon, b::FibonacciAnyon, c::FibonacciAnyon,
        d::FibonacciAnyon, e::FibonacciAnyon, f::FibonacciAnyon
    )
    Fsymbol(a, b, c, d, e, f) / √(qdim(e)*qdim(f))
end

function virtualindices(T::ITensor)
    [i for i in inds(T) if hastags(i, "virt")]
end

function physicalindices(T::ITensor)
    [i for i in inds(T) if hastags(i, "phys")]
end

function virtualindices(V::Vector{Index})
    [i for i in V if hastags(i, "virt")]
end

function physicalindices(V::Vector{Index})
    [i for i in V if hastags(i, "phys")]
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

function StringTripletVector(a::Int)
    x = Index(5, "virt,i1")
    onehot(x=>a)
end

function StringTripletReflector(i1::Index, i2::Index)
    if i1.space != i2.space != 5
        throw(ArgumentError("indices must be dim 5"))
    end
    arr = [1 0 0 0 0;
           0 0 0 1 0;
           0 0 1 0 0;
           0 1 0 0 0;
           0 0 0 0 1]
    ITensor(arr, i1, i2)
end

###############################################################################
# GROUND STATE TENSORS
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

function GSTriangle()
    i1 = Index(5, "virt,i1")
    i2 = Index(5, "virt,i2")
    i3 = Index(5, "virt,i3")
    p1 = Index(5, "phys,p1")
    ITensor(GSTriangle_data(), i1, i2, i3, p1)
end

function GSSquare()
end

function GSCircle()
end

function GSTail()
end

###############################################################################
# EXCITED STATE TENSORS
###############################################################################
