module IndexTriplets

export ijk2p, p2ijk, abc2ijkλμνp

function ijk2p(i::Int8, j::Int8, k::Int8)
    if (i == j == k) p = (i == 0) ? 1 : 5
    elseif (i == 0) p = 2
    elseif (j == 0) p = 3
    elseif (k == 0) p = 4
    end
    p
end

function p2ijk(p::Int8)
    i = j = k = 1
    if p == 1 i = j = k = 0
    elseif p == 2 i = 0
    elseif p == 3 j = 0
    elseif p == 4 k = 0
    end
    i, j, k
end

function abc2ijkλμνp(a::Int8, b::Int8, c::Int8)
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
    i = a ∈ [1, 3] ? 0 : 1
    ν = a ∈ [1, 4] ? 0 : 1
    j = b ∈ [1, 3] ? 0 : 1
    λ = b ∈ [1, 4] ? 0 : 1
    k = c ∈ [1, 3] ? 0 : 1
    μ = c ∈ [1, 4] ? 0 : 1
    i, j, k, λ, μ, ν, ijk2p(i, j, k)
end

end # module IndexTriplets
