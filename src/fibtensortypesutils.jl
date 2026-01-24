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
