# @author Nikhil Maserang
# @date 2025/10/22

# so we know we didn't code bugs
using Test

# Fibonacci input category data; quantum dims, F symbols, etc
using TensorKitSectors 

# make tensors and do contractions
using TensorKit

# don't store all 7 gorillion 0 entries
using SparseArrayKit

function Gsymbol(
        a::FibonacciAnyon, b::FibonacciAnyon, c::FibonacciAnyon,
        d::FibonacciAnyon, e::FibonacciAnyon, f::FibonacciAnyon
    )
    Fsymbol(a, b, c, d, e, f) / √(dim(e)*dim(f))
end

# Helpers

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
    # aggregate qubit label
    if (i == j == k) p = (i == FibonacciAnyon(:I)) ? 1 : 5
    elseif (i == FibonacciAnyon(:I)) p = 2
    elseif (j == FibonacciAnyon(:I)) p = 3
    elseif (k == FibonacciAnyon(:I)) p = 4
    end
    i, j, k, λ, μ, ν, p
end

function gen_GSTriangle_data()
    GSTriangle_data = SparseArray{Float64}(undef, 5, 5, 5, 5)
    for a in 1:5
        for b in 1:5
            for c in 1:5
                local i, j, k, λ, μ, ν, p
                try i, j, k, λ, μ, ν, p = abc2etc(a, b, c) catch; continue end
                GSTriangle_data[a, b, c, p] = Gsymbol(i, j, λ, μ, k, ν) * √√(dim(i)*dim(j)*dim(k))
                # @show a, b, c, p, μ, i, ν, j, λ, k
            end
        end
    end
    GSTriangle_data
end

function gen_GSTriangle_data_test()
    # @show dim(O(2))^(-2)
    # @show dim(O(2))^(-5/4)
    # @show dim(O(2))^(-1)
    # @show dim(O(2))^(-1/2)
    # @show dim(O(2))^(-1/4)
    # @show 1
    # @show dim(O(2))^(1/4)
    # @show dim(O(2))^(1/2)
    # @show dim(O(2))^(3/4)
    # @show dim(O(2))

    # nonzero combinations of tensor indices, access as abcvals[:, idx]
    avals = [1 1 2 2 2 3 3 3 3 3 4 4 4 5 5 5 5 5]
    bvals = [1 2 3 4 5 3 3 4 5 5 1 2 2 3 3 4 5 5]
    cvals = [1 4 4 1 4 3 5 2 3 5 2 3 5 3 5 2 3 5]
    abcvals = [avals ; bvals ; cvals]

    GSTriangle_data = SparseArray{Float64}(undef, 5, 5, 5, 5)
    for x in axes(abcvals, 2)
        a, b, c = abcvals[:, x] 
        i, j, k, λ, μ, ν, p = abc2etc(a, b, c)
        GSTriangle_data[a, b, c, p] = Gsymbol(i, j, λ, μ, k, ν) * √√(dim(i)*dim(j)*dim(k))
        # @show a, b, c, p, i, j, k, λ, μ, ν, √√(dim(i)*dim(j)*dim(k))
        # @show a, b, c, p, i, j, k, λ, μ, ν, Fsymbol(i, j, λ, μ, k, ν)
        # @show a, b, c, p, i, j, k, λ, μ, ν, Gsymbol(i, j, λ, μ, k, ν)
        # @show a, b, c, p, i, j, k, λ, μ, ν, GSTriangle_data[a, b, c, ijkidx...]
    end
    removezeros = x->x!=0
    @test filter(removezeros, GSTriangle_data) == filter(removezeros, gen_GSTriangle_data())
end

gen_GSTriangle_data_test()

# segment contraction
GSTriangle_data = gen_GSTriangle_data()
@tensor GSSquare_data[b, c, b′, c′, p, p′] := GSTriangle_data[a, b, c, p] * GSTriangle_data[a, b′, c′, p′]

# 4-plaquette contraction
@tensor ψ[O1N, O1S, O2N, O2S, O3N, O3S, O4N, O4S] := GSSquare_data[E1, N1, S1, W1, O1N, O1S] * GSSquare_data[E2, N2, S2, W2, O2N, O2S] * GSSquare_data[W2, S1, N1, E2, O3N, O3S] * GSSquare_data[W1, S2, N2, E1, O4N, O4S]


