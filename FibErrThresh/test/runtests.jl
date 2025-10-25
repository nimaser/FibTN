# @author Nikhil Maserang
# @date 2025/10/24

using Test
using FibErrThresh

function gen_GSTriangle_ref()
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

    GSTriangle_ref = Array{Float64}(undef, 5, 5, 5, 5)
    for x in axes(abcvals, 2)
        a, b, c = abcvals[:, x] 
        i, j, k, λ, μ, ν, p = abc2etc(a, b, c)
        GSTriangle_ref[a, b, c, p] = Gsymbol(i, j, λ, μ, k, ν) * √√(dim(i)*dim(j)*dim(k))
        # @show a, b, c, p, i, j, k, λ, μ, ν, √√(dim(i)*dim(j)*dim(k))
        # @show a, b, c, p, i, j, k, λ, μ, ν, Fsymbol(i, j, λ, μ, k, ν)
        # @show a, b, c, p, i, j, k, λ, μ, ν, Gsymbol(i, j, λ, μ, k, ν)
        # @show a, b, c, p, i, j, k, λ, μ, ν, GSTriangle_data[a, b, c, ijkidx...]
    end
    GSTriangle_ref
end

@test gen_GSTriangle_ref() == gen_GSTriangle_data()

