using FibErrThresh: abc2etc, Gsymbol, GSTriangle_data

using TensorKitSectors

@testset "GSTriangle" begin
    debug = false
    if debug
        @show  -1/6, dim(FibonacciAnyon(:τ))^(-1/6)
        @show -1/12, dim(FibonacciAnyon(:τ))^(-1/12)
        @show  1/12, dim(FibonacciAnyon(:τ))^(1/12)
        @show   1/6, dim(FibonacciAnyon(:τ))^(1/6)
        @show  -3/4, dim(FibonacciAnyon(:τ))^(-3/4)
    end

    # nonzero combinations of tensor indices, access as abcvals[:, idx]
    avals = [1 1 2 2 2 3 3 3 4 4 4 5 5 5 5]
    bvals = [1 2 3 4 5 3 4 5 1 2 2 3 4 5 5]
    cvals = [1 4 4 1 4 3 2 5 2 3 5 5 2 3 5]
    abcvals = [avals ; bvals ; cvals]
                                                                                          
    # construct reference array, computed manually
    GSTriangle_ref = zeros(Float64, 5, 5, 5, 5)
    for x in axes(abcvals, 2)
        a, b, c = abcvals[:, x] 
        i, j, k, λ, μ, ν, p = abc2etc(a, b, c)
        GSTriangle_ref[a, b, c, p] = (dim(λ)*dim(μ)*dim(ν))^(1/6) * Gsymbol(i, j, λ, μ, k, ν) * √√(dim(i)*dim(j)*dim(k))
        if debug
            @show a, b, c, p, i, j, k, λ, μ, ν, Gsymbol(i, j, λ, μ, k, ν), (dim(λ)*dim(μ)*dim(ν))^(1/6), √√(dim(i)*dim(j)*dim(k)), GSTriangle_ref[a, b, c, p]
        end
    end
    @test GSTriangle_ref == GSTriangle_data()
end;

