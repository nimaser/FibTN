# Fibonacci input category data; dim and N, F, R symbols
using TensorKitSectors

function Gsymbol(
        a::FibonacciAnyon, b::FibonacciAnyon, c::FibonacciAnyon,
        d::FibonacciAnyon, e::FibonacciAnyon, f::FibonacciAnyon
    )
    Fsymbol(a, b, c, d, e, f) / √(TensorKitSectors.dim(e)*TensorKitSectors.dim(f))
end
