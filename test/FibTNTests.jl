module FibTNTests

using ReTest
using FibTN

include("tensornetworkstest.jl")
include("tobackendtest.jl")

include("indextripletstest.jl")
include("fibtensortypestest.jl")
include("fibtensornetworkstest.jl")

include("segmentstest.jl")

include("qubitlatticestest.jl")

end # module FibTNTests
