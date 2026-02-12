module FibTNTests

using ReTest
using FibTN

include("TensorNetworks/tensornetworktest.jl")
include("TensorNetworks/typedtensornetworktest.jl")
include("TensorNetworks/tobackendtest.jl")

include("indextripletstest.jl")
include("qubitlatticestest.jl")

include("TensorTypes/fibtensortypestest.jl")
include("TensorTypes/segmenttensortypestest.jl")

end # module FibTNTests
