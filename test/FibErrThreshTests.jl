module FibErrThreshTests

using ReTest
using FibErrThresh

include("TensorNetworks/tensornetworktest.jl")
include("TensorNetworks/typedtensornetworktest.jl")
include("TensorNetworks/tobackendtest.jl")

include("indextripletstest.jl")
include("qubitlatticestest.jl")

include("TensorTypes/fibtensortypestest.jl")
include("TensorTypes/segmenttensortypestest.jl")

include("FibTNs/gridutilstest.jl")
include("FibTNs/segmenttest.jl")
include("FibTNs/segmentmaskgridtest.jl")
include("FibTNs/fibtntest.jl")

end # module FibErrThreshTests
