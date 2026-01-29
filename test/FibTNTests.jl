module FibTNTests

using ReTest
using FibTN

include("sparsearrayutilstest.jl")

include("indextripletstest.jl")

include("indicestest.jl")
include("tensornetworkstest.jl")
include("executortest.jl")
include("qubitlatticestest.jl")

include("fibtensortypestest.jl")

end # module FibTNTests
