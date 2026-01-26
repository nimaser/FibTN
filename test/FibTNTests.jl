module FibTNTests

using ReTest
using FibTN

include("indextripletstest.jl")
include("fibtensortypestest.jl")

include("indicestest.jl")
include("tensornetworkstest.jl")
include("executortest.jl")
include("qubitlatticestest.jl")

include("integrationtests.jl")

include("sparsearrayutilstest.jl")

end # module FibTNTests
