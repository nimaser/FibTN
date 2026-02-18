module FibErrThresh

# this function has methods added to it in ext/GLMakieVisualizationExt.jl
export visualize
function visualize end

include("TensorNetworks/TensorNetworks.jl")

include("indextriplets.jl")
include("QubitLattices/QubitLattices.jl")

include("TensorTypes/fibtensortypes.jl")
include("TensorTypes/segmenttensortypes.jl")

include("FibTNs/FibTNs.jl")

include("miscutils.jl")

end # module FibErrThresh
