module FibTN

include("TensorNetworks/TensorNetworks.jl")

include("indextriplets.jl")
include("QubitLattices/QubitLattices.jl")

include("TensorTypes/fibtensortypes.jl")
include("TensorTypes/segmenttensortypes.jl")

# this function has methods added to it in ext/GLMakieVisualizationExt.jl
export visualize
function visualize end

end # module FibTN
