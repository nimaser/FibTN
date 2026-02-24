module GLMakieVisualizationExt

using FibErrThresh
using GLMakie

include("visualizeutils.jl")
include("../src/TensorNetworks/visualizer.jl")
include("../src/QubitLattices/visualizer.jl")
include("../src/FibTNs/visualizer.jl")

end # module GLMakieVisualizationExt
