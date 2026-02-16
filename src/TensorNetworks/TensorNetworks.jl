module TensorNetworks

# symbolic layer
include("tensornetwork.jl")
include("typedtensornetwork.jl")
# visualizer in GLMakieVisualizationExt.jl

# execution layer
include("backends/tobackend.jl")
include("backends/itbackend.jl")

end # module TensorNetworks
