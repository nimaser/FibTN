using ..TensorNetworks
using ..ITensorsBackend
using ..TensorHandles
using ..TensorTypes

plaquette = TensorNetwork()

add_tensor(plaquette, Tensorhandle(ITensorsBackend, tensor_data(Vertex), index_data(Vertex)))

# TODO: write tests for validate_contraction and validate_trace
