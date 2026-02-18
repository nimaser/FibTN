module FibTNs

using ..TensorNetworks
using ..QubitLattices

using ..FibTensorTypes
using ..SegmentTensorTypes

using GeometryBasics

const IL = IndexLabel
const IC = IndexContraction

include("gridutils.jl")
include("segment.jl")
include("segmentmaskgrid.jl")
include("fibtn.jl")

# using Serialization
# export serialize, deserialize

# serialize(ftn::FibTN, inds::Vector{IndexLabel}, data::SparseArray) =
#     Serialization.serialize(pwd() * "/out/$(ftn.w)x$(ftn.h)", (ftn, inds, data))

# deserialize(w::Int, h::Int) =
#     Serialization.deserialize(pwd() * "/out/$(w)x$(h)")

end # module FibTNs
