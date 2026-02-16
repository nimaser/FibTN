module SegmentGrids

import FibTN: visualize

using ..TensorNetworks
using ..QubitLattices

using ..FibTensorTypes
using ..SegmentTensorTypes

using GeometryBasics
using SparseArrayKit
using Serialization

const IL = IndexLabel
const IC = IndexContraction

export visualize, serialize, deserialize

include("segment.jl")
include("segmentgrid.jl")

function visualize(sg::SegmentGrid)
    tnf, tnax = visualize(sg.ttn, sg.tpos)
end

function visualize(sg::SegmentGrid, inds::Vector{IndexLabel}, data::SparseArray; interactive=false)
    if interactive
        qlf, qlax = visualize(ql, ipos, inds, data; tail_length=√3/2)
    else
        states, amps = get_states_and_amps(sg.ql, inds, data)
        qlf, qlaxs = visualize(ql, ipos, states, amps; tail_length=√3/2)
    end
end

serialize(sg::SegmentGrid, inds::Vector{IndexLabel}, data::SparseArray) =
    Serialization.serialize(pwd() * "/out/$(sg.w)x$(sg.h)", (sg, inds, data))

deserialize(w::Int, h::Int) =
    Serialization.deserialize(pwd() * "/out/$(w)x$(h)")

end # module SegmentGrids
