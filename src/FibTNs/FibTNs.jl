module FibTNs

import FibErrThresh: visualize

using ..TensorNetworks
import ..TensorNetworks: add_contraction!
using ..QubitLattices

using ..FibTensorTypes
using ..SegmentTensorTypes

using GeometryBasics
using SparseArrayKit
using Serialization

const IL = IndexLabel
const IC = IndexContraction

const GridPosition = Tuple{Int, Int}
const GridEdge = Tuple{GridPosition, GridPosition}

export GridPosition, GridEdge
export FibTN
export set_leaf!, add_crossings!, add_fusions!, add_contraction!
export visualize, serialize, deserialize

include("segment.jl")
include("fibtn.jl")

function visualize(ftn::FibTN)
    tnf, tnax = visualize(ftn.ttn, ftn.tpos)
end

function visualize(ftn::FibTN, inds::Vector{IndexLabel}, data::SparseArray; interactive=false)
    if interactive
        qlf, qlax = visualize(ql, ipos, inds, data; tail_length=√3/2)
    else
        states, amps = get_states_and_amps(ftn.ql, inds, data)
        qlf, qlaxs = visualize(ql, ipos, states, amps; tail_length=√3/2)
    end
end

serialize(peps::FibTN, inds::Vector{IndexLabel}, data::SparseArray) =
    Serialization.serialize(pwd() * "/out/$(peps.w)x$(peps.h)", (peps, inds, data))

deserialize(w::Int, h::Int) =
    Serialization.deserialize(pwd() * "/out/$(w)x$(h)")

end # module FibTNs
