module GridTNs

export GridTN

using ..TensorHandles
using ..TensorNetworks

@enum IndexDirection UP DOWN LEFT RIGHT

struct GridTN{B <: AbstractBackend}
    tn::TensorNetwork{B}
    tensor_at::Dict{Tuple{Int, Int}, TensorHandle{B}}
    index_direction::Dict{IndexLabel, IndexDirection}
    function GridTN{B}(tn) where B <: AbstractBackend
        if length(tn.contractions) != 0 error("GridTN requires instantiating TN to have no contractions") end
        new(tn, Dict(), Dict())
    end
end

function set_position(gtn::GridTN{B}, th::TensorHandle{B}, pos::Tuple{Int, Int}) where B <: AbstractBackend
    if th \notin gtn.tn.tensors error("tensorhandle not in the grid tensor network") end
    if haskey(gtn.tensor_at, pos) error("grid tensor network already has a tensor at position $pos") end
    gtn.tensor_at[pos] = th
end

function set_position(gtn::GridTN{B}, il::IndexLabel, pos::Tuple{Int, Int}) where B <: AbstractBackend
    set_position(gtn, gtn.tn.tensor_with_index(il), pos)
end
    

end # module GridTNs
