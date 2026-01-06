module TensorNetworks

export NodeData, TensorNetwork
export add_tensor, add_contraction, do_contraction

using ..TensorHandles
using Graphs, MetaGraphsNext

struct NodeData
    handle::TensorHandle
    position::Tuple(Float64, Float64)
    color::Symbol
end

struct TensorNetwork
    g::MetaGraph



end

function add_tensor


end



end # module TensorNetworks
