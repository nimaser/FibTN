module TOBackend

using SparseArrayKit, TensorOperations
using ..TensorNetworks

export ExecutionTensor, ExecutionState, ExecutionStep, execute_step!
export get_ids, get_indices, get_tensors, get_tensor
export ContractionStep, FetchResultStep

"""
Represents a single tensor (in the sense of one numerical array in
memory) in an ExecutionState during numeric tensor manipulation
(execution of contraction and approximation algorithms).

As manipulation proceeds, multiple separate tensors may be combined via
contraction into a single tensor. Likewise, a single tensor may be
decomposed into several. Thus the group id from the symbolic layer is
not the right value to use to identify tensors in the execution layer.
Instead, each ExecutionTensor has a unique numeric id which stays fixed
as long as it is present in the ExecutionState.

The groups of the symbolic tensors which have contributed to this
specific tensor (either by contracting or being decomposed into it) are
stored for debugging and analytics purposes.

Since TensorOperations.jl uses index ordering for contractions, we just
store the numerical data as a SparseArray, and keep the list of indices
corresponding to the data's dimensions. We can then translate any
IndexLabels to a position in a dimension list when contraction needs to
occur.
"""
struct ExecutionTensor
    id::Int
    groups::Set{Int}
    indices::Vector{IndexLabel}
    data::SparseArray
    function ExecutionTensor(id::Int, groups::Set{Int}, indices::Vector{IndexLabel}, data::AbstractArray)
        if (li = length(indices)) != (nd = ndims(data)) error("number of indices $li differs from array ndims $nd\n Indices: $indices") end
        new(id, groups, indices, SparseArray(data))
    end
end

"""
Represents a TensorNetwork's state during numeric tensor manipulation
(execution of contraction and approximation algorithms).

Each tensor (here defined as a single numeric array in memory) is
represented by an ExecutionTensor, which is labeled by a unique id.

The constructor requires a TensorNetwork and a mapping of group number
to numerical data for all of its TensorLabels, which is used to
initialize the ExecutionTensor corresponding (initially, before
manipulation) to each TensorLabel. This data should be permuted such
that its indices correspond with the order of indices in the indices
vector of the TensorLabel.
"""
mutable struct ExecutionState
    tensor_from_id::Dict{Int, ExecutionTensor}
    id_from_index::Dict{IndexLabel, Int}
    _next_id::Int
    function ExecutionState(tn::TensorNetwork, tensordata_from_group::Dict{Int, <: AbstractArray})
        # check that every TensorLabel has data
        for g in get_groups(tn)
            haskey(tensordata_from_group, g) || throw(ArgumentError("missing data for TensorLabel with group $group"))
        end
        # initialize fields
        tensor_from_id = Dict{Int, ExecutionTensor}()
        id_from_index = Dict{IndexLabel, Int}()
        _next_id = 1
        # convert TensorLabels to ExecutionTensors
        for tl in tn.tensors
            et = ExecutionTensor(_next_id,
                                 Set(tl.group),
                                 copy(tl.indices),
                                 tensordata_from_group[tl.group],
                                )
            tensor_from_id[_next_id] = et
            for index in tl.indices id_from_index[index] = _next_id end
            _next_id += 1
        end
        new(tensor_from_id, id_from_index, _next_id)
    end
end

"""Get all ids in this ExecutionState."""
get_ids(es::ExecutionState) =
    keys(es.tensor_from_id)
    
"""Get all indices in this ExecutionState."""
get_indices(es::ExecutionState) =
    keys(es.id_from_index)
    
"""Get all ExecutionTensors resulting from the specified group."""
get_tensors(es::ExecutionState, group::Int) =
    [et for (_, et) in es.tensor_from_id if group ∈ et.groups]
    
"""Get the ExecutionTensor containing the specified IndexLabel."""
get_tensor(es::ExecutionState, il::IndexLabel) =
    es.tensor_from_id[es.id_from_index[il]]

"""
Represents a single numeric tensor manipulation step.

Tensor contractions, tensor decompositions, tensor truncation, and
fetching of numerical results are all concrete subtypes of this type.
"""
abstract type ExecutionStep end

"""Default (unimplemented) implementation of execute_step!"""
function execute_step!(es::ExecutionState, step::ExecutionStep)
    error("$(typeof(step)) not implemented")
end

"""A contraction between two indices."""
struct ContractionStep <: ExecutionStep
    a::IndexLabel
    b::IndexLabel
    ContractionStep(ic::IndexContraction) = new(ic.a, ic.b)
end

"""
Multiple index pair contractions, grouped together to exploit backend
optimizations.
"""
struct MultiContractionStep <: ExecutionStep
    contractions::Vector{ContractionStep}
end

struct QRDecompStep <: ExecutionStep
    # TODO
end

function execute_step!(es::ExecutionState, cs::ContractionStep)
    # make handles to id and exectensor
    ida = es.id_from_index[cs.a]
    idb = es.id_from_index[cs.b]
    eta = es.tensor_from_id[ida]
    etb = es.tensor_from_id[idb]
    # find contracted index positions
    pa = findfirst(==(cs.a), eta.indices)
    pb = findfirst(==(cs.b), etb.indices)
    # determine whether indices are on same exectensor, then use tensortrace or tensorcontract
    if eta === etb
        # build index label lists for tensortrace, all distinct except contracted indices
        IA = collect(1:length(eta.indices))
        IA[pa] = 0
        IA[pb] = 0
        
        # perform trace
        Z = tensortrace(eta.data, IA, false)
        # new indices are the uncontracted ones
        new_indices = deleteat!(copy(eta.indices), pa < pb ? (pa, pb) : (pb, pa))
    else
        # build index label lists for tensorcontract, all distinct except contracted indices
        IA = collect(1:length(eta.indices))
        IB = collect(length(eta.indices)+1:length(eta.indices)+length(etb.indices))
        IA[pa] = 0
        IB[pb] = 0
        
        # perform contraction
        Z = tensorcontract(
            eta.data, IA, false,
            etb.data, IB, false
        )
        # new indices are the uncontracted ones
        new_indices = vcat(
            deleteat!(copy(eta.indices), pa),
            deleteat!(copy(etb.indices), pb)
        )
    end
    # new tensor
    new_id = es._next_id
    new_groups = union(eta.groups, etb.groups)
    etz = ExecutionTensor(new_id, new_groups, new_indices, Z)
    
    # remove old tensors and ids: delete! is idempotent so no issues if we did the trace
    delete!(es.tensor_from_id, ida)
    delete!(es.tensor_from_id, idb)
    for idx in eta.indices delete!(es.id_from_index, idx) end
    for idx in etb.indices delete!(es.id_from_index, idx) end
    # add new tensor and id
    es.tensor_from_id[new_id] = etz
    for idx in new_indices es.id_from_index[idx] = new_id end
    # update overall id counter
    es._next_id += 1
        
    return nothing
end

function execute_step!(es::ExecutionState, mcs::MultiContractionStep)
    # TODO
end

function execute_step(es::ExecutionState, qrd::QRDecompStep)
    # TODO
end

end # module TOBackend
