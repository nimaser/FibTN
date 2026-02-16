module ITBackend

using ITensors
using ITensors: optimal_contraction_sequence
using TensorOperations: TensorOperations
using ..TensorNetworks

export get_itensor_network, optimized_exact_contract

"""
Returns a dictionary mapping `IndexLabel`s in `tn` to `ITensors.Index` objects, and a dictionary mapping
group numbers in `tn` to `ITensors.ITensor` objects. If two `IndexLabel`s are contracted together in `tn`,
they will be mapped to the same `ITensors.Index`. Thus the returned collection of `ITensor`s forms an
immediately contractible tensor network, without any delta tensors being required.
"""
function get_itensor_network(tn::TensorNetwork, tensordata_from_group::Dict{Int, <:AbstractArray})
    # check that every TensorLabel has data, and that that data has the correct dimensions
    for g in get_groups(tn)
        haskey(tensordata_from_group, g) || throw(ArgumentError("missing data for TensorLabel with group $g"))
        tl = get_tensor(tn, g)
        data = tensordata_from_group[g]
        (li = length(tl.indices)) == (nd = ndims(data)) ||
            throw(ArgumentError("number of indices $li differs from array ndims $nd\n Indices: $(tl.indices)"))
    end
    # make itensor network
    itensor_from_id = Dict{Int, ITensor}()
    index_from_indexlabel = Dict{IndexLabel, Index}()
    for g in get_groups(tn)
        tl = get_tensor(tn, g)
        data = tensordata_from_group[g]
        # craete Index objects: for each IndexLabel il in each TensorLabel, make a new ITensors.Index
        # if il is not contracted, or if it is contracted, make a new Index if its partner doesn't
        # have one, and if its partner does have one, use that Index for il as well
        for (il, n) in zip(tl.indices, size(data))
            if has_contraction(tn, il)
                c = get_contraction(tn, il)
                partner_il = get_partner(c, il)
                # make sure contracted IndexLabels share an ITensors.Index
                if haskey(index_from_indexlabel, partner_il)
                    index_from_indexlabel[il] = index_from_indexlabel[partner_il]
                    continue
                end
            end
            index_from_indexlabel[il] = Index(n)
        end
        # make ITensor object
        indices = [index_from_indexlabel[il] for il in tl.indices]
        itensor_from_id[g] = ITensor(SparseArrayDOK(data), indices...)
    end
    index_from_indexlabel, itensor_from_id
end

"""
Wrapper around get_itensor_network for `TensorNetwork`s that automatically fetches
tensordata from tensortypes.
"""
get_itensor_network(ttn::TypedTensorNetwork) =
    get_itensor_network(ttn.tn, tensordata_from_group(ttn))

"""
Uses ITensors' optimization routines (which call TensorOperations) under the
hood to try to reduce the sizes of intermediate tensors.
"""
function optimized_exact_contract(itensors::Vector{ITensor})
    sequence = optimal_contraction_sequence(itensors)
    contract(itensors; sequence=sequence)
end

end # module ITBackend
