module MiscUtils

using ..TensorNetworks
using ..QubitLattices

using SparseArrayKit

export get_states_and_amps

"""
Convert tensor indices and data to a set of states and state amplitudes using the mapping
established by `ql`. A state, in this context, is a mapping from qubit ids to qubit values.

Assumes TOBackend was used to compute the tensor, and so that `data` is a SparseArray.

Returns corresponding lists of states and amps.
"""
function get_states_and_amps(ql::QubitLattice, inds::Vector{IndexLabel}, data::SparseArray)
    states, amps = Vector{Dict{Int, Int}}(), Vector{Real}()
    for (cartesian_idxvals, amp) in nonzero_pairs(data)
        push!(states, idxvals2qubitvals(ql, inds, [Tuple(cartesian_idxvals)...]))
        push!(amps, amp)
    end
    states, amps
end

end # module MiscUtils
